function [V_current, Policy_3Row] = ValueFnIter_FHorz_vectorized_DC1_GI1(...
    eval_kernel, ReturnFnParamsVec, EV, a_work, z_work, d_work, ...
    n_a, n_z, n_d, pi_z_j, beta_j, vfoptions)

G = vfoptions.ngridinterp;

tau_vec = linspace(0, (G - 1) / G, G);
if vfoptions.parallel == 2
    tau_vec = gpuArray(tau_vec);
end

% Dense sub-grid for interpolation: (n_a x G)
a_diff = [diff(a_work); 0];
Apr_dense = a_work + a_diff * tau_vec;

% Dense continuation values: (n_a x G x n_z)
EV_pad = [EV; EV(end, :)];
tau_3d = reshape(tau_vec, [1, G, 1]);
EV_dense_3d = (1 - tau_3d) .* reshape(EV, [n_a, 1, n_z]) + ...
              tau_3d .* reshape(EV_pad(2:end, :), [n_a, 1, n_z]);

% Determine if this subproblem instance is solving e simultaneously:
has_e_simul = isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0 && ...
    (~isfield(vfoptions, 'lowmemory') || vfoptions.lowmemory == 0);

if has_e_simul
    n_e = prod(vfoptions.n_e);
    state_dims = [n_a, n_z, n_e];
else
    n_e = 1;
    state_dims = [n_a, n_z];
end

n_states_total = prod(state_dims);

V_current   = zeros(state_dims, 'like', a_work);
Policy_row1 = zeros(state_dims, 'like', a_work);
Policy_row2 = zeros(state_dims, 'like', a_work);

%% =========================================================================
% PASS 1: Coarse Anchors (NO G subgrid, strictly coarse n_a choices)
% Memory footprint drops by factor of G (20x)
% =========================================================================
level1ii  = round(linspace(1, n_a, vfoptions.level1n));
n_anchors = length(level1ii);
a_anchors = a_work(level1ii);

% Broadcast shapes (strictly 4D: n_anchors x n_z x n_d x n_a):
% a:   (n_anchors, 1,   1,   1,   1)
% z:   (1,         n_z, 1,   1,   1)
% d:   (1,         1,   1,   n_d, 1)
% apr: (1,         1,   1,   1,   n_a)  <-- coarse only!
A_1   = reshape(a_anchors, [n_anchors, 1,   1,   1,   1]);
Z_1   = reshape(z_work,    [1,         n_z, 1,   1,   1]);
% Save room for n_e
D_1   = reshape(d_work,    [1,         1,   1,   n_d, 1]);
Apr_1 = reshape(a_work,    [1,         1,   1,   1,   n_a]);

F_1 = eval_kernel(D_1, Apr_1, A_1, Z_1);

guard_1 = zeros([n_anchors, n_z, n_e, n_d, n_a], 'like', a_work);
F_1 = F_1 + guard_1;

% EV continuation values on coarse grid: EV is (n_a x n_z) -> align with (1, n_z, 1, 1, n_a)
EV_broadcast1 = permute(EV, [3, 2, 4, 5, 1]);

RHS_1 = F_1 + beta_j .* EV_broadcast1;

% States: (n_anchors * n_z * n_e)
% Choices: (n_d * n_a)
n_states_1  = n_anchors * n_z * n_e;
n_choices_1 = n_d * n_a;
RHS_m1 = reshape(RHS_1, [n_states_1, n_choices_1]);
[sub_V1, sub_Pol1] = max(RHS_m1, [], 2);

% Choice unpacking: d varies fastest, coarse_apr varies slower
coarse_apr_opt1 = ceil(sub_Pol1 ./ n_d);

if has_e_simul
    opt_coarse_anchors = reshape(coarse_apr_opt1, [n_anchors, n_z, n_e]);
    V_current(level1ii, :, :)   = reshape(sub_V1, [n_anchors, n_z, n_e]);
    Policy_row1(level1ii, :, :) = reshape(sub_Pol1, [n_anchors, n_z, n_e]);
    Policy_row2(level1ii, :, :) = ones(n_anchors, n_z, n_e, 'like', a_work);
else
    opt_coarse_anchors = reshape(coarse_apr_opt1, [n_anchors, n_z]);
    V_current(level1ii, :)   = reshape(sub_V1, [n_anchors, n_z]);
    Policy_row1(level1ii, :) = reshape(sub_Pol1, [n_anchors, n_z]);
    Policy_row2(level1ii, :) = ones(n_anchors, n_z, 'like', a_work);
end

%% =========================================================================
% PASS 2: Bounded Monotonic Intervals (Vectorized per bin)
% Peak memory per bin: <= 110 MB (prevents Pass 2 multi-GB blowup)
% =========================================================================
D_2 = reshape(d_work, [1, 1, 1, n_d, 1, 1]);
Z_2 = reshape(z_work, [1, n_z, 1, 1, 1, 1]);
z_sub_idx   = reshape(1:n_z, [1, n_z, 1, 1, 1, 1]);
tau_sub_idx = reshape(1:G, [1, 1, 1, 1, 1, G]);

for bin = 1:(n_anchors - 1)
    % Interior asset indices between anchor(bin) and anchor(bin+1)
    idx_start = level1ii(bin) + 1;
    idx_end   = level1ii(bin + 1) - 1;
    if idx_start > idx_end
        continue;
    end
    
    bin_a_idx = idx_start:idx_end;
    n_bin_a   = length(bin_a_idx);
    a_bin     = a_work(bin_a_idx);
    
    % Monotonic bounds established by Pass 1 anchors
    if has_e_simul
        lb_bin = opt_coarse_anchors(bin, :, :);     % (1 x n_z x n_e)
        ub_bin = opt_coarse_anchors(bin + 1, :, :); % (1 x n_z x n_e)
    else
        lb_bin = opt_coarse_anchors(bin, :);        % (1 x n_z)
        ub_bin = opt_coarse_anchors(bin + 1, :);    % (1 x n_z)
    end
    
    lb_bin_pad = max(1, lb_bin - 1);
    ub_bin_pad = min(n_a, ub_bin + 1);
    
    maxgap_bin = max(ub_bin_pad(:) - lb_bin_pad(:));
    n_cand_bin = maxgap_bin + 1;
    
    % Candidates span dimension 5
    k_offsets = reshape(0:maxgap_bin, [1, 1, 1, 1, n_cand_bin, 1]);
    
    coarse_cand_idx = min(reshape(lb_bin_pad, [1, n_z, n_e, 1, 1, 1]) + k_offsets, ...
                          reshape(ub_bin_pad, [1, n_z, n_e, 1, 1, 1]));

    % Candidate assets on subgrid: shape (1, n_z, 1, 1, n_cand_bin, G)
    apr_cand_lin = coarse_cand_idx + (tau_sub_idx - 1) .* n_a;
    Apr_val_bin  = Apr_dense(apr_cand_lin);

    A_bin = reshape(a_bin, [n_bin_a, 1, 1, 1, 1, 1]);

    % Evaluation across 6D broadcast: (a, z, e, d, cand, G)
    F_bin = eval_kernel(D_2, Apr_val_bin, A_bin, Z_2);
    
    guard_bin  = zeros([n_bin_a, n_z, n_e, n_d, n_cand_bin, G], 'like', a_work);
    F_bin      = F_bin + guard_bin;

    % Continuation values: EV_dense_3d has shape (n_a, G, n_z)
    ev_cand_lin = (coarse_cand_idx - 1) + (tau_sub_idx - 1) .* n_a + ...
                  (z_sub_idx - 1) .* (n_a * G) + 1;
    V_cont_bin  = EV_dense_3d(ev_cand_lin);

    RHS_bin = F_bin + beta_j .* V_cont_bin;

    % Fold states: (n_bin_a * n_z * n_e)
    % Fold choices: (n_d * n_cand_bin * G)
    n_states_bin  = n_bin_a * n_z * n_e;
    n_choices_bin = n_d * n_cand_bin * G;
    RHS_m_bin     = reshape(RHS_bin, [n_states_bin, n_choices_bin]);
    [sub_V_bin, sub_Pol_bin] = max(RHS_m_bin, [], 2);
    
    % Unpack choices
    d_chosen     = mod(sub_Pol_bin - 1, n_d) + 1;
    cand_tau_opt = ceil(sub_Pol_bin ./ n_d);
    
    coarse_offset_opt = mod(cand_tau_opt - 1, n_cand_bin) + 1;
    tau_idx_opt       = ceil(cand_tau_opt ./ n_cand_bin);
    
    % Map candidate offset back to global coarse asset index
    % coarse_cand_idx is (1, n_z, 1, n_cand_bin, 1) -> expand across n_bin_a
    coarse_cand_expanded = repmat(coarse_cand_idx, [n_bin_a, 1, 1, 1, 1]);
    coarse_cand_2d = reshape(coarse_cand_expanded, [n_states_bin, n_cand_bin]);
    
    state_lin = (1:n_states_bin)';
    chosen_lin = sub2ind([n_states_bin, n_cand_bin], state_lin, coarse_offset_opt);
    coarse_apr_bin = coarse_cand_2d(chosen_lin);
    
    % Boundary clamp at upper bound
    at_upper = (coarse_apr_bin == n_a);
    tau_idx_opt(at_upper) = 1;
    
    row1_kron_bin = (coarse_apr_bin - 1) .* n_d + d_chosen;
    
    if has_e_simul
        V_current(bin_a_idx, :, :)   = reshape(sub_V_bin, [n_bin_a, n_z, n_e]);
        Policy_row1(bin_a_idx, :, :) = reshape(row1_kron_bin, [n_bin_a, n_z, n_e]);
        Policy_row2(bin_a_idx, :, :) = reshape(tau_idx_opt, [n_bin_a, n_z, n_e]);
    else
        V_current(bin_a_idx, :)   = reshape(sub_V_bin, [n_bin_a, n_z]);
        Policy_row1(bin_a_idx, :) = reshape(row1_kron_bin, [n_bin_a, n_z]);
        Policy_row2(bin_a_idx, :) = reshape(tau_idx_opt, [n_bin_a, n_z]);
    end
end

if has_e_simul
    Policy_3Row = zeros([3, n_a, n_z, n_e], 'like', a_work);
    Policy_3Row(1, :, :, :) = Policy_row1;
    Policy_3Row(2, :, :, :) = Policy_row2;
    Policy_3Row(3, :, :, :) = ones(n_a, n_z, n_e, 'like', a_work);
else
    Policy_3Row = zeros([3, n_a, n_z], 'like', a_work);
    Policy_3Row(1, :, :) = Policy_row1;
    Policy_3Row(2, :, :) = Policy_row2;
    Policy_3Row(3, :, :) = ones(n_a, n_z, 'like', a_work);
end

end