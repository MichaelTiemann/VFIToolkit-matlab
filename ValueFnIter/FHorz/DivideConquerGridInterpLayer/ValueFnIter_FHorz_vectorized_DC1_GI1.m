function [V_current, Policy_3Row] = ValueFnIter_FHorz_vectorized_DC1_GI1(...
    eval_kernel, ReturnFnParamsVec, EV, a_work, z_work, d_work, ...
    N_a, N_z, N_d, pi_z_j, beta_j, vfoptions)

G = vfoptions.ngridinterp;
tau_vec = linspace(0, (G - 1) / G, G);
if vfoptions.parallel == 2
    tau_vec = gpuArray(tau_vec);
end

% Dense sub-grid for interpolation: (N_a x G)
a_diff = [diff(a_work); 0];
Apr_dense = a_work + a_diff * tau_vec;

% Dense continuation values: (N_a x G x N_z)
EV_pad = [EV; EV(end, :)];
EV_dense_3d = (1 - tau_vec) .* reshape(EV, [N_a, 1, N_z]) + ...
              tau_vec .* reshape(EV_pad(2:end, :), [N_a, 1, N_z]);

% Check if e exists in the model
has_e = isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0;

% If lowmemory >= 1, the caller loops over e sequentially, so this call only sees 1 slice.
% If lowmemory == 0, this call processes all N_e simultaneously.
if has_e && vfoptions.lowmemory == 0
    N_e = prod(vfoptions.n_e);
    state_dims = [N_a, N_z, N_e];
else
    N_e = 1;
    state_dims = [N_a, N_z];
end

n_states_total = prod(state_dims);

V_current   = zeros(state_dims, 'like', a_work);
Policy_row1 = zeros(state_dims, 'like', a_work);
Policy_row2 = zeros(state_dims, 'like', a_work);

%% =========================================================================
% PASS 1: Coarse Anchors (NO G subgrid, strictly coarse N_a choices)
% =========================================================================
level1ii  = round(linspace(1, N_a, vfoptions.level1n(1)));
n_anchors = length(level1ii);
a_anchors = a_work(level1ii);

% Broadcast shapes (Dim 1: a, Dim 2: z, Dim 4: d, Dim 5: aprime)
A_1   = a_anchors;                % Natively spans Dim 1
Z_1   = shiftdim(z_work(:), -1);  % Pushed to Dim 2
D_1   = shiftdim(d_work(:), -3);  % Pushed to Dim 4
Apr_1 = shiftdim(a_work(:), -4);  % Pushed to Dim 5

F_1 = eval_kernel(D_1, Apr_1, A_1, Z_1);
guard_1 = zeros([n_anchors, N_z, N_e, N_d, N_a], 'like', a_work);
F_1 = F_1 + guard_1;

% EV continuation values on coarse grid: EV is (N_a x N_z) 
EV_broadcast1 = permute(EV, [3, 2, 4, 5, 1]);
RHS_1 = F_1 + beta_j .* EV_broadcast1;

% States: (n_anchors * N_z * N_e)
% Choices: (N_d * N_a)
n_states_1  = n_anchors * N_z * N_e;
n_choices_1 = N_d * N_a;
RHS_m1 = reshape(RHS_1, [n_states_1, n_choices_1]);
[sub_V1, sub_Pol1] = max(RHS_m1, [], 2);

coarse_apr_opt1 = ceil(sub_Pol1 ./ N_d);

if has_e
    opt_coarse_anchors = reshape(coarse_apr_opt1, [n_anchors, N_z, N_e]);
    V_current(level1ii, :, :)   = reshape(sub_V1, [n_anchors, N_z, N_e]);
    Policy_row1(level1ii, :, :) = reshape(sub_Pol1, [n_anchors, N_z, N_e]);
    Policy_row2(level1ii, :, :) = ones(n_anchors, N_z, N_e, 'like', a_work);
else
    opt_coarse_anchors = reshape(coarse_apr_opt1, [n_anchors, N_z]);
    V_current(level1ii, :)   = reshape(sub_V1, [n_anchors, N_z]);
    Policy_row1(level1ii, :) = reshape(sub_Pol1, [n_anchors, N_z]);
    Policy_row2(level1ii, :) = ones(n_anchors, N_z, 'like', a_work);
end

%% =========================================================================
% PASS 2: Bounded Monotonic Intervals (Vectorized per bin)
% =========================================================================
D_2         = shiftdim(d_work(:), -3);    % Dim 4
Z_2         = shiftdim(z_work(:), -1);    % Dim 2
z_sub_idx   = shiftdim((1:N_z)',  -1);    % Dim 2
tau_sub_idx = shiftdim((1:G)',    -5);    % Dim 6

for bin = 1:(n_anchors - 1)
    idx_start = level1ii(bin) + 1;
    idx_end   = level1ii(bin + 1) - 1;
    if idx_start > idx_end
        continue;
    end
    
    bin_a_idx = idx_start:idx_end;
    n_bin_a   = length(bin_a_idx);
    a_bin     = a_work(bin_a_idx);
    
    if has_e
        lb_bin = opt_coarse_anchors(bin, :, :);     % (1 x N_z x N_e)
        ub_bin = opt_coarse_anchors(bin + 1, :, :); % (1 x N_z x N_e)
    else
        lb_bin = opt_coarse_anchors(bin, :);        % (1 x N_z)
        ub_bin = opt_coarse_anchors(bin + 1, :);    % (1 x N_z)
    end

    lb_bin_pad = max(1, lb_bin - 1);
    ub_bin_pad = min(N_a, ub_bin + 1);
    
    maxgap_bin = max(ub_bin_pad(:) - lb_bin_pad(:));
    n_cand_bin = maxgap_bin + 1;
    
    % k_offsets pushed to Dim 5
    k_offsets = shiftdim((0:maxgap_bin)', -4); 
    
    % MATLAB broadcasts [1, N_z, N_e] + [1, 1, 1, 1, n_cand_bin] natively
    coarse_cand_idx = min(lb_bin_pad + k_offsets, ub_bin_pad);

    % Candidate assets on subgrid
    apr_cand_lin = coarse_cand_idx + (tau_sub_idx - 1) .* N_a;
    Apr_val_bin  = Apr_dense(apr_cand_lin);

    A_bin = a_bin; % Natively spans Dim 1

    F_bin = eval_kernel(D_2, Apr_val_bin, A_bin, Z_2);
    guard_bin  = zeros([n_bin_a, N_z, N_e, N_d, n_cand_bin, G], 'like', a_work);
    F_bin      = F_bin + guard_bin;

    % Continuation values
    ev_cand_lin = (coarse_cand_idx - 1) + (tau_sub_idx - 1) .* N_a + ...
                  (z_sub_idx - 1) .* (N_a * G) + 1;
    V_cont_bin  = EV_dense_3d(ev_cand_lin);

    RHS_bin = F_bin + beta_j .* V_cont_bin;

    n_states_bin  = n_bin_a * N_z * N_e;
    n_choices_bin = N_d * n_cand_bin * G;
    RHS_m_bin     = reshape(RHS_bin, [n_states_bin, n_choices_bin]);
    [sub_V_bin, sub_Pol_bin] = max(RHS_m_bin, [], 2);

    % Unpack choices
    d_chosen          = mod(sub_Pol_bin - 1, N_d) + 1;
    cand_tau_opt      = ceil(sub_Pol_bin ./ N_d);
    coarse_offset_opt = mod(cand_tau_opt - 1, n_cand_bin) + 1;
    tau_idx_opt       = ceil(cand_tau_opt ./ n_cand_bin);

    coarse_cand_expanded = repmat(coarse_cand_idx, [n_bin_a, 1, 1, 1, 1]);
    coarse_cand_2d = reshape(coarse_cand_expanded, [n_states_bin, n_cand_bin]);
    state_lin = (1:n_states_bin)';
    chosen_lin = sub2ind([n_states_bin, n_cand_bin], state_lin, coarse_offset_opt);
    coarse_apr_bin = coarse_cand_2d(chosen_lin);

    % Boundary clamp at upper bound
    at_upper = (coarse_apr_bin == N_a);
    tau_idx_opt(at_upper) = 1;

    row1_kron_bin = (coarse_apr_bin - 1) .* N_d + d_chosen;

    if has_e
        V_current(bin_a_idx, :, :)   = reshape(sub_V_bin, [n_bin_a, N_z, N_e]);
        Policy_row1(bin_a_idx, :, :) = reshape(row1_kron_bin, [n_bin_a, N_z, N_e]);
        Policy_row2(bin_a_idx, :, :) = reshape(tau_idx_opt, [n_bin_a, N_z, N_e]);
    else
        V_current(bin_a_idx, :)   = reshape(sub_V_bin, [n_bin_a, N_z]);
        Policy_row1(bin_a_idx, :) = reshape(row1_kron_bin, [n_bin_a, N_z]);
        Policy_row2(bin_a_idx, :) = reshape(tau_idx_opt, [n_bin_a, N_z]);
    end
end

if has_e
    Policy_3Row = zeros([3, N_a, N_z, N_e], 'like', a_work);
    Policy_3Row(1, :, :, :) = Policy_row1;
    Policy_3Row(2, :, :, :) = Policy_row2;
    Policy_3Row(3, :, :, :) = ones(N_a, N_z, N_e, 'like', a_work);
else
    Policy_3Row = zeros([3, N_a, N_z], 'like', a_work);
    Policy_3Row(1, :, :) = Policy_row1;
    Policy_3Row(2, :, :) = Policy_row2;
    Policy_3Row(3, :, :) = ones(N_a, N_z, 'like', a_work);
end

end