function [V_current, Policy_3Row] = ValueFnIter_FHorz_vectorized_DC1_GI1(...
    eval_kernel, ReturnFnParamsVec, EV, a_work, z_work, d_work, ...
    n_a, n_z, n_d, pi_z_j, beta_j, vfoptions)

G = vfoptions.ngridinterp;

tau_vec = linspace(0, (G - 1) / G, G);
if vfoptions.parallel == 2
    tau_vec = gpuArray(tau_vec);
end

% Dense interpolated asset choices: (n_a x G)
a_diff = [diff(a_work); 0];
Apr_dense = a_work + a_diff * tau_vec;

% EV is pre-integrated upstream, shape: (n_a x n_z)
EV_pad = [EV; EV(end, :)];
tau_3d = reshape(tau_vec, [1, G, 1]);
EV_dense_3d = (1 - tau_3d) .* reshape(EV, [n_a, 1, n_z]) + ...
              tau_3d .* reshape(EV_pad(2:end, :), [n_a, 1, n_z]);

V_current   = zeros(n_a, n_z, 'like', a_work);
Policy_row1 = zeros(n_a, n_z, 'like', a_work);
Policy_row2 = zeros(n_a, n_z, 'like', a_work);

%% Pass 1: Solve Coarse Anchor States via Broadcasting
level1ii  = round(linspace(1, n_a, vfoptions.level1n));
n_anchors = length(level1ii);
a_anchors = a_work(level1ii);

% Broadcast shapes:
% a:   (n_anchors, 1,   1,   1,   1)
% z:   (1,         n_z, 1,   1,   1)
% d:   (1,         1,   n_d, 1,   1)
% apr: (1,         1,   1,   n_a, G)
A_1   = reshape(a_anchors, [n_anchors, 1,   1,   1,   1]);
Z_1   = reshape(z_work,    [1,         n_z, 1,   1,   1]);
D_1   = reshape(d_work,    [1,         1,   n_d, 1,   1]);
Apr_1 = reshape(Apr_dense, [1,         1,   1,   n_a, G]);

F_1 = eval_kernel(D_1, Apr_1, A_1, Z_1);

guard_1 = zeros([n_anchors, n_z, n_d, n_a, G], 'like', a_work);
F_1 = F_1 + guard_1;

% EV_dense_3d is (n_a, G, n_z) -> align with (1, n_z, 1, n_a, G)
EV_broadcast1 = permute(EV_dense_3d, [4, 3, 5, 1, 2]);

RHS_1 = F_1 + beta_j .* EV_broadcast1;

% Fold states into rows, choices into columns
n_states_1  = n_anchors * n_z;
n_choices_1 = n_d * n_a * G;
RHS_m1 = reshape(RHS_1, [n_states_1, n_choices_1]);
[sub_V1, sub_Pol1] = max(RHS_m1, [], 2);

% Unpack choices: d fastest, coarse_apr middle, tau slowest
d_opt1       = mod(sub_Pol1 - 1, n_d) + 1;
apr_tau_opt1 = ceil(sub_Pol1 ./ n_d);

coarse_apr_opt1 = mod(apr_tau_opt1 - 1, n_a) + 1;
tau_idx_opt1    = ceil(apr_tau_opt1 ./ n_a);

at_upper1 = (coarse_apr_opt1 == n_a);
tau_idx_opt1(at_upper1) = 1;

row1_kron1 = (coarse_apr_opt1 - 1) .* n_d + d_opt1;

V_current(level1ii, :)   = reshape(sub_V1, [n_anchors, n_z]);
Policy_row1(level1ii, :) = reshape(row1_kron1, [n_anchors, n_z]);
Policy_row2(level1ii, :) = reshape(tau_idx_opt1, [n_anchors, n_z]);

opt_coarse_anchors = reshape(coarse_apr_opt1, [n_anchors, n_z]);

%% Pass 2: Vectorized Batch Evaluation via Broadcasting
rem_mask = true(n_a, 1);
rem_mask(level1ii) = false;
rem_a_idx = find(rem_mask);
n_rem = length(rem_a_idx);

if n_rem > 0
    rem_a = a_work(rem_a_idx);
    
    bin_idx = discretize(rem_a_idx, level1ii);
    lb_rem  = opt_coarse_anchors(bin_idx, :);     % (n_rem x n_z)
    ub_rem  = opt_coarse_anchors(bin_idx + 1, :); % (n_rem x n_z)
    
    maxgap = max(ub_rem(:) - lb_rem(:));
    n_cand_coarse = maxgap + 1;
    
    k_offsets = reshape(0:maxgap, [1, 1, 1, n_cand_coarse, 1]);
    
    coarse_cand_idx = min(reshape(lb_rem, [n_rem, n_z, 1, 1, 1]) + k_offsets, ...
                          reshape(ub_rem, [n_rem, n_z, 1, 1, 1]));
    
    tau_sub_idx = reshape(1:G, [1, 1, 1, 1, G]);
    
    apr_cand_lin = coarse_cand_idx + (tau_sub_idx - 1) .* n_a;
    Apr_val_5d   = Apr_dense(apr_cand_lin); % (n_rem, n_z, 1, n_cand_coarse, G)
    
    A_2 = reshape(rem_a,  [n_rem, 1,   1,   1,   1]);
    Z_2 = reshape(z_work, [1,     n_z, 1,   1,   1]);
    D_2 = reshape(d_work, [1,     1,   n_d, 1,   1]);
    
    F_2 = eval_kernel(D_2, Apr_val_5d, A_2, Z_2);
    
    guard_2 = zeros([n_rem, n_z, n_d, n_cand_coarse, G], 'like', a_work);
    F_2 = F_2 + guard_2;
    
    z_sub_idx = reshape(1:n_z, [1, n_z, 1, 1, 1]);
    ev_cand_lin = (coarse_cand_idx - 1) + (tau_sub_idx - 1) .* n_a + ...
                  (z_sub_idx - 1) .* (n_a * G) + 1;
    V_cont_5d = EV_dense_3d(ev_cand_lin);
    
    RHS_2 = F_2 + beta_j .* V_cont_5d;
    
    n_states_2  = n_rem * n_z;
    n_choices_2 = n_d * n_cand_coarse * G;
    RHS_m2 = reshape(RHS_2, [n_states_2, n_choices_2]);
    [sub_V2, sub_Pol2] = max(RHS_m2, [], 2);
    
    % Unpack choices
    d_chosen2     = mod(sub_Pol2 - 1, n_d) + 1;
    cand_tau_opt2 = ceil(sub_Pol2 ./ n_d);
    
    coarse_offset_opt2 = mod(cand_tau_opt2 - 1, n_cand_coarse) + 1;
    tau_idx_opt2       = ceil(cand_tau_opt2 ./ n_cand_coarse);
    
    coarse_cand_2d = reshape(coarse_cand_idx, [n_states_2, n_cand_coarse]);
    state_lin2 = (1:n_states_2)';
    chosen_coarse_lin = sub2ind([n_states_2, n_cand_coarse], state_lin2, coarse_offset_opt2);
    coarse_apr_opt2 = coarse_cand_2d(chosen_coarse_lin);
    
    at_upper2 = (coarse_apr_opt2 == n_a);
    tau_idx_opt2(at_upper2) = 1;
    
    row1_kron2 = (coarse_apr_opt2 - 1) .* n_d + d_chosen2;
    
    V_current(rem_a_idx, :)   = reshape(sub_V2, [n_rem, n_z]);
    Policy_row1(rem_a_idx, :) = reshape(row1_kron2, [n_rem, n_z]);
    Policy_row2(rem_a_idx, :) = reshape(tau_idx_opt2, [n_rem, n_z]);
end

Policy_3Row = zeros(3, n_a, n_z, 'like', a_work);
Policy_3Row(1, :, :) = Policy_row1;
Policy_3Row(2, :, :) = Policy_row2;
Policy_3Row(3, :, :) = ones(n_a, n_z, 'like', a_work);

end
