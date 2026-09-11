function [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_DC1(...
    eval_func, V_next, a_work, z_work, d_work, n_a, n_z, n_d, pi_z_j, beta_j, vfoptions)

% Expected continuation value: (n_a x n_z)
EV_next = V_next * (pi_z_j');

V_current = zeros(n_a, n_z, 'like', a_work);
Policy_Indices = zeros(n_a, n_z, 'like', a_work);

%% Pass 1: Solve Coarse Anchor States in Parallel
level1ii = round(linspace(1, n_a, vfoptions.level1n));
n_anchors = length(level1ii);
a_anchors = a_work(level1ii);

[A_m1, Z_m1, D_m1, Apr_m1] = ndgrid(a_anchors, z_work, d_work, a_work);
[~, ~, ~, Apr_idx_m1] = ndgrid(1:n_anchors, 1:n_z, 1:n_d, 1:n_a);

F_f1 = eval_func(D_m1(:), Apr_m1(:), A_m1(:), Z_m1(:));

z_idx_anchor = repelem((1:n_z)', n_anchors, 1);
z_idx_f1 = repmat(z_idx_anchor, n_d * n_a, 1);
lin_idx1 = sub2ind([n_a, n_z], Apr_idx_m1(:), z_idx_f1);
V_cont_f1 = EV_next(lin_idx1);

RHS_m1 = reshape(F_f1 + beta_j .* V_cont_f1, [n_anchors * n_z, n_d * n_a]);
[sub_V1, sub_Pol1] = max(RHS_m1, [], 2);

V_current(level1ii, :) = reshape(sub_V1, [n_anchors, n_z]);
Policy_Indices(level1ii, :) = reshape(sub_Pol1, [n_anchors, n_z]);

% Optimal aprime index for anchors (d varies fastest, aprime slowest)
opt_apr_anchors = ceil(reshape(sub_Pol1, [n_anchors, n_z]) ./ n_d);

%% Pass 2: Vectorized Batch Evaluation of All Remaining States
rem_mask = true(n_a, 1);
rem_mask(level1ii) = false;
rem_a_idx = find(rem_mask);
n_rem = length(rem_a_idx);

if n_rem > 0
    rem_a = a_work(rem_a_idx);
    
    % Map remaining states to their left and right anchor interval indices
    % discretize assigns each index to bin ii such that level1ii(ii) <= idx <= level1ii(ii+1)
    bin_idx = discretize(rem_a_idx, level1ii);
    
    % Extract lower and upper bounds for each remaining state and shock: shape (n_rem, n_z)
    lb_rem = opt_apr_anchors(bin_idx, :);
    ub_rem = opt_apr_anchors(bin_idx + 1, :);
    
    maxgap = max(ub_rem(:) - lb_rem(:));
    n_cand = maxgap + 1;
    k_offsets = reshape(0:maxgap, [1, 1, 1, n_cand]);
    
    % Bounded candidates: shape (n_rem, n_z, 1, n_cand)
    Apr_idx_4d = min(reshape(lb_rem, [n_rem, n_z, 1, 1]) + k_offsets, ...
                     reshape(ub_rem, [n_rem, n_z, 1, 1]));
    
    % Expand over d: shape (n_rem, n_z, n_d, n_cand)
    Apr_idx_m2 = repmat(Apr_idx_4d, [1, 1, n_d, 1]);
    Apr_m2 = a_work(Apr_idx_m2);
    
    [A_m2, Z_m2, D_m2] = ndgrid(rem_a, z_work, d_work);
    A_m2 = repmat(A_m2, [1, 1, 1, n_cand]);
    Z_m2 = repmat(Z_m2, [1, 1, 1, n_cand]);
    D_m2 = repmat(D_m2, [1, 1, 1, n_cand]);
    
    Apr_idx_f2 = Apr_idx_m2(:);
    F_f2 = eval_func(D_m2(:), Apr_m2(:), A_m2(:), Z_m2(:));
    
    z_idx_rem = repelem((1:n_z)', n_rem, 1);
    z_idx_f2 = repmat(z_idx_rem, n_d * n_cand, 1);
    lin_idx2 = sub2ind([n_a, n_z], Apr_idx_f2, z_idx_f2);
    V_cont_f2 = EV_next(lin_idx2);
    
    RHS_m2 = reshape(F_f2 + beta_j .* V_cont_f2, [n_rem * n_z, n_d * n_cand]);
    [sub_V2, sub_Pol2] = max(RHS_m2, [], 2);
    
    % Extract chosen d and candidate offset
    d_chosen = mod(sub_Pol2 - 1, n_d) + 1;
    cand_chosen = ceil(sub_Pol2 ./ n_d);
    
    % Reconstruct global aprime index from Apr_idx_4d
    % Apr_idx_4d has shape (n_rem, n_z, 1, n_cand)
    Apr_idx_flat_map = reshape(Apr_idx_4d, [n_rem * n_z, n_cand]);
    state_idx = (1:(n_rem * n_z))';
    chosen_lin = sub2ind([n_rem * n_z, n_cand], state_idx, cand_chosen);
    apr_chosen_global = Apr_idx_flat_map(chosen_lin);
    
    global_Pol2 = (apr_chosen_global - 1) .* n_d + d_chosen;
    
    V_current(rem_a_idx, :) = reshape(sub_V2, [n_rem, n_z]);
    Policy_Indices(rem_a_idx, :) = reshape(global_Pol2, [n_rem, n_z]);
end

V_current = V_current(:);
Policy_Indices = Policy_Indices(:);

end