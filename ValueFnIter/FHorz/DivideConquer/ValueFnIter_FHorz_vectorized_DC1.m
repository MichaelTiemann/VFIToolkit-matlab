function [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_DC1(...
    eval_func, V_next, a_work, z_work, d_work, n_a, n_z, n_d, pi_z_j, beta_j, vfoptions)

% Expected continuation value: (n_a x n_z)
EV_next = V_next * (pi_z_j');

V_current = zeros(n_a, n_z, 'like', a_work);
Policy_Indices = zeros(n_a, n_z, 'like', a_work);

%% Pass 1: Solve Coarse Anchor States
level1ii = round(linspace(1, n_a, vfoptions.level1n));
n_anchors = length(level1ii);
a_anchors = a_work(level1ii);

% Canonical grid for anchors: ndgrid(a_anchors, z, d, aprime)
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

% Extract optimal aprime index: d varies fastest, aprime slowest
% Shape: (n_anchors, n_z)
opt_apr_anchors = ceil(reshape(sub_Pol1, [n_anchors, n_z]) ./ n_d);

%% Pass 2: Interval-by-Interval Bounded Refinement
for ii = 1:(n_anchors - 1)
    curraindex = (level1ii(ii) + 1):(level1ii(ii + 1) - 1);
    n_sub = length(curraindex);
    if n_sub == 0, continue; end
    
    sub_a = a_work(curraindex);
    
    % Monotonic bounds across intervals
    lb_z = opt_apr_anchors(ii, :);       % (1 x n_z)
    ub_z = opt_apr_anchors(ii + 1, :);   % (1 x n_z)
    
    % Span bounds to determine candidate window
    min_apr = min(lb_z);
    max_apr = max(ub_z);
    apr_candidates_idx = min_apr:max_apr;
    apr_candidates = a_work(apr_candidates_idx);
    n_cand = length(apr_candidates_idx);
    
    % Narrow tensor: (n_sub x n_z x n_d x n_cand)
    [A_m2, Z_m2, D_m2, Apr_m2] = ndgrid(sub_a, z_work, d_work, apr_candidates);
    [~, ~, ~, Cand_idx_m2] = ndgrid(1:n_sub, 1:n_z, 1:n_d, 1:n_cand);
    
    Apr_idx_f2 = apr_candidates_idx(Cand_idx_m2(:))';
    
    F_f2 = eval_func(D_m2(:), Apr_m2(:), A_m2(:), Z_m2(:));
    
    z_idx_sub = repelem((1:n_z)', n_sub, 1);
    z_idx_f2 = repmat(z_idx_sub, n_d * n_cand, 1);
    lin_idx2 = sub2ind([n_a, n_z], Apr_idx_f2, z_idx_f2);
    V_cont_f2 = EV_next(lin_idx2);
    
    RHS_f2 = F_f2 + beta_j .* V_cont_f2;
    
    % Enforce per-shock bounds
    lb_expanded = repmat(repelem(lb_z(:), n_sub, 1), n_d * n_cand, 1);
    ub_expanded = repmat(repelem(ub_z(:), n_sub, 1), n_d * n_cand, 1);
    
    invalid = (Apr_idx_f2 < lb_expanded) | (Apr_idx_f2 > ub_expanded);
    RHS_f2(invalid) = -Inf;
    
    RHS_m2 = reshape(RHS_f2, [n_sub * n_z, n_d * n_cand]);
    [sub_V2, sub_Pol2] = max(RHS_m2, [], 2);
    
    % Reconstruct global Kron choice index: (aprime_idx - 1)*n_d + d_idx
    d_chosen = mod(sub_Pol2 - 1, n_d) + 1;
    cand_chosen = ceil(sub_Pol2 ./ n_d);
    apr_chosen_global = apr_candidates_idx(cand_chosen)';
    global_Pol2 = (apr_chosen_global - 1) .* n_d + d_chosen;
    
    V_current(curraindex, :) = reshape(sub_V2, [n_sub, n_z]);
    Policy_Indices(curraindex, :) = reshape(global_Pol2, [n_sub, n_z]);
end

V_current = V_current(:);
Policy_Indices = Policy_Indices(:);

end
