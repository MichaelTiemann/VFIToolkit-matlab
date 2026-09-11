function [V_current, Policy_3Row] = ValueFnIter_FHorz_vectorized_GI1_raw(...
    ReturnFn, ReturnFnParamsVec, V_next, a_work, z_work, d_work, ...
    n_a, n_z, n_d, pi_z_j, beta_j, vfoptions)

G = vfoptions.ngridinterp;

% Step weights: tau in [0, (G-1)/G]
tau_vec = linspace(0, (G - 1) / G, G);
if vfoptions.parallel == 2
    tau_vec = gpuArray(tau_vec);
end

% Construct dense interpolated asset choices
% For a_work(end), upper interval clamps to a_work(end)
a_diff = [diff(a_work); 0];
% Shape: (n_a, G)
Apr_dense = a_work + a_diff * tau_vec;
Apr_dense_flat = Apr_dense(:);      % (n_a * G x 1)
n_dense_apr = n_a * G;

% Expected continuation value on coarse grid: (n_a x n_z)
EV_next = V_next * (pi_z_j');

% Linear interpolation of continuation values:
% EV_dense(k, tau, z') = (1 - tau)*EV(k, z') + tau*EV(k+1, z')
EV_next_pad = [EV_next; EV_next(end, :)]; % clamp at boundary
tau_3d = reshape(tau_vec, [1, G, 1]);
EV_dense = (1 - tau_3d) .* reshape(EV_next, [n_a, 1, n_z]) + ...
    tau_3d .* reshape(EV_next_pad(2:end, :), [n_a, 1, n_z]);
% Reshape to (n_dense_apr x n_z)
EV_dense = reshape(permute(EV_dense, [1, 2, 3]), [n_dense_apr, n_z]);

% Canonical 4D evaluation tensor: States (a, z), Choices (d, apr_dense)
[A_m, Z_m, D_m, Apr_m] = ndgrid(a_work, z_work, d_work, Apr_dense_flat);
[~, ~, ~, Apr_dense_idx_m] = ndgrid(1:n_a, 1:n_z, 1:n_d, 1:n_dense_apr);

n_states = n_a * n_z;
n_choices_dense = n_d * n_dense_apr;

% Vectorized ReturnFn evaluation
F_flat = ReturnFn(D_m(:), Apr_m(:), A_m(:), Z_m(:), ReturnFnParamsVec{:});

% Vectorized continuation value lookup
z_idx_state = repelem((1:n_z)', n_a, 1);
z_idx_f = repmat(z_idx_state, n_choices_dense, 1);
lin_idx = sub2ind([n_dense_apr, n_z], Apr_dense_idx_m(:), z_idx_f);
V_cont_flat = EV_dense(lin_idx);

RHS_flat = F_flat + beta_j .* V_cont_flat;
RHS_matrix = reshape(RHS_flat, [n_states, n_choices_dense]);

[V_current, best_choice_idx] = max(RHS_matrix, [], 2);

% Unpack best_choice_idx (1 : n_d * n_a * G)
% Choice dimension ordering: d varies fastest, coarse a' middle, tau slowest
d_opt = mod(best_choice_idx - 1, n_d) + 1;
dense_apr_opt = ceil(best_choice_idx ./ n_d);

coarse_apr_opt = mod(dense_apr_opt - 1, n_a) + 1;
tau_idx_opt = ceil(dense_apr_opt ./ n_a);

% Reconstruct row 1 Kron index: (coarse_apr_opt - 1)*n_d + d_opt
row1_kron = (coarse_apr_opt - 1) .* n_d + d_opt;
row2_tau = tau_idx_opt;
row3_flag = ones(n_states, 1, 'like', a_work);

% Shape: (3, n_a, n_z)
Policy_3Row = zeros(3, n_a, n_z, 'like', a_work);
Policy_3Row(1, :, :) = reshape(row1_kron, [n_a, n_z]);
Policy_3Row(2, :, :) = reshape(row2_tau, [n_a, n_z]);
Policy_3Row(3, :, :) = reshape(row3_flag, [n_a, n_z]);

V_current = reshape(V_current, [n_a, n_z]);

end