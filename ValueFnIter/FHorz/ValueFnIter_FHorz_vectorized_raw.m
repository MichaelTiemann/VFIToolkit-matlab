function [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_raw(eval_func, BellmanCombiner, V_next, A_flat, Aprime_flat, AprimeIdx_flat, n_states, n_choices, n_a, n_z, pi_z_j, vfoptions)

F_flat = eval_func(Aprime_flat, A_flat);

% Expected continuation value over tomorrow's shocks (z'):
% V_next is (n_a x n_z), pi_z_j is (n_z x n_z) -> EV_next is (n_a x n_z)
EV_next = V_next * (pi_z_j');

% Map the flattened choice grid (AprimeIdx_flat) and state shocks (z_idx_vec)
% Canonical grid ordering: ndgrid(a, z, d, aprime)
% States (a, z) vary fastest, choices (d, aprime) vary slowest.
z_idx_state = repelem((1:n_z)', n_a, 1);
z_idx_flat = repmat(z_idx_state, n_choices, 1);

linear_indices = sub2ind([n_a, n_z], AprimeIdx_flat, z_idx_flat);
V_cont_flat = EV_next(linear_indices);

RHS_flat = BellmanCombiner(F_flat, V_cont_flat);
RHS_matrix = reshape(RHS_flat, n_states, n_choices);

[V_current, Policy_Indices] = max(RHS_matrix, [], 2);


end
