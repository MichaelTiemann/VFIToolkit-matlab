function [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_raw(eval_func, V_next, A_flat, Aprime_flat, AprimeIdx_flat, n_a, n_choices, beta_j)

% Evaluate the return function across the entire flattened grid in one shot.
F_flat = eval_func(Aprime_flat, A_flat);

% Expand V_next to map to the correct next-period asset choice.
% AprimeIdx_flat explicitly maps every point in the flattened grid to its V_next index.
V_next_expanded = V_next(AprimeIdx_flat);

% Calculate the right-hand side of the Bellman equation
RHS_flat = F_flat + beta_j * V_next_expanded;

% Reshape back to the 2D grid (States x Choices) to find the max
RHS_matrix = reshape(RHS_flat, n_a, n_choices);

% Maximize over the choices (dimension 2)
[V_current, Policy_Indices] = max(RHS_matrix, [], 2);


end
