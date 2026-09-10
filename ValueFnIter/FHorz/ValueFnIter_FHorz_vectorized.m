function [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized(eval_func, V_next, A_flat, Aprime_flat, n_a, n_aprime)

% 1. Evaluate the return function across the entire flattened grid in one shot
% If A_flat and Aprime_flat are gpuArrays, this executes entirely on the GPU.
F_flat = eval_func(A_flat, Aprime_flat);

% 2. Add the discounted future value (V_next)
% Since Aprime_flat represents the choices, we index into V_next.
% We assume V_next is a column vector matching aprime_grid.
% We need to expand V_next to match the flattened grid.
% (Note: For models with transition matrices (z), you'd multiply them here).
beta = 0.96; % Should be passed in dynamically in a real implementation
RHS_flat = F_flat + beta * V_next(repmat(1:n_aprime, n_a, 1)'); 

% 3. Reshape back to the 2D grid (States x Choices) to find the max
RHS_matrix = reshape(RHS_flat, n_a, n_aprime);

% 4. Maximize over the choices (dimension 2)
[V_current, Policy_Indices] = max(RHS_matrix, [], 2);


end
