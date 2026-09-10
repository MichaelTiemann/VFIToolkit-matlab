function [V, Policy] = ValueFnIter_Case1_VFHorz(V_next, a_grid, aprime_grid, params, vfoptions)

% 1. Generate the Full State-Choice Grid (N-Dimensional)
% Instead of nested loops, we create a massive grid of all combinations.
[A_mat, Aprime_mat] = ndgrid(a_grid, aprime_grid);

% 2. Flatten for GPU/Vectorized SIMT execution
A_flat = A_mat(:);
Aprime_flat = Aprime_mat(:);

% 3. Create the Closure (The "eval_func")
% We bake in the static parameters, exposing only the states and choices.
% This closure maps exactly to what the optimization algorithms expect.
eval_func = @(states_a, choices_aprime) LifeCycleModel1_VReturnFn(...
    choices_aprime, states_a, params.w, params.r, params.sigma, params.beta);

% 4. Dispatch to a SINGLE optimization core
if vfoptions.divideandconquer
    % Future implementation
    % [V, Policy] = ValueFnIter_FHorz_DC_vectorized_raw(eval_func, V_next, A_flat, Aprime_flat);
else
    % Brute Force fallback
    [V, Policy] = ValueFnIter_FHorz_vectorized_raw(eval_func, V_next, A_flat, Aprime_flat, size(a_grid, 1), size(aprime_grid, 1));
end


end
