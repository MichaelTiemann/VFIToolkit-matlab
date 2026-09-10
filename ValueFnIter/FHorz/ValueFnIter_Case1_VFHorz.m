function varargout=ValueFnIter_Case1_VFHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% Extract ReturnFn parameters into a cell array for easy injection
n_params = length(ReturnFnParamNames);
param_vals = cell(1, n_params);
for i = 1:n_params
    param_vals{i} = Parameters.(ReturnFnParamNames{i});
end

% Build the N-dimensional grid combinations and flatten them
% For Case 1 (no shocks, no decisions), a_grid is used for both a and aprime
[A_mat, Aprime_mat] = ndgrid(a_grid, a_grid);
A_flat = A_mat(:);
Aprime_flat = Aprime_mat(:);

% Preallocate Value and Policy arrays
V = zeros(n_a, N_j);
Policy = zeros(n_a, N_j);

% Terminal period continuation value (V_next) is 0
V_next = zeros(n_a, 1);

% Backward Induction Loop
for j = N_j:-1:1
    
    % 1. Calculate the period-specific cumulative discount factor (beta_j)
    % Multiplies all parameters listed in DiscountFactorParamNames
    beta_j = 1;
    for b = 1:length(DiscountFactorParamNames)
        param_val = Parameters.(DiscountFactorParamNames{b});
        if length(param_val) == N_j
            beta_j = beta_j * param_val(j);
        else
            beta_j = beta_j * param_val;
        end
    end
    
    % 2. Extract period-specific parameters if they are age-dependent (length N_j)
    current_param_vals = cell(1, n_params);
    for p = 1:n_params
        if length(param_vals{p}) == N_j
            current_param_vals{p} = param_vals{p}(j);
        else
            current_param_vals{p} = param_vals{p};
        end
    end
    
    % 3. Create the period-specific closure
    eval_func = @(choices_aprime, states_a) ReturnFn(choices_aprime, states_a, current_param_vals{:});
    
    % 4. Dispatch to the Vectorized Raw Solver
    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        % To be implemented
        error('Divide and Conquer VCore not yet implemented.');
    else
        % Pass beta_j dynamically into the raw solver
        [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_raw(eval_func, V_next, A_flat, Aprime_flat, n_a, n_a, beta_j);
    end
    
    % 5. Store the results
    V(:, j) = V_current;
    Policy(:, j) = Policy_Indices;
    
    % 6. Update V_next for the next iteration (backward in time)
    V_next = V_current;
end

varargout{1} = V;
varargout{2} = Policy;


end
