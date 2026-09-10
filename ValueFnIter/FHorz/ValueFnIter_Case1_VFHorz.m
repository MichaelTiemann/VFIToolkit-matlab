function varargout=ValueFnIter_Case1_VFHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

%% Implement new way of handling ReturnFn inputs
if isempty(ReturnFnParamNames)
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
end
% Basic setup: the first inputs of ReturnFn will be (d,aprime,a,z,..) and everything after this is a parameter, so we get the names of all these parameters.
% But this changes if you have e, semiz, or just multiple d, and if you use riskyasset, expasset, etc.
% So figure out which setup we have, and get the relevant ReturnFnParamNames

% Create index array for safe V_next lookups
a_idx = 1:n_a;

% Build the N-dimensional grid combinations and flatten them
if n_d > 0
    [A_mat, Aprime_mat, D_mat] = ndgrid(a_grid, a_grid, d_grid);
    [~, AprimeIdx_mat, ~] = ndgrid(a_idx, a_idx, 1:n_d);
    
    A_flat = A_mat(:);
    Aprime_flat = Aprime_mat(:);
    D_flat = D_mat(:);
    AprimeIdx_flat = AprimeIdx_mat(:);
    
    n_choices = n_a * n_d;
else
    [A_mat, Aprime_mat] = ndgrid(a_grid, a_grid);
    [~, AprimeIdx_mat] = ndgrid(a_idx, a_idx);
    
    A_flat = A_mat(:);
    Aprime_flat = Aprime_mat(:);
    AprimeIdx_flat = AprimeIdx_mat(:);
    
    n_choices = n_a;
end

% Preallocate Value and Policy arrays
V = zeros(n_a, N_j);
Policy = zeros(n_a, N_j);

% Terminal period continuation value (V_next) is 0
V_next = zeros(n_a, 1);

% Backward Induction Loop
for j = N_j:-1:1
    
    % 1. Extract period-specific discount factor using the toolkit's native parser
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, j);
    beta_j = prod(DiscountFactorParamsVec);
    
    % 2. Extract period-specific return function parameters
    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
    
    % Ensure it's a cell array for dynamic unpacking into the closure
    if ~iscell(ReturnFnParamsVec)
        ReturnFnParamsVec = num2cell(ReturnFnParamsVec);
    end
    
    % 3. Create the period-specific closure mapped to the flattened arrays
    if n_d > 0
        eval_func = @(choices_aprime, states_a) ReturnFn(D_flat, choices_aprime, states_a, ReturnFnParamsVec{:});
    else
        eval_func = @(choices_aprime, states_a) ReturnFn(choices_aprime, states_a, ReturnFnParamsVec{:});
    end
    
    % 4. Dispatch to the Vectorized Raw Solver
    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        error('Divide and Conquer VCore not yet implemented.');
    else
        % Pass AprimeIdx_flat to guarantee safe continuation value lookups
        [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_raw(eval_func, V_next, A_flat, Aprime_flat, AprimeIdx_flat, n_a, n_choices, beta_j);
    end
    
    % 5. Store the results
    V(:, j) = V_current;
    Policy(:, j) = Policy_Indices;
    
    % 6. Update V_next for the next iteration
    V_next = V_current;
end

varargout{1} = V;
varargout{2} = Policy;


end
