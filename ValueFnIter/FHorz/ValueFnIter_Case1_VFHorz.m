function varargout=ValueFnIter_Case1_VFHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

%% Check which vfoptions have been used, set all others to defaults
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    % If vfoptions is not given, just use all the defaults
    vfoptions.verbose=0; % =1 print out feedback on what is happening internally
    vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    vfoptions.gridinterplayer=0; % Interpolate between grid points (not yet implemented for alternative preferences)
    vfoptions.lowmemory=0; % use more loops and less parallelization, reduce memory use but at the cost of slower runtimes
    % Alternative model setups
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    vfoptions.exoticpreferences='None';
    vfoptions.dynasty=0;
    vfoptions.experienceasset=0;
    vfoptions.experienceassetu=0;
    vfoptions.experienceassete=0;
    vfoptions.experienceassetz=0;
    vfoptions.experienceassetze=0;
    vfoptions.experienceassetsemiz=0;
    vfoptions.riskyasset=0;
    vfoptions.residualasset=0;
    vfoptions.n_ambiguity=0;
    vfoptions.n_e=0;
    vfoptions.n_semiz=0;
    % Largely just for internal use only
    vfoptions.parallel=1+(gpuDeviceCount>0);
    % When calling as a subcommand, the following are used internally
    vfoptions.outputkron=0; % If 1 then leave output in Kron form
    vfoptions.alreadygridvals=0; % =1 when calling as a subcommand
    vfoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
else
    % Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0; % =1 Use divide-and-conquer to exploit monotonicity
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0; % =1 Interpolate between grid points (not yet implemented for most cases)
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp (number of points to interpolate for aprime between each consecutive pair of points in a_grid)')
        end
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    % Alternative model setups
    if ~isfield(vfoptions,'incrementaltype')
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if ~isfield(vfoptions,'dynasty')
        vfoptions.dynasty=0;
    end
    if ~isfield(vfoptions,'experienceasset')
        vfoptions.experienceasset=0;
    end
    if ~isfield(vfoptions,'experienceassetu')
        vfoptions.experienceassetu=0;
    end
    if ~isfield(vfoptions,'experienceassete')
        vfoptions.experienceassete=0;
    end
    if ~isfield(vfoptions,'experienceassetz')
        vfoptions.experienceassetz=0;
    end
    if ~isfield(vfoptions,'experienceassetze')
        vfoptions.experienceassetze=0;
    end
    if ~isfield(vfoptions,'experienceassetsemiz')
        vfoptions.experienceassetsemiz=0;
    end
    if ~isfield(vfoptions,'riskyasset')
        vfoptions.riskyasset=0;
    end
    if ~isfield(vfoptions,'residualasset')
        vfoptions.residualasset=0;
    end
    if ~isfield(vfoptions,'n_ambiguity')
        vfoptions.n_ambiguity=0;
    end
    if ~isfield(vfoptions,'n_e')
        vfoptions.n_e=0;
    end
    if ~isfield(vfoptions,'n_semiz')
        vfoptions.n_semiz=0;
    end
    % Largely just for internal use only
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    % When calling as a subcommand, the following are used internally
    if ~isfield(vfoptions,'outputkron')
        vfoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(vfoptions,'alreadygridvals')
        vfoptions.alreadygridvals=0; % =1 when calling as a subcommand
    end
    if ~isfield(vfoptions,'alreadygridvals_semiexo')
        vfoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    end
end

% Implement VFIToolkit way of handling ReturnFn inputs
if isempty(ReturnFnParamNames)
    ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
end

% Create index array for safe V_next lookups
a_idx = 1:n_a;
N_d = prod(n_d);

% Build the N-dimensional grid combinations and flatten them
if N_d > 0
    % VFIToolkit choice order is [d, aprime], so d must vary faster than aprime!
    [A_mat, D_mat, Aprime_mat] = ndgrid(a_grid, d_grid, a_grid);
    [~, ~, AprimeIdx_mat] = ndgrid(a_idx, 1:N_d, a_idx);
    
    A_flat = A_mat(:);
    D_flat = D_mat(:);
    Aprime_flat = Aprime_mat(:);
    AprimeIdx_flat = AprimeIdx_mat(:);
    
    n_choices = N_d * n_a;
else
    [A_mat, Aprime_mat] = ndgrid(a_grid, a_grid);
    [~, AprimeIdx_mat] = ndgrid(a_idx, a_idx);
    
    A_flat = A_mat(:);
    Aprime_flat = Aprime_mat(:);
    AprimeIdx_flat = AprimeIdx_mat(:);
    
    n_choices = n_a;
end

% Preallocate Value and Policy arrays
% Our Policy array will hold the raw linear indices (PolicyKron) until the end
V = zeros(n_a, N_j);
PolicyKron = zeros(n_a, N_j);

% Terminal period continuation value (V_next) is 0
V_next = zeros(n_a, 1);

% Backward Induction Loop
for j = N_j:-1:1
    
    % 1. Extract period-specific discount factor
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, j);
    beta_j = prod(DiscountFactorParamsVec);
    
    % 2. Extract period-specific return function parameters
    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
    if ~iscell(ReturnFnParamsVec)
        ReturnFnParamsVec = num2cell(ReturnFnParamsVec);
    end
    
    % 3. Create the period-specific closure mapped to the flattened arrays
    if N_d > 0
        eval_func = @(choices_aprime, states_a) ReturnFn(D_flat, choices_aprime, states_a, ReturnFnParamsVec{:});
    else
        eval_func = @(choices_aprime, states_a) ReturnFn(choices_aprime, states_a, ReturnFnParamsVec{:});
    end
    
    % 4. Dispatch to the Vectorized Raw Solver
    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        error('Divide and Conquer VCore not yet implemented.');
    else
        [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_raw(eval_func, V_next, A_flat, Aprime_flat, AprimeIdx_flat, n_a, n_choices, beta_j);
    end
    
    % 5. Store the results
    V(:, j) = V_current;
    PolicyKron(:, j) = Policy_Indices;
    
    % 6. Update V_next for the next iteration
    V_next = V_current;
end

%% Format output strictly to VFIToolkit expectations
if N_d == 0
    n_daprime = n_a;
else
    n_daprime = [n_d, n_a];
end

if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    varargout{1} = V;
    varargout{2} = PolicyKron;
    return
end

% VFIToolkit expects PolicyKron (in this simple case) to have a singleton first dimension (1, n_a, N_j)
% Later we will handle more exotic things, like L2 interpolation index, L2 flag, etc.
PolicyKron = shiftdim(PolicyKron, -1);

% Let the toolkit wrap our optimal indices into the expected cell array structure
Policy = UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_daprime, n_a, N_j, vfoptions);

varargout{1} = V;
varargout{2} = Policy;

end