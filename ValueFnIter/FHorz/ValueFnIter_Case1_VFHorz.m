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

if isempty(ReturnFnParamNames)
    ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters);
end

if vfoptions.parallel == 2
    if ~isempty(d_grid), d_grid = gpuArray(d_grid); end
    if ~isempty(a_grid), a_grid = gpuArray(a_grid); end
    if ~isempty(z_grid), z_grid = gpuArray(z_grid); end
    if ~isempty(pi_z),   pi_z   = gpuArray(pi_z);   end
end

N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);

if N_z > 0
    if vfoptions.alreadygridvals == 0
        [z_gridvals_J, pi_z_J, vfoptions] = ExogShockSetup_FHorz(n_z, z_grid, pi_z, N_j, Parameters, vfoptions, 3, 0);
    else
        z_gridvals_J = z_grid;
        pi_z_J = pi_z;
    end
else
    z_gridvals_J = [];
    pi_z_J = [];
end

% Standardize missing dimensions to length-1 singletons
if isempty(d_grid) || N_d == 0
    d_work = zeros(1, 1, 'like', a_grid);
    n_d_work = 1;
else
    d_work = d_grid;
    n_d_work = N_d;
end

a_work = a_grid;
n_a_work = N_a;

if isempty(z_gridvals_J) || N_z == 0
    z_work_1 = zeros(1, 1, 'like', a_grid);
    n_z_work = 1;
else
    z_work_1 = squeeze(z_gridvals_J(:, :, 1));
    n_z_work = N_z;
end

% Canonical grid: States (a, z), Choices (d, aprime)
[A_mat, Z_mat, D_mat, Aprime_mat] = ndgrid(a_work, z_work_1, d_work, a_work);
[~, ~, ~, AprimeIdx_mat] = ndgrid(1:n_a_work, 1:n_z_work, 1:n_d_work, 1:n_a_work);

A_flat = A_mat(:);
Z_flat = Z_mat(:);
D_flat = D_mat(:);
Aprime_flat = Aprime_mat(:);
AprimeIdx_flat = AprimeIdx_mat(:);

n_states = n_a_work * n_z_work;
n_choices = n_d_work * n_a_work;

V = zeros(n_a_work, n_z_work, N_j, 'like', a_grid);
PolicyKron = zeros(n_a_work, n_z_work, N_j, 'like', a_grid);
V_next = zeros(n_a_work, n_z_work, 'like', a_grid);

for j = N_j:-1:1
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, j);
    beta_j = prod(DiscountFactorParamsVec);
    
    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, j);
    if ~iscell(ReturnFnParamsVec)
        ReturnFnParamsVec = num2cell(ReturnFnParamsVec);
    end
    
    if N_z > 0
        if size(z_gridvals_J, 3) > 1
            z_work_j = squeeze(z_gridvals_J(:, :, j));
            [~, Z_mat, ~, ~] = ndgrid(a_work, z_work_j, d_work, a_work);
            Z_flat = Z_mat(:);
        end
        if j < N_j
            pi_z_j = pi_z_J(:, :, j);
        else
            pi_z_j = eye(n_z_work, 'like', a_grid);
        end
    else
        pi_z_j = ones(1, 1, 'like', a_grid);
    end
    
    if N_z > 0
        eval_func = @(aprime_in, a_in) ReturnFn(D_flat, aprime_in, a_in, Z_flat, ReturnFnParamsVec{:});
    elseif N_d > 0
        eval_func = @(aprime_in, a_in) ReturnFn(D_flat, aprime_in, a_in, ReturnFnParamsVec{:});
    else
        eval_func = @(aprime_in, a_in) ReturnFn(aprime_in, a_in, ReturnFnParamsVec{:});
    end
    
    [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_raw(eval_func, V_next, A_flat, Aprime_flat, AprimeIdx_flat, n_states, n_choices, n_a_work, n_z_work, pi_z_j, beta_j);
    
    V(:, :, j) = reshape(V_current, [n_a_work, n_z_work]);
    PolicyKron(:, :, j) = reshape(Policy_Indices, [n_a_work, n_z_work]);
    V_next = reshape(V_current, [n_a_work, n_z_work]);
end

if N_z == 0
    V = squeeze(V);
end

if N_d == 0
    n_daprime = n_a;
else
    n_daprime = [n_d, n_a];
end

PolicyKron = shiftdim(PolicyKron, -1);

if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    varargout{1} = V;
    varargout{2} = PolicyKron;
    return
end

if N_z > 0
    Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_daprime, n_a, N_z, N_j, vfoptions);
else
    Policy = UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_daprime, n_a, N_j, vfoptions);
end

varargout{1} = V;
varargout{2} = Policy;


end
