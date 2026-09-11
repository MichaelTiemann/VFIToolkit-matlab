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

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        if isscalar(n_a)
            vfoptions.level1n=floor(sqrt(n_a(1)));
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        elseif length(n_a)==2
            vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch
            if n_a(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
        end
        if vfoptions.verbose==1
            fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
        end
    else
        if ~isscalar(n_a) && isscalar(vfoptions.level1n)
            vfoptions.level1n=[vfoptions.level1n,n_a(2:end)]; % user only needs to declare level1n for first dimension. Fill out the rest with n_a(2:end).
        end
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

if vfoptions.divideandconquer == 0 && vfoptions.gridinterplayer == 0
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
else
    A_flat = []; Z_flat = []; D_flat = []; Aprime_flat = []; AprimeIdx_flat = [];
    n_states = n_a_work * n_z_work;
    n_choices = n_d_work * n_a_work;
end

has_e = isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0;
if has_e
    n_e_work = prod(vfoptions.n_e);
    e_work   = shiftdim(gpuArray(vfoptions.e_grid),-2);
else
    n_e_work = 1;
    e_work   = gpuArray(0); % dummy scalar keeping rank/signatures consistent
end

V = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
if vfoptions.gridinterplayer == 1
    PolicyKron = zeros(3, n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
else
    PolicyKron = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
end
V_next = zeros(n_a_work, n_z_work, n_e_work, 'like', a_grid);

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
        else
            z_work_j = z_work_1;
        end
        if j < N_j
            pi_z_j = pi_z_J(:, :, j);
        else
            pi_z_j = eye(n_z_work, 'like', a_grid);
        end
    else
        z_work_j = zeros(1, 1, 'like', a_grid);
        pi_z_j   = ones(1, 1, 'like', a_grid);
    end

    % Wrap user ReturnFn into standard signature: eval_kernel(d_in, apr_in, a_in, z_in)
    has_d = (n_d_work > 0 && n_d(1) > 0);
    has_z = (n_z_work > 0 && N_z > 0);

    if has_e
        if isfield(vfoptions, 'pi_e_J') && ~isempty(vfoptions.pi_e_J)
            % Age-specific column slice: size [n_e, 1]
            pi_e_j = gpuArray(vfoptions.pi_e_J(:, j));
        elseif isfield(vfoptions, 'pi_e') && ~isempty(vfoptions.pi_e)
            % Time-invariant distribution: size [n_e, 1]
            pi_e_j = gpuArray(vfoptions.pi_e(:));
        else
            error('has_e is true, but neither pi_e nor pi_e_J is defined in vfoptions.');
        end
    else
        pi_e_j = gpuArray(1);
    end

    % ---------------------------------------------------------------------
    % 1. Continuation Value Integration: Integrate e' out, then Markov z' -> z
    % ---------------------------------------------------------------------
    if j == N_j
        EV_next = zeros(n_a_work, n_z_work, 'like', a_work);
    else
        % Tomorrow's marginal shock distribution: pi_e(j+1)
        if has_e
            if isfield(vfoptions, 'pi_e_J') && ~isempty(vfoptions.pi_e_J)
                pi_e_tomorrow = gpuArray(vfoptions.pi_e_J(:, j + 1));
            else
                pi_e_tomorrow = gpuArray(vfoptions.pi_e(:));
            end

            % V_next has shape (n_a x n_z x n_e)
            % Integrate across dimension 3: dot product with pi_e_tomorrow
            V_next_inte = sum(V_next .* reshape(pi_e_tomorrow, [1, 1, n_e_work]), 3); % -> (n_a x n_z)

            if has_z
                EV_next = V_next_inte * (pi_z_j'); % (n_a x n_z)
            else
                EV_next = V_next_inte;
            end
        else
            if has_z
                EV_next = V_next * (pi_z_j');
            else
                EV_next = V_next;
            end
        end
    end

    % ---------------------------------------------------------------------
    % 2. Memory Throttling Bounds (Hoisted across all kernels)
    % ---------------------------------------------------------------------
    lowmem = 0;
    if isfield(vfoptions, 'lowmemory')
        lowmem = vfoptions.lowmemory;
    end

    % lowmem >= 1: loop sequentially over e
    use_loop_e = (has_e && lowmem >= 1);
    if use_loop_e
        n_e_loops = n_e_work;
    else
        n_e_loops = 1;
    end

    % lowmem >= 2: loop sequentially over z
    use_loop_z = (has_z && lowmem >= 2);
    if use_loop_z
        n_z_loops = n_z_work;
    else
        n_z_loops = 1;
    end

    % Container allocation for period j
    % Policy has 2 rows (standard) or 3 rows (gridinterplayer)
    n_pol_rows = 2 + (vfoptions.gridinterplayer == 1);
    if has_e
        V_j_all = zeros(n_a_work, n_z_work, n_e_work, 'like', a_work);
        Pol_j_all = zeros(n_pol_rows, n_a_work, n_z_work, n_e_work, 'like', a_work);
    else
        V_j_all = zeros(n_a_work, n_z_work, 'like', a_work);
        Pol_j_all = zeros(n_pol_rows, n_a_work, n_z_work, 'like', a_work);
    end

    % ---------------------------------------------------------------------
    % 3. Nested Shocks Iteration
    % ---------------------------------------------------------------------
    for z_iter = 1:n_z_loops
        if use_loop_z
            z_slice = z_work_j(z_iter);
            n_z_slice = 1;
            z_idx_range = z_iter;
            % Slice continuation value for this specific z: (n_a x 1)
            EV_slice = EV_next(:, z_iter);
        else
            z_slice = z_work_j;
            n_z_slice = n_z_work;
            z_idx_range = 1:n_z_work;
            EV_slice = EV_next; % (n_a x n_z)
        end

        for e_iter = 1:n_e_loops
            if use_loop_e
                e_slice = e_work(1, 1, e_iter); % keeps it scalar [1, 1, 1]
                n_e_slice = 1;
                e_idx_range = e_iter;
            else
                e_slice = e_work;
                n_e_slice = n_e_work;
                e_idx_range = 1:n_e_work;
            end
            
            % Construct unified kernel adapter for this e-slice
            % Signature inside all kernels: eval_kernel(d_in, apr_in, a_in, z_in)
            if has_d && has_z && has_e
                eval_kernel = @(d_in, apr_in, a_in, z_in) ReturnFn(d_in, apr_in, a_in, z_in, e_slice, ReturnFnParamsVec{:});
            elseif has_d && has_z && ~has_e
                eval_kernel = @(d_in, apr_in, a_in, z_in) ReturnFn(d_in, apr_in, a_in, z_in, ReturnFnParamsVec{:});
            elseif ~has_d && has_z && has_e
                eval_kernel = @(d_in, apr_in, a_in, z_in) ReturnFn(apr_in, a_in, z_in, e_slice, ReturnFnParamsVec{:});
            elseif ~has_d && has_z && ~has_e
                eval_kernel = @(d_in, apr_in, a_in, z_in) ReturnFn(apr_in, a_in, z_in, ReturnFnParamsVec{:});
            elseif has_d && ~has_z && ~has_e
                eval_kernel = @(d_in, apr_in, a_in, z_in) ReturnFn(d_in, apr_in, a_in, ReturnFnParamsVec{:});
            else
                eval_kernel = @(d_in, apr_in, a_in, z_in) ReturnFn(apr_in, a_in, ReturnFnParamsVec{:});
            end

            % -------------------------------------------------------------
            % 4. Method Dispatch (Consumes standard n_z_slice, EV_slice)
            % -------------------------------------------------------------
            if vfoptions.divideandconquer == 1 && vfoptions.gridinterplayer == 1
                [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_DC1_GI1(...
                    eval_kernel, ReturnFnParamsVec, EV_slice, a_work, z_slice, d_work, ...
                    n_a_work, n_z_slice, n_d_work, pi_z_j, beta_j, vfoptions);

            elseif vfoptions.gridinterplayer == 1
                [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_GI1_raw(...
                    eval_kernel, ReturnFnParamsVec, EV_slice, a_work, z_slice, d_work, ...
                    n_a_work, n_z_slice, n_d_work, pi_z_j, beta_j, vfoptions);

            elseif vfoptions.divideandconquer == 1
                [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_DC1(...
                    eval_kernel, EV_slice, a_work, z_slice, d_work, ...
                    n_a_work, n_z_slice, n_d_work, pi_z_j, beta_j, vfoptions);

            else
                eval_func_raw = @(apr_in, a_in) eval_kernel(D_flat, apr_in, a_in);
                [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_raw(...
                    eval_func_raw, EV_slice, A_flat, Aprime_flat, AprimeIdx_flat, ...
                    n_states, n_choices, n_a_work, n_z_slice, pi_z_j, beta_j);
            end

            % Store results into period containers
            if has_e
                V_j_all(:, z_idx_range, e_idx_range) = reshape(V_sub, [n_a_work, n_z_slice, n_e_slice]);
                Pol_j_all(:, :, z_idx_range, e_idx_range) = reshape(Pol_sub, [n_pol_rows, n_a_work, n_z_slice, n_e_slice]);
            else
                V_j_all(:, z_idx_range) = reshape(V_sub, [n_a_work, n_z_slice]);
                Pol_j_all(:, :, z_idx_range) = reshape(Pol_sub, [n_pol_rows, n_a_work, n_z_slice]);
            end
        end
    end

    V(:, :, :, j) = V_j_all;
    PolicyKron(:, :, :, :, j) = Pol_j_all;
    V_next = V_j_all;
end

if N_z == 0
    V = squeeze(V);
end

if N_d == 0
    n_daprime = n_a;
else
    n_daprime = [n_d, n_a];
end

if vfoptions.gridinterplayer == 0
    PolicyKron = shiftdim(PolicyKron, -1);
end

if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    varargout{1} = V;
    varargout{2} = PolicyKron;
    return
end

if has_z && has_e
    Policy = UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_daprime, n_a, N_z, n_e_work, N_j, vfoptions);
elseif has_z && ~has_e
    Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_daprime, n_a, N_z, N_j, vfoptions);
elseif ~has_z && has_e
    Policy = UnKronPolicyIndexes1_FHorz_e(PolicyKron, n_daprime, n_a, n_e_work, N_j, vfoptions);
else
    Policy = UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_daprime, n_a, N_j, vfoptions);
end

varargout{1} = V;
varargout{2} = Policy;


end
