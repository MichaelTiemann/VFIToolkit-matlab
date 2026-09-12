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

is_EZ = strcmp(vfoptions.exoticpreferences, 'EpsteinZin');
if is_EZ
    % Reject asset types this dispatcher does not handle: every asset type it does handle is
    % dispatched below and returns, so an unsupported flag would otherwise be silently ignored.
    if vfoptions.experienceasset>=1 || vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
        error('Epstein-Zin preferences are not implemented for the experience assets (only for riskyasset, or for the standard endogenous states)')
    end
    if vfoptions.residualasset==1
        error('Epstein-Zin preferences are not implemented for residualasset')
    end
    if vfoptions.dynasty==1
        error('Epstein-Zin preferences are not implemented for dynasty')
    end

    %% Some Epstein-Zin specific options need to be set if they are not already declared
    if ~isfield(vfoptions,'EZriskaversion')
        error('When using Epstein-Zin preferences you must declare vfoptions.EZriskaversion (coefficient controlling risk aversion)')
    end
    if ~isfield(vfoptions,'EZutils')
        vfoptions.EZutils=1; % Use EZ preferences with general utility function (0 gives traditional EZ with exogenous labor, 2 gives traditional EZ with endogenous labor)
    end
    if vfoptions.EZutils==1
        % Have to do EZ preferences differently depending on whether the utility function is >=0 or <=0.
        % vfoptions.EZpositiveutility=1 if utility is positive; Note, in this case when EZriskaversion is higher, the risk aversion is larger (EZriskaversion>0 is risk averse)
        % vfoptions.EZpositiveutility=0 if utility is negative; Note, in this case when EZriskaversion is lower, the risk aversion is larger  (EZriskaversion<0 is risk averse)
        if ~isfield(vfoptions,'EZpositiveutility')
            warning('Using Epstein-Zin preferences it is assumed the utility/return function is negative valued, if not you need to set vfoptions.EZpositiveutility=1')
            vfoptions.EZpositiveutility=0; % User did not specify. Guess that it is negative as most common things (like CES) are negative valued.
        end
    else
        % Traditional EZ preferences requires you to specify the EIS parameter
        if ~isfield(vfoptions,'EZeis')
            error('When using Epstein-Zin preferences you must declare vfoptions.EZeis (elasticity of intertemporal substitution)')
        end
    end
    if ~isfield(vfoptions,'EZoneminusbeta')
        vfoptions.EZoneminusbeta=0; % default essentially does nothing
        %=1 Put a (1-beta)* term on the this period return
        %=2 Put a (1-sj*beta)* term on the this period return
    end
    % Set up sj
    if isfield(vfoptions,'survivalprobability')
        sj=Parameters.(vfoptions.survivalprobability);
        if length(sj)~=N_j
            error('Survival probabilities must be of the same length as N_j')
        end
    elseif isfield(vfoptions,'WarmGlowBequestsFn')
        % If you have warm-glow but do not specify survival probabilities it is assumed you only get it at end of final period
        sj=ones(N_j,1); % conditional survival probabilities
        sj(end)=0;
        warning('You have used vfoptions.WarmGlowBequestsFn, but have not set vfoptions.survivalprobability, it is assumed you only want to have the warm-glow at the end of the final period')
    else
        sj=ones(N_j,1); % conditional survival probabilities
    end
    % Declare warmglow indicator
    if isfield(vfoptions,'WarmGlowBequestsFn')
        warmglow=1;
        temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
        vfoptions.WarmGlowBequestsFnParamsNames={temp{2:end}};
    else
        warmglow=0;
    end
    [ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8, sj, warmglow] = ...
        EpsteinZinSetup_FHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions);
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

%% Semi-exogenous shock gridvals and pi
if vfoptions.alreadygridvals_semiexo==0
    if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz)>0
        % Internally, only ever use age-dependent joint-grids
        vfoptions = SemiExogShockSetup_FHorz(n_d, N_j, d_grid, Parameters, vfoptions, 3);
    end
end

N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);

%% Exogenous shock gridvals and pi
if N_z > 0
    if vfoptions.alreadygridvals == 0
        % ExogShockSetup_FHorz is called with KeepOriginalGrid==0 here
        [z_gridvals_J, pi_z_J, vfoptions] = ExogShockSetup_FHorz(n_z, z_grid, pi_z, N_j, Parameters, vfoptions, 3, 0);
    else
        z_gridvals_J = z_grid;
        pi_z_J = pi_z;
    end
else
    z_gridvals_J = [];
    pi_z_J = [];
end

%% Semi-exogenous state Dispatch
% The transition matrix of the exogenous shocks depends on the value of the 'last' decision variable(s).
if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz)>0
    if length(n_d) > vfoptions.l_dsemiz
        n_d1 = n_d(1:end-vfoptions.l_dsemiz);
        d1_grid = d_grid(1:sum(n_d1));
    else
        n_d1 = 0; 
        d1_grid = [];
    end
    n_d2 = n_d(end-vfoptions.l_dsemiz+1:end); % n_d2 influences transition probs
    d2_grid = d_grid(sum(n_d1)+1:end);

    d1_gridvals = CreateGridvals(n_d1, d1_grid, 1);
    d2_gridvals = CreateGridvals(n_d2, d2_grid, 1);

    % Dispatch to the vectorized SemiExo handler and bail out of Case1
    [V, Policy] = ValueFnIter_VFHorz_SemiExo(n_d1, n_d2, n_a, vfoptions.n_semiz, n_z, N_j, ...
        d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, ...
        pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, ...
        DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

    varargout = {V, Policy};
    return
end

% Standardize missing dimensions to length-1 singletons
if isempty(d_grid) || N_d == 0
    n_d_vars = 0;
    d_work = zeros(1, 1, 'like', a_grid);
    n_d_work = 1;
else
    n_d_vars = length(n_d);
    d_work = d_grid;
    n_d_work = N_d;
end
has_d = (n_d_work > 0 && n_d(1) > 0);

% Set up D_cells once outside the reverse_j loop
if has_d
    num_d = length(n_d);
    if num_d > 1
        % 1. Extract 1D grid vectors from the stacked d_grid
        d_grids_1d = cell(1, num_d);
        offset = 0;
        for i_d = 1:num_d
            d_grids_1d{i_d} = d_grid((offset + 1):(offset + n_d(i_d)));
            offset = offset + n_d(i_d);
        end
        
        % 2. Form Cartesian coordinates matching the Kron order: [N_d x num_d]
        [D_mesh{1:num_d}] = ndgrid(d_grids_1d{:});
        
        % 3. Pack into cell array, each variable spanning Dim 4: [1, 1, 1, N_d]
        D_cells = cell(1, num_d);
        for i_d = 1:num_d
            D_cells{i_d} = shiftdim(D_mesh{i_d}(:), -3);
        end
    else
        D_cells = { shiftdim(d_work(:), -3) };
    end
else
    D_cells = {};
end

% ---------------------------------------------------------------------
% Unstack Endogenous States (a, n1, n2, ...)
% ---------------------------------------------------------------------
num_a = length(n_a);
if num_a > 1
    a_grids_1d = cell(1, num_a);
    offset = 0;
    for i_a = 1:num_a
        a_grids_1d{i_a} = a_grid((offset + 1):(offset + n_a(i_a)));
        offset = offset + n_a(i_a);
    end
    [A_mesh_raw{1:num_a}] = ndgrid(a_grids_1d{:});
    A_mat = zeros(N_a, num_a, 'like', a_grid);
    for i_a = 1:num_a
        A_mat(:, i_a) = A_mesh_raw{i_a}(:);
    end
else
    A_mat = a_grid(:);
end
a_work = A_mat(:, 1); % Primary asset grid for interpolation
n_a_work = N_a;

if isempty(z_gridvals_J) || N_z == 0
    N_z_exog = 0;
    z_work_1 = zeros(1, 1, 'like', a_grid);
else
    N_z_exog = N_z;
    z_work_1 = squeeze(z_gridvals_J(:, :, 1));
end

has_semiz = isfield(vfoptions, 'n_semiz') && ~isempty(vfoptions.n_semiz) && prod(vfoptions.n_semiz) > 0;
if has_semiz
    N_semiz = prod(vfoptions.n_semiz);
    n_all_z = [vfoptions.n_semiz, n_z];
else
    N_semiz = 1;
    n_all_z = n_z;
end
n_z_work = N_semiz * max(1, N_z_exog);
has_z = (N_z_exog > 0);

has_e = isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0;
if has_e
    n_e_vars = length(vfoptions.n_e);
    n_e_work = prod(vfoptions.n_e);
    e_work   = shiftdim(gpuArray(vfoptions.e_grid),-2);
else
    n_e_vars = 0;
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

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;

    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(gpuArray(vfoptions.V_Jplus1), size(V_next));
    end

    % 1. Get standard discount factor
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);

    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    if ~iscell(ReturnFnParamsVec)
        ReturnFnParamsVec = num2cell(ReturnFnParamsVec);
    end

    if N_z > 0
        if size(z_gridvals_J, 3) > 1
            z_work_j = squeeze(z_gridvals_J(:, :, jj));
        else
            z_work_j = z_work_1;
        end
        pi_z_j = pi_z_J(:, :, jj);
    else
        z_work_j = zeros(1, 1, 'like', a_grid);
        pi_z_j   = ones(1, 1, 'like', a_grid);
    end

    if has_z
        num_z = length(n_all_z);

        if has_semiz
            if isfield(vfoptions, 'semiexog_grid')
                semiz_work = vfoptions.semiexog_grid;
            elseif isfield(vfoptions, 'semiz_grid')
                semiz_work = vfoptions.semiz_grid;
            elseif isfield(vfoptions, 'semiz_gridvals')
                semiz_work = vfoptions.semiz_gridvals;
            end
            s_grids_1d = cell(1, length(vfoptions.n_semiz));
            offset = 0;
            for i_s = 1:length(vfoptions.n_semiz)
                s_grids_1d{i_s} = semiz_work((offset + 1):(offset + vfoptions.n_semiz(i_s)));
                offset = offset + vfoptions.n_semiz(i_s);
            end
        else
            s_grids_1d = {};
        end

        z_grids_1d = cell(1, length(n_z));
        offset = 0;
        for i_z = 1:length(n_z)
            z_grids_1d{i_z} = z_work_j((offset + 1):(offset + n_z(i_z)));
            offset = offset + n_z(i_z);
        end

        all_grids_1d = [s_grids_1d, z_grids_1d];
        [Z_mesh{1:num_z}] = ndgrid(all_grids_1d{:});

        Z_cells = cell(1, num_z);
        for i_z = 1:num_z
            Z_cells{i_z} = shiftdim(Z_mesh{i_z}(:), -1); % Dim 2
        end
    else
        Z_cells = {};
    end

    if has_e
        if isfield(vfoptions, 'pi_e_J') && ~isempty(vfoptions.pi_e_J)
            % Age-specific column slice: size [n_e, 1]
            pi_e_j = gpuArray(vfoptions.pi_e_J(:, jj));
        elseif isfield(vfoptions, 'pi_e') && ~isempty(vfoptions.pi_e)
            % Time-invariant distribution: size [n_e, 1]
            pi_e_j = gpuArray(vfoptions.pi_e(:));
        else
            error('has_e is true, but neither pi_e nor pi_e_J is defined in vfoptions.');
        end
        % Set up E_cells outside z_iter loop
        num_e = length(vfoptions.n_e);
        if num_e > 1
            if size(vfoptions.e_grid, 2) == num_e
                % Already Cartesian coordinates: [N_e x num_e]
                E_cells = cell(1, num_e);
                for i_e = 1:num_e
                    E_cells{i_e} = shiftdim(gpuArray(vfoptions.e_grid(:, i_e)), -2); % Dim 3: [1, 1, N_e]
                end
            else
                % Stacked grid of length sum(n_e)
                e_grids_1d = cell(1, num_e);
                offset = 0;
                for i_e = 1:num_e
                    e_grids_1d{i_e} = vfoptions.e_grid((offset + 1):(offset + vfoptions.n_e(i_e)));
                    offset = offset + vfoptions.n_e(i_e);
                end
                [E_mesh{1:num_e}] = ndgrid(e_grids_1d{:});
                E_cells = cell(1, num_e);
                for i_e = 1:num_e
                    E_cells{i_e} = shiftdim(gpuArray(E_mesh{i_e}(:)), -2); % Dim 3: [1, 1, N_e]
                end
            end
        else
            E_cells = { e_work }; % Already pushed to Dim 3
        end
    else
        E_cells = {};
    end

    % ---------------------------------------------------------------------
    % 1. Pre-Expectation Transform
    % ---------------------------------------------------------------------
    if is_EZ
        if vfoptions.EZoneminusbeta == 1
            ezc1 = 1 - beta_j;
        elseif vfoptions.EZoneminusbeta == 2
            ezc1 = 1 - sj(jj) * beta_j;
        else
            ezc1 = 1;
        end

        temp_V = V_next;
        valid_V = isfinite(V_next);
        temp_V(valid_V) = (ezc4 * V_next(valid_V)).^ezc5(jj);
        temp_V(V_next == 0) = 0;
        temp_V(~isfinite(V_next)) = ezc4 * V_next(~isfinite(V_next));
    else
        temp_V = V_next;
    end

    % ---------------------------------------------------------------------
    % 2. Continuation Value Integration: Integrate e' out, then Markov z' -> z
    % ---------------------------------------------------------------------
    if jj == N_j && ~(isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1))
        EV_raw = zeros(n_a_work, n_z_work, 'like', a_work);
    else
        if has_e
            if isfield(vfoptions, 'pi_e_J') && ~isempty(vfoptions.pi_e_J)
                pi_e_tomorrow = gpuArray(vfoptions.pi_e_J(:, min(jj + 1, size(vfoptions.pi_e_J, 2))));
            else
                pi_e_tomorrow = gpuArray(vfoptions.pi_e(:));
            end
            temp_V_inte = sum(temp_V .* reshape(pi_e_tomorrow, [1, 1, n_e_work]), 3);
            if has_z
                if has_semiz
                    temp_V_inte_rs = reshape(temp_V_inte, [n_a_work * N_semiz, N_z_exog]);
                    EV_raw_rs = temp_V_inte_rs * (pi_z_j');
                    EV_raw = reshape(EV_raw_rs, [n_a_work, n_z_work]);
                else
                    EV_raw = temp_V_inte * (pi_z_j');
                end
            else
                EV_raw = temp_V_inte;
            end
        else
            if has_z
                if has_semiz
                    temp_V_rs = reshape(temp_V, [n_a_work * N_semiz, N_z_exog]);
                    EV_raw_rs = temp_V_rs * (pi_z_j');
                    EV_raw = reshape(EV_raw_rs, [n_a_work, n_z_work]);
                else
                    EV_raw = temp_V * (pi_z_j');
                end
            else
                EV_raw = temp_V;
            end
        end
    end

    % ---------------------------------------------------------------------
    % 3. Post-Expectation Transform (CE) & Warm Glow Matrix
    % ---------------------------------------------------------------------
    if is_EZ
        if warmglow == 1
            WGParamsCell = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            WGmatrix = vfoptions.WarmGlowBequestsFn(a_work, WGParamsCell{:});
        else
            WGmatrix = 0;
        end

        EV_next = zeros(size(EV_raw), 'like', EV_raw);
        valid_EV = isfinite(EV_raw);

        if warmglow == 1
            % Broadcast full 2D tensors: (N_a x N_z) + (N_a x 1)
            CE_base = sj(jj) .* (EV_raw .^ ezc8(jj)) + (1 - sj(jj)) .* (WGmatrix .^ ezc8(jj));
            EV_next(valid_EV) = CE_base(valid_EV) .^ ezc6(jj);
            EV_next((EV_raw == 0) & (WGmatrix == 0)) = 0;
        else
            CE_base = sj(jj) .* (EV_raw .^ ezc8(jj));
            EV_next(valid_EV) = CE_base(valid_EV) .^ ezc6(jj);
            EV_next(EV_raw == 0) = 0;
        end

        BellmanCombiner = @(F, EV_cont) Compute_EZ_RHS(F, EV_cont, ezc1, ezc2(jj), ezc3, ezc7(jj), beta_j);
    else
        EV_next = EV_raw;
        BellmanCombiner = @(F, EV_cont) F + beta_j .* EV_cont;
    end

    % ---------------------------------------------------------------------
    % 4. Memory Throttling Bounds (Hoisted across all kernels)
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
    if vfoptions.gridinterplayer == 1
        n_pol_rows = 3;
        if has_e
            V_j_all   = zeros(n_a_work, n_z_work, n_e_work, 'like', a_work);
            Pol_j_all = zeros(n_pol_rows, n_a_work, n_z_work, n_e_work, 'like', a_work);
        else
            V_j_all   = zeros(n_a_work, n_z_work, 'like', a_work);
            Pol_j_all = zeros(n_pol_rows, n_a_work, n_z_work, 'like', a_work);
        end
    else
        if has_e
            V_j_all   = zeros(n_a_work, n_z_work, n_e_work, 'like', a_work);
            Pol_j_all = zeros(n_a_work, n_z_work, n_e_work, 'like', a_work);
        else
            V_j_all   = zeros(n_a_work, n_z_work, 'like', a_work);
            Pol_j_all = zeros(n_a_work, n_z_work, 'like', a_work);
        end
    end

    % ---------------------------------------------------------------------
    % 5. Nested Shocks Iteration
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
                e_slice = e_work(1, 1, e_iter);
                n_e_slice = 1;
                e_idx_range = e_iter;
                
                % Slice each coordinate in E_cells to the current e_iter point: size [1, 1, 1]
                E_cells_slice = cell(1, length(E_cells));
                for i_e = 1:length(E_cells)
                    E_cells_slice{i_e} = E_cells{i_e}(1, 1, e_iter);
                end
            else
                e_slice = e_work;
                n_e_slice = n_e_work;
                e_idx_range = 1:n_e_work;
                E_cells_slice = E_cells;
            end
    
            % Construct unified kernel adapter for this e-slice
            if has_d && has_z && has_e
                eval_kernel = @(d_in, apr_in, A_cells, z_in) ReturnFn(...
                    D_cells{:}, apr_in, A_cells{:}, Z_cells{:}, E_cells_slice{:}, ReturnFnParamsVec{:});
            elseif has_d && has_z && ~has_e
                eval_kernel = @(d_in, apr_in, A_cells, z_in) ReturnFn(...
                    D_cells{:}, apr_in, A_cells{:}, Z_cells{:}, ReturnFnParamsVec{:});
            elseif ~has_d && has_z && has_e
                eval_kernel = @(d_in, apr_in, A_cells, z_in) ReturnFn(...
                    apr_in, A_cells{:}, Z_cells{:}, E_cells_slice{:}, ReturnFnParamsVec{:});
            elseif ~has_d && has_z && ~has_e
                eval_kernel = @(d_in, apr_in, A_cells, z_in) ReturnFn(...
                    apr_in, A_cells{:}, Z_cells{:}, ReturnFnParamsVec{:});
            elseif has_d && ~has_z && ~has_e
                eval_kernel = @(d_in, apr_in, A_cells, z_in) ReturnFn(...
                    D_cells{:}, apr_in, A_cells{:}, ReturnFnParamsVec{:});
            else
                eval_kernel = @(d_in, apr_in, A_cells, z_in) ReturnFn(...
                    apr_in, A_cells{:}, ReturnFnParamsVec{:});
            end

            % -------------------------------------------------------------
            % 6. Method Dispatch (Consumes standard n_z_slice, EV_slice)
            % -------------------------------------------------------------
            if vfoptions.divideandconquer == 1 && vfoptions.gridinterplayer == 1
                [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_DC1_GI1(...
                    eval_kernel, BellmanCombiner, EV_slice, A_mat, z_slice, d_work, ...
                    n_a_work, n_z_slice, n_d_work, pi_z_j, ReturnFnParamsVec, vfoptions);
                % Pol_sub contains:
                % Row 1: Optimal coarse asset index a'_opt
                % Row 2: Optimal subgrid index tau_opt
                % If n_d == 0, evaluate optimal continuous choice h* on the optimal policy grid
                if N_d == 0 && nargout(ReturnFn) >= 2
                    % Reconstruct optimal continuous a'
                    a_diff_j = [diff(a_work); 0];
                    tau_step = (Pol_sub(2, :) - 1) ./ G;
                    apr_star = a_work(Pol_sub(1, :)) + a_diff_j(Pol_sub(1, :)) .* tau_step;

                    [~, h_star] = ReturnFn(apr_star, a_grid_expanded, z_grid_expanded, e_grid_expanded, ReturnFnParamsVec{:});

                    % Store h* into Policy row 1 or as a separate Policy array
                    Pol_j_all(1, :, z_idx_range, e_idx_range) = reshape(h_star, [1, n_a_work, n_z_slice, n_e_slice]);
                end

            elseif vfoptions.gridinterplayer == 1
                [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_GI1_raw(...
                    eval_kernel, BellmanCombiner, EV_slice, A_mat, z_slice, d_work, ...
                    n_a_work, n_z_slice, n_d_work, pi_z_j, ReturnFnParamsVec, vfoptions);

            elseif vfoptions.divideandconquer == 1
                [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_DC1(...
                    eval_kernel, BellmanCombiner, EV_slice, A_mat, z_slice, d_work, ...
                    n_a_work, n_z_slice, n_d_work, pi_z_j, vfoptions);

            else
                % pi_z_j has already done its work
                [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_raw(...
                    eval_kernel, BellmanCombiner, EV_slice, A_mat, z_slice, d_work,  ...
                    n_a_work, n_z_slice, n_d_work, vfoptions);
            end

            % Store results into period containers
            if vfoptions.gridinterplayer == 1
                if has_e
                    V_j_all(:, z_idx_range, e_idx_range)       = reshape(V_sub, [n_a_work, n_z_slice, n_e_slice]);
                    Pol_j_all(:, :, z_idx_range, e_idx_range) = reshape(Pol_sub, [n_pol_rows, n_a_work, n_z_slice, n_e_slice]);
                else
                    V_j_all(:, z_idx_range)       = reshape(V_sub, [n_a_work, n_z_slice]);
                    Pol_j_all(:, :, z_idx_range) = reshape(Pol_sub, [n_pol_rows, n_a_work, n_z_slice]);
                end
            else
                if has_e
                    V_j_all(:, z_idx_range, e_idx_range)     = reshape(V_sub, [n_a_work, n_z_slice, n_e_slice]);
                    Pol_j_all(:, z_idx_range, e_idx_range)   = reshape(Pol_sub, [n_a_work, n_z_slice, n_e_slice]);
                else
                    V_j_all(:, z_idx_range)     = reshape(V_sub, [n_a_work, n_z_slice]);
                    Pol_j_all(:, z_idx_range)   = reshape(Pol_sub, [n_a_work, n_z_slice]);
                end
            end
        end
    end

    V(:, :, :, jj) = V_j_all;
    if vfoptions.gridinterplayer == 1
        PolicyKron(:, :, :, :, jj) = Pol_j_all;
    else
        PolicyKron(:, :, :, jj) = Pol_j_all;
    end
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
    Policy = UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_daprime, n_a, n_all_z, n_e_work, N_j, vfoptions);
    % Flatten compound Z and E dimensions for downstream toolkit compatibility
    Policy = reshape(Policy, [size(Policy, 1), n_a_work, n_z_work, n_e_work, N_j]);
    V = reshape(V, [n_a_work, n_z_work, n_e_work, N_j]);
elseif has_z && ~has_e
    Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_daprime, n_a, n_all_z, N_j, vfoptions);
    Policy = reshape(Policy, [size(Policy, 1), n_a_work, n_z_work, N_j]);
    V = reshape(V, [n_a_work, n_z_work, N_j]);
elseif ~has_z && has_e
    Policy = UnKronPolicyIndexes1_FHorz_e(PolicyKron, n_daprime, n_a, n_e_work, N_j, vfoptions);
    Policy = reshape(Policy, [size(Policy, 1), n_a_work, n_e_work, N_j]);
    V = reshape(V, [n_a_work, n_e_work, N_j]);
else
    Policy = UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_daprime, n_a, N_j, vfoptions);
    Policy = reshape(Policy, [size(Policy, 1), n_a_work, N_j]);
    V = reshape(V, [n_a_work, N_j]);
end

varargout{1} = V;
varargout{2} = Policy;


end