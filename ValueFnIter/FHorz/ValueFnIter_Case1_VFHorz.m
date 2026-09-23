function varargout=ValueFnIter_Case1_VFHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
%% Check which vfoptions have been used, set all others to defaults
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    vfoptions.verbose=0;
    vfoptions.divideandconquer=0;
    vfoptions.gridinterplayer=0;
    vfoptions.lowmemory=0;
    vfoptions.incrementaltype=0;
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
    vfoptions.parallel=1+(gpuDeviceCount>0);
    vfoptions.outputkron=0;
    vfoptions.alreadygridvals=0;
    vfoptions.alreadygridvals_semiexo=0;
    vfoptions.precision = underlyingType(a_grid);
else
    if ~isfield(vfoptions,'verbose'); vfoptions.verbose=0; end
    if ~isfield(vfoptions,'divideandconquer'); vfoptions.divideandconquer=0; end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    elseif vfoptions.gridinterplayer(1)==1
        if ~isfield(vfoptions,'ngridinterp')
            error('When using vfoptions.gridinterplayer=1 you must set vfoptions.ngridinterp')
        end
    end
    if ~isfield(vfoptions,'lowmemory'); vfoptions.lowmemory=0; end
    if ~isfield(vfoptions,'incrementaltype'); vfoptions.incrementaltype=0; end
    if ~isfield(vfoptions,'exoticpreferences'); vfoptions.exoticpreferences='None'; end
    if ~isfield(vfoptions,'dynasty'); vfoptions.dynasty=0; end
    if ~isfield(vfoptions,'experienceasset'); vfoptions.experienceasset=0; end
    if ~isfield(vfoptions,'experienceassetu'); vfoptions.experienceassetu=0; end
    if ~isfield(vfoptions,'experienceassete'); vfoptions.experienceassete=0; end
    if ~isfield(vfoptions,'experienceassetz'); vfoptions.experienceassetz=0; end
    if ~isfield(vfoptions,'experienceassetze'); vfoptions.experienceassetze=0; end
    if ~isfield(vfoptions,'experienceassetsemiz'); vfoptions.experienceassetsemiz=0; end
    if ~isfield(vfoptions,'riskyasset'); vfoptions.riskyasset=0; end
    if ~isfield(vfoptions,'residualasset'); vfoptions.residualasset=0; end
    if ~isfield(vfoptions,'n_ambiguity'); vfoptions.n_ambiguity=0; end
    if ~isfield(vfoptions,'n_e'); vfoptions.n_e=0; end
    if ~isfield(vfoptions,'n_semiz'); vfoptions.n_semiz=0; end
    if ~isfield(vfoptions,'parallel'); vfoptions.parallel=1+(gpuDeviceCount>0); end
    if ~isfield(vfoptions,'outputkron'); vfoptions.outputkron=0; end
    if ~isfield(vfoptions,'alreadygridvals'); vfoptions.alreadygridvals=0; end
    if ~isfield(vfoptions,'alreadygridvals_semiexo'); vfoptions.alreadygridvals_semiexo=0; end
    if ~isfield(vfoptions,'precision'); vfoptions.precision = underlyingType(a_grid); end
end

if isempty(n_d)
    error('If you have no d (decision) variables, set n_d=0;')
end
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~all(size(d_grid)==[sum(n_d), 1])
    if ~isempty(n_d) % Make sure d is being used before complaining about size of d_grid
        if n_d~=0
            error('d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
        end
    end
end
if ~all(size(a_grid)==[sum(n_a), 1])
    error('a_grid is not the correct shape (should be of size sum(n_a)-by-1; a (stacked) column vector)')
end

%% z_grid/pi_z/e_grid/pi_e shape validation is performed inside ExogShockSetup_FHorz (called below).

if vfoptions.parallel<2
    if N_e>0
        error('Sorry but e (i.i.d) variables are not implemented for cpu, you will need a gpu to use them')
    end
    if prod(vfoptions.n_semiz)>0
        error('Sorry but Semi-Exogenous states are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.divideandconquer==0
        error('Sorry but divideandconquer is not implemented for cpu, you will need a gpu to use this algorithm')
    end
    if ~strcmp(vfoptions.exoticpreferences,'None')
        error('Sorry but exoticpreferences are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.experienceasset==0 || ~vfoptions.experienceassetu==0  || ~vfoptions.experienceassetz==0  || ~vfoptions.experienceassete==0  || ~vfoptions.experienceassetze==0  || ~vfoptions.experienceassetsemiz==0
        error('Sorry but experience assets are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.riskyasset==0
        error('Sorry but riskyasset are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.residualasset==0
        error('Sorry but residualasset are not implemented for cpu, you will need a gpu to use them')
    end
    if ~vfoptions.dynasty==0
        error('Sorry but dynasty are not implemented for cpu, you will need a gpu to use them')
    end
end


%%
if vfoptions.parallel==2
    % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    z_grid=gpuArray(z_grid);
    pi_z=gpuArray(pi_z);
    if size(d_grid,2)==1
        d_gridvals=CreateGridvals(n_d,d_grid,1);
    else % already d_gridvals
        d_gridvals=d_grid;
    end
else
    % CPU can be used, but only for the basics. Is kept separate here so that the rest of the codes can just assume you have GPU and work with it.
    [V,Policy]=ValueFnIter_FHorz_CPU(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V,Policy};
    return
end

% Let VFIToolkit's native parser slice the grids and define l_a2 / l_d2
vfoptions = SetupNonStandardEndoStates_FHorz(n_d, n_a, d_grid, a_grid, vfoptions);

% --- SMART nargin PARSER ---
if isempty(ReturnFnParamNames)
    if isfield(vfoptions, 'ReturnFnParamNames')
        ReturnFnParamNames = vfoptions.ReturnFnParamNames;
    else
        temp = getAnonymousFnInputNames(ReturnFn);
        if isequal(n_d, 0) || isempty(n_d); num_d_vars = 0; else; num_d_vars = length(n_d); end
        if isequal(n_z, 0) || isempty(n_z); num_z_vars = 0; else; num_z_vars = length(n_z); end

        l_a_exp = 0;
        if vfoptions.experienceasset > 0; l_a_exp = vfoptions.experienceasset; end
        if vfoptions.experienceassetz > 0; l_a_exp = vfoptions.experienceassetz; end
        n_a2 = l_a_exp;
        l_a1 = length(n_a) - n_a2;

        num_semiz_vars = 0; if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0; num_semiz_vars = length(vfoptions.n_semiz); end
        num_e_vars = 0; if isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0; num_e_vars = length(vfoptions.n_e); end
        num_u_vars = 0; if vfoptions.riskyasset == 1 && isfield(vfoptions, 'n_u'); num_u_vars = length(vfoptions.n_u); end

        if vfoptions.riskyasset == 1
            num_d1 = 0; if length(vfoptions.refine_d) >= 1; num_d1 = vfoptions.refine_d(1); end
            num_d3 = 0; if length(vfoptions.refine_d) >= 3; num_d3 = vfoptions.refine_d(3); end
            num_prefix_args = num_d1 + num_d3 + 1 + num_semiz_vars + num_z_vars;
        else
            num_prefix_args = num_d_vars + (2 * l_a1) + n_a2 + num_semiz_vars + num_z_vars + num_e_vars + num_u_vars;
        end
        if length(temp) > num_prefix_args; ReturnFnParamNames = {temp{num_prefix_args + 1 : end}}; else; ReturnFnParamNames = {}; end
        ReturnFnParamNames = ReturnFnParamNames(isfield(Parameters, ReturnFnParamNames));
    end
end

is_EZ = strcmp(vfoptions.exoticpreferences, 'EpsteinZin') || strcmp(vfoptions.exoticpreferences, 'QHEpsteinZin');

if isfield(vfoptions,'survivalprobability')
    sj=Parameters.(vfoptions.survivalprobability);
elseif isfield(vfoptions,'WarmGlowBequestsFn')
    sj=ones(N_j,1); sj(end)=0;
else
    sj=ones(N_j,1);
end

if isfield(vfoptions,'WarmGlowBequestsFn')
    warmglow=1;
    temp=getAnonymousFnInputNames(vfoptions.WarmGlowBequestsFn);
    vfoptions.WarmGlowBequestsFnParamsNames={temp{2:end}};
else
    warmglow=0;
end

if is_EZ
    vfoptions = EpsteinZinSetup_VFHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions);
end

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        if isscalar(n_a)
            vfoptions.level1n=floor(sqrt(n_a(1)));
        elseif length(n_a)>=2
            vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2:end)];
        end
    else
        if ~isscalar(n_a) && isscalar(vfoptions.level1n)
            vfoptions.level1n=[vfoptions.level1n,n_a(2:end)];
        end
    end
end

if vfoptions.alreadygridvals==0
    [z_gridvals_J, pi_z_J, vfoptions] = ExogShockSetup_FHorz(n_z, z_grid, pi_z, N_j, Parameters, vfoptions, 3, 0);
else
    z_gridvals_J = z_grid;
    pi_z_J = pi_z;
end
if isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0; N_semiz = prod(vfoptions.n_semiz); else; N_semiz = 0; end
if vfoptions.alreadygridvals_semiexo==0
    if N_semiz > 0; vfoptions = SemiExogShockSetup_FHorz(n_d, N_j, d_grid, Parameters, vfoptions, 3); end
end

if vfoptions.parallel == 2
    z_gridvals_J = gpuArray(z_gridvals_J);
    pi_z_J = gpuArray(pi_z_J);
    if isfield(vfoptions, 'e_gridvals_J'); vfoptions.e_gridvals_J = gpuArray(vfoptions.e_gridvals_J); end
    if isfield(vfoptions, 'semiz_gridvals_J'); vfoptions.semiz_gridvals_J = gpuArray(vfoptions.semiz_gridvals_J); end
    if isfield(vfoptions, 'pi_e_J'); vfoptions.pi_e_J = gpuArray(vfoptions.pi_e_J); end
    if isfield(vfoptions, 'pi_semiz_J'); vfoptions.pi_semiz_J = gpuArray(vfoptions.pi_semiz_J); end
end

N_d = prod(n_d); N_a = prod(n_a); N_z = prod(n_z); N_z_safe = max(1, N_z);
if N_semiz > 0 && isfield(vfoptions, 'semiz_gridvals_J')
    sz_J = vfoptions.semiz_gridvals_J; num_semiz_vars = size(sz_J, 2); num_periods = size(sz_J, 3);
    if N_z > 0; num_z_vars = size(z_gridvals_J, 2); else; num_z_vars = 0; end
    z_gridvals_J_combined = zeros(N_semiz * max(1, N_z), num_semiz_vars + num_z_vars, num_periods, 'like', sz_J);
    for t = 1:num_periods
        if N_z > 0
            semiz_expanded = kron(ones(N_z, 1), sz_J(:,:,t));
            z_expanded = kron(z_gridvals_J(:,:,t), ones(N_semiz, 1));
            z_gridvals_J_combined(:,:,t) = [semiz_expanded, z_expanded];
        else
            z_gridvals_J_combined(:,:,t) = sz_J(:,:,t);
        end
    end
    z_gridvals_J = z_gridvals_J_combined; n_combined_z = [vfoptions.n_semiz, n_z];
else
    n_combined_z = n_z;
end

if strcmp(vfoptions.exoticpreferences, 'QuasiHyperbolic') || strcmp(vfoptions.exoticpreferences, 'QHEpsteinZin')
    if nargout == 4
        [V, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_combined_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout = {V, Policy, Valt, Policyalt};
    else
        [V, Policy, Valt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_combined_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout = {V, Policy, Valt, []};
    end
    return;
end

% ---------------------------------------------------------------------
% MULTI-AXIS STATE PARSER: Leverage Native Split Grids & Counts
% ---------------------------------------------------------------------
has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
n_e_pass = 0;
e_grid_pass = [];
if has_e; n_e_pass = vfoptions.n_e; e_grid_pass = vfoptions.e_grid; e_work = vfoptions.e_grid; else; e_work = ones(1, 1, 'like', a_grid); end

% Extract active flags
l_exp_base  = vfoptions.experienceasset >= 1;
l_exp_u     = vfoptions.experienceassetu >= 1;
l_exp_z     = vfoptions.experienceassetz >= 1;
l_exp_e     = vfoptions.experienceassete >= 1;
l_exp_ze    = vfoptions.experienceassetze >= 1;
l_exp_semiz = vfoptions.experienceassetsemiz >= 1;
is_exp_asset = l_exp_base || l_exp_u || l_exp_z || l_exp_e || l_exp_ze || l_exp_semiz;

% Trust SetupNonStandardEndoStates for all geometry bounds
if is_exp_asset || vfoptions.riskyasset == 1 || vfoptions.residualasset == 1
    a1_endo_grid_vals = vfoptions.a1_grid;
    a2_exp_grid_vals  = vfoptions.a2_grid;
    n_a1_dc = vfoptions.n_a1(1);

    if isequal(vfoptions.n_a1, 0)
        l_a1 = 0;
    else
        l_a1 = length(vfoptions.n_a1);
    end

    if isequal(vfoptions.n_d1, 0)
        l_d1 = 0;
    else
        l_d1 = length(vfoptions.n_d1);
    end

    if l_a1 > 1
        n_a1_other = vfoptions.n_a1(2:end);
    else
        n_a1_other = [];
    end

    n_a2 = vfoptions.n_a2;
    l_a2 = length(n_a2);
else
    a1_endo_grid_vals = a_grid;
    a2_exp_grid_vals  = [];
    n_a1_dc = n_a(1);

    l_a1 = length(n_a);
    l_d1 = length(n_d);

    if l_a1 > 1
        n_a1_other = n_a(2:end);
    else
        n_a1_other = [];
    end

    n_a2 = [];
    l_a2 = 0;
end

N_a1_dc = n_a1_dc;
N_a1_other = max(1, prod(n_a1_other));
N_a2 = max(1, prod(n_a2));

A1_grids_1d = cell(1, l_a1);
offset = 0;
for i = 1:l_a1
    A1_grids_1d{i} = a1_endo_grid_vals((offset + 1):(offset + n_a(i)));
    offset = offset + n_a(i);
end

% 1. Create the master TensorReturnFn using n_daprime so the signature expects (d, aprime, a1, a2, z, e)
if isempty(n_d) || isequal(n_d, 0)
    n_daprime_sig = n_a(1:l_a1);
else
    n_daprime_sig = [n_d, n_a(1:l_a1)];
end
[TensorReturnFn, ~, ~, ~, ~] = CreateTensorFnAndCells(ReturnFn, n_daprime_sig, n_a, n_combined_z, n_e_pass, [], [], [], []);

% 2. Generate strictly separated D, A1, Z, and E cells using only l_a1 to prevent cross-meshing memory blowouts
[~, D_cells_block, A1_cells, Z_cells_block, E_cells_block] = CreateTensorFnAndCells(ReturnFn, n_d, n_a(1:l_a1), n_combined_z, n_e_pass, d_grid, a1_endo_grid_vals, [], []);

if isfield(vfoptions, 'gpu') && vfoptions.gpu == 1
    for i = 1:length(D_cells_block); D_cells_block{i} = gpuArray(D_cells_block{i}); end
    for i = 1:length(Z_cells_block); Z_cells_block{i} = gpuArray(Z_cells_block{i}); end
    for i = 1:length(E_cells_block); E_cells_block{i} = gpuArray(E_cells_block{i}); end
end

if l_a2 > 0
    n_z_pass_exp = 0;
    if l_exp_z || l_exp_ze
        n_z_pass_exp = n_z;
    end

    n_e_pass_exp = 0;
    if l_exp_e || l_exp_ze
        n_e_pass_exp = n_e_pass;
    end

    [TensoraprimeFn, ~, A2_cells, ~, ~] = CreateTensorFnAndCells(vfoptions.aprimeFn, vfoptions.n_d2, n_a2, n_z_pass_exp, n_e_pass_exp, [], a2_exp_grid_vals, [], []);
else
    TensoraprimeFn = [];
    A2_cells = {};
end

A1_mat = zeros(N_a1_dc * N_a1_other, l_a1, 'like', a_grid);
for i_a = 1:l_a1
    A1_mat(:, i_a) = A1_cells{i_a}(:);
end

A2_mat = zeros(N_a2, l_a2, 'like', a_grid);
a2_grids_1d = cell(1, l_a2);
offset = 0;
for i_a = 1:l_a2
    A2_mat(:, i_a) = A2_cells{i_a}(:);
    a2_grids_1d{i_a} = a2_exp_grid_vals((offset + 1):(offset + n_a2(i_a)));
    offset = offset + n_a2(i_a);
end

for i_d = 1:length(D_cells_block); D_cells_block{i_d} = reshape(D_cells_block{i_d}, [max(1,prod(n_d)), 1, 1, 1, 1]); end

if is_exp_asset || vfoptions.riskyasset == 1
    aprimeFn = vfoptions.aprimeFn;
    if isfield(vfoptions, 'aprimeFnParamNames')
        aprimeFnParamNames = vfoptions.aprimeFnParamNames;
    else
        temp = getAnonymousFnInputNames(aprimeFn);

        num_extra = 0;
        if l_exp_z;     num_extra = length(n_z); end
        if l_exp_e;     num_extra = length(vfoptions.n_e); end
        if l_exp_ze;    num_extra = length(n_z) + length(vfoptions.n_e); end
        if l_exp_u;     num_extra = length(vfoptions.n_u); end
        if l_exp_semiz; num_extra = length(vfoptions.n_semiz); end

        if vfoptions.riskyasset == 1
            l_d_aprime = length(n_d);
            l_a_aprime = 1;
        else
            l_d_aprime = vfoptions.l_d2;
            l_a_aprime = vfoptions.l_a2;
        end

        num_prefix = l_d_aprime + l_a_aprime + num_extra;

        if length(temp) > num_prefix
            aprimeFnParamNames = {temp{num_prefix+1:end}};
        else
            aprimeFnParamNames = {};
        end
    end
    aprimeFnParamNames = aprimeFnParamNames(isfield(Parameters, aprimeFnParamNames));

    % Smart Wrapper: Isolate the exact decisions that drive the non-standard asset
    if is_exp_asset
        d2_idx = (l_d1 + 1) : (l_d1 + vfoptions.l_d2);
    else
        d2_idx = 1:length(n_d); % Risky assets evaluate all decisions
    end

    BaseTensoraprimeFn = TensoraprimeFn;
    if l_exp_ze
        TensoraprimeFn = @(D, A, Z, E, P) BaseTensoraprimeFn(D{d2_idx}, A{:}, Z{:}, E{:}, P{:});
    elseif l_exp_z
        TensoraprimeFn = @(D, A, Z, E, P) BaseTensoraprimeFn(D{d2_idx}, A{:}, Z{:}, P{:});
    elseif l_exp_e
        TensoraprimeFn = @(D, A, Z, E, P) BaseTensoraprimeFn(D{d2_idx}, A{:}, E{:}, P{:});
    else
        TensoraprimeFn = @(D, A, Z, E, P) BaseTensoraprimeFn(D{d2_idx}, A{:}, P{:});
    end
else
    aprimeFn = [];
    aprimeFnParamNames = {};
end

N_d_safe = max(1, prod(n_d)); n_a_work = prod(n_a);

has_semiz = prod(vfoptions.n_semiz) > 0;
if has_semiz
    if length(n_z) >= length(vfoptions.n_semiz) && isequal(n_z(1:length(vfoptions.n_semiz)), vfoptions.n_semiz)
        N_semiz = prod(vfoptions.n_semiz); n_all_z = n_z; N_z_exog = max(1, prod(n_z) / N_semiz);
    else
        N_semiz = prod(vfoptions.n_semiz); n_all_z = [vfoptions.n_semiz, n_z]; N_z_exog = max(1, prod(n_z));
    end
else
    N_semiz = 1; n_all_z = n_z; N_z_exog = max(1, prod(n_z));
end
has_z = prod(n_z) > 0; n_z_work = N_semiz * N_z_exog; n_e_work = max(1, prod(n_e_pass)); N_ze = n_z_work * n_e_work;

if vfoptions.gridinterplayer(1) == 1
    PolicyKron = zeros(3, n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
else
    PolicyKron = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid);
end
V = zeros(n_a_work, n_z_work, n_e_work, N_j, 'like', a_grid); V_next = zeros(n_a_work, n_z_work, n_e_work, 'like', a_grid);

% --- Grid Interpolation Setup (Strictly bounds A1_DC) ---
if vfoptions.gridinterplayer(1) == 1
    n2short = vfoptions.ngridinterp; n2long  = n2short * 2 + 3;
    a1_dc_grid = A1_grids_1d{1};
    a1prime_grid = interp1(1:1:N_a1_dc, a1_dc_grid, linspace(1, N_a1_dc, N_a1_dc + (N_a1_dc - 1) * n2short))';
    idx = discretize(a1prime_grid, a1_dc_grid); idx(isnan(idx) | idx == length(a1_dc_grid)) = length(a1_dc_grid) - 1;
    interp_left_idx = idx(:); interp_right_idx = idx(:) + 1;
    a1_left = a1_dc_grid(interp_left_idx);
    a1_right = a1_dc_grid(interp_right_idx);
    interp_weights = (a1prime_grid(:) - a1_left) ./ (a1_right - a1_left);
    interp_weights(a1_right == a1_left) = 0;
    interp_weights(abs(interp_weights) < 1e-12) = 0;
    interp_weights(abs(interp_weights - 1) < 1e-12) = 1;
    interp_left_idx = gpuArray(interp_left_idx);
    interp_right_idx = gpuArray(interp_right_idx);
    interp_weights = gpuArray(interp_weights);
    a1prime_grid = gpuArray(a1prime_grid);
    for i = 1:l_a1; A1_grids_1d{i} = gpuArray(A1_grids_1d{i}); end
    if l_a2 > 0
        for i = 1:l_a2; a2_grids_1d{i} = gpuArray(a2_grids_1d{i}); end
    end
else
    n2short = 0; n2long  = 0; a1prime_grid = []; interp_left_idx = []; interp_right_idx = []; interp_weights = [];
end

if is_EZ
    ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4; ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; ezc8 = vfoptions.ezc8;
else
    ezc2 = ones(N_j,1); ezc3 = 1; ezc4 = 1; ezc5 = ones(N_j,1); ezc6 = ones(N_j,1); ezc7 = ones(N_j,1); ezc8 = ones(N_j,1);
end

if vfoptions.riskyasset == 1
    disp('V-World: Dispatching Risky Asset model to Tensor Bridge...');
    if length(n_a) > 1
        pass_n_a1 = n_a(1:end-1); pass_n_a2 = n_a(end);
        a1_grid_len = sum(pass_n_a1); pass_a1_grid = a_grid(1:a1_grid_len); pass_a2_grid = a_grid(a1_grid_len+1:end);
    else
        pass_n_a1 = []; pass_n_a2 = n_a; pass_a1_grid = []; pass_a2_grid = a_grid;
    end
    [V, Policy] = ValueFnIter_VFHorz_RiskyAsset_EpsteinZin(...
        n_d, pass_n_a1, pass_n_a2, n_combined_z, vfoptions.n_u, N_j, ...
        d_grid, pass_a1_grid, pass_a2_grid, z_gridvals_J, vfoptions.u_grid, pi_z_J, vfoptions.pi_u, ...
        ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ...
        ReturnFnParamNames, aprimeFnParamNames, vfoptions, ...
        sj, warmglow, ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8);
    varargout{1} = V; varargout{2} = Policy; if nargout > 2, varargout{3} = []; end; if nargout > 3, varargout{4} = []; end
    return;
end

if ismember(vfoptions.lowmemory, [0, 5]); ze_chunks = {1:N_ze};
elseif vfoptions.lowmemory == 1
    e_chunk_size = max(1, floor(300 / n_z_work)); num_chunks = ceil(n_e_work / e_chunk_size); ze_chunks = cell(1, num_chunks);
    for c = 1:num_chunks
        e_start = (c-1)*e_chunk_size + 1; e_end   = min(c*e_chunk_size, n_e_work);
        [Z_sub, E_sub] = ndgrid(1:n_z_work, e_start:e_end); ze_chunks{c} = sub2ind([n_z_work, n_e_work], Z_sub(:), E_sub(:))';
    end
else; ze_chunks = num2cell(1:N_ze); end

if ismember(vfoptions.lowmemory, [4, 5]) && n_a2 > 0; a2_chunks = num2cell(1:N_a2); else; a2_chunks = {1:N_a2}; end

chunk_meta = cell(1, length(ze_chunks));
for i_ze = 1:length(ze_chunks)
    c_ze = ze_chunks{i_ze}; if isa(c_ze, 'gpuArray'), c_ze_cpu = gather(c_ze); else, c_ze_cpu = c_ze; end
    [z_ind, e_ind] = ind2sub([n_z_work, n_e_work], c_ze_cpu);
    meta.z_vals = gpuArray(unique(z_ind));
    meta.e_vals = gpuArray(unique(e_ind));
    meta.n_z_loc = length(meta.z_vals); meta.n_e_loc = length(meta.e_vals); meta.N_ze_local = length(c_ze);
    meta.z_offset_local = reshape((0:meta.N_ze_local-1) * N_a, [1, 1, 1, meta.N_ze_local]);
    if vfoptions.gridinterplayer(1) == 1; meta.z_offset_fine_local = reshape((0:meta.N_ze_local-1) * length(a1prime_grid), [1, 1, 1, meta.N_ze_local]); else; meta.z_offset_fine_local = []; end
    chunk_meta{i_ze} = meta;
end

base_ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);
is_age_dependent = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j; is_age_dependent(ip) = true; end
    if isnumeric(base_ReturnFnParamsCell{ip}) && ~isa(base_ReturnFnParamsCell{ip}, 'gpuArray'); base_ReturnFnParamsCell{ip} = gpuArray(base_ReturnFnParamsCell{ip}); end
end

N_semiz_local = 1; N_dsemiz = 1;
if has_semiz && length(n_d) > 0
    N_semiz_local = max(1, prod(vfoptions.n_semiz));
    if isfield(vfoptions, 'l_dsemiz'); N_dsemiz = prod(n_d(end-vfoptions.l_dsemiz+1:end)); else; N_dsemiz = n_d(end); end
end
N_z_exog = max(1, n_z_work / N_semiz_local);

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose==1; fprintf('Finite horizon: %i of %i \n',jj, N_j); end
    ReturnFnParamsCell = base_ReturnFnParamsCell; pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    for ip = find(is_age_dependent)
        ReturnFnParamsCell{ip} = cast(Parameters.(ReturnFnParamNames{ip})(jj), 'like', a_grid);
    end
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision); beta_j = prod(DiscountFactorParamsVec);
    if l_a_exp > 0; aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj); else; aprimeFnParamsCell = {}; end

    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(vfoptions.V_Jplus1, [n_a_work, n_z_work, n_e_work]);
        V_next = gpuArray(V_next);
    end

    if jj == N_j && (~isfield(vfoptions, 'V_Jplus1') || isempty(vfoptions.V_Jplus1))
        if warmglow == 1
            wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            WG_eval = vfoptions.WarmGlowBequestsFn(a_grid, wg_params{:});
            if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a_grid), 'like', a_grid); end
            if is_EZ
                valid_wg = isfinite(WG_eval) & (WG_eval ~= 0); WG_transformed = WG_eval;
                if ezc5(jj) == 1; WG_transformed(valid_wg) = ezc4 * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4 * WG_eval(valid_wg), 0).^ezc5(jj); end
                WG_transformed(WG_eval == 0) = 0; WG_eval = WG_transformed;
            end
            EV = repmat(reshape(WG_eval, [N_a, 1, 1, 1]), [1, N_semiz_local * N_z_exog, n_e_work, N_dsemiz]);
        else
            EV = zeros(N_a, N_semiz_local * N_z_exog, n_e_work, N_dsemiz, 'like', a_grid);
        end
        V_next = zeros(n_a_work, n_z_work, n_e_work, 'like', a_grid);
    else
        if has_e && isfield(vfoptions, 'e_gridvals_J'); e_work = vfoptions.e_gridvals_J(:, :, min(jj, size(vfoptions.e_gridvals_J, 3))); end
        valid_V = isfinite(V_next) & (V_next ~= 0); V_transformed = V_next;
        if ezc5(jj) == 1; V_transformed(valid_V) = ezc4 * V_next(valid_V); else; V_transformed(valid_V) = max(ezc4 * V_next(valid_V), 0).^ezc5(jj); end
        V_transformed(V_next == 0) = 0;

        if has_e
            if isfield(vfoptions, 'pi_e_J'); pi_e_j = vfoptions.pi_e_J(:, min(jj + 1, size(vfoptions.pi_e_J, 2))); else; pi_e_j = vfoptions.pi_e; end
            pi_e_j = gpuArray(pi_e_j);
            V_trans_flat = reshape(V_transformed, [N_a * n_z_work, n_e_work]);
            V_inf_mask = (V_trans_flat == -Inf); V_safe = V_trans_flat; V_safe(V_inf_mask) = -1e250;
            V_expected_e = V_safe * pi_e_j(:);
            inf_restore = (V_inf_mask * (pi_e_j(:) > 0)) > 0; V_expected_e(inf_restore) = -Inf;
            V_transformed = repmat(reshape(V_expected_e, [N_a, n_z_work, 1]), [1, 1, n_e_work]);
        end

        EV = zeros(N_a, N_semiz_local * N_z_exog, n_e_work, N_dsemiz, 'like', V_next);
        for ie = 1:n_e_work
            V_curr = V_transformed(:,:,ie);
            if N_z_exog > 1 && has_z
                V_slice = reshape(V_curr, [N_a * N_semiz_local, N_z_exog]);
                V_inf_mask = (V_slice == -Inf); V_safe = V_slice; V_safe(V_inf_mask) = -1e250;
                V_z_eval = V_safe * pi_z_j';
                inf_restore = (V_inf_mask * (pi_z_j' > 0)) > 0; V_z_eval(inf_restore) = -Inf;
                V_z_eval = reshape(V_z_eval, [N_a, N_semiz_local, N_z_exog]);
            else
                V_z_eval = reshape(V_curr, [N_a, N_semiz_local, N_z_exog]);
            end
            if has_semiz
                pi_semiz_j = vfoptions.pi_semiz_J(:, :, :, min(jj, size(vfoptions.pi_semiz_J, 4)));
                V_perm = reshape(permute(V_z_eval, [2, 1, 3]), [N_semiz_local, N_a * N_z_exog]);
                V_inf_mask = (V_perm == -Inf); V_safe = V_perm; V_safe(V_inf_mask) = -1e250;
                for idsemiz = 1:N_dsemiz
                    pi_semiz_d = pi_semiz_j(:, :, idsemiz); EV_perm = pi_semiz_d * V_safe;
                    inf_restore = (pi_semiz_d > 0) * V_inf_mask > 0; EV_perm(inf_restore) = -Inf;
                    EV_d = permute(reshape(EV_perm, [N_semiz_local, N_a, N_z_exog]), [2, 1, 3]);
                    EV(:,:,ie,idsemiz) = reshape(EV_d, [N_a, N_semiz_local * N_z_exog]);
                end
            else; EV(:,:,ie,1) = reshape(V_z_eval, [N_a, N_semiz_local * N_z_exog]); end
        end

        if warmglow == 1
            wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            WG_eval = vfoptions.WarmGlowBequestsFn(a_grid, wg_params{:});
            if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a_grid), 'like', a_grid); end
            if is_EZ
                valid_wg = isfinite(WG_eval) & (WG_eval ~= 0); WG_transformed = WG_eval;
                if ezc5(jj) == 1; WG_transformed(valid_wg) = ezc4 * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4 * WG_eval(valid_wg), 0).^ezc5(jj); end
                WG_transformed(WG_eval == 0) = 0; WG_eval = WG_transformed;
            end
            WG_eval = reshape(WG_eval, [N_a, 1, 1, 1]); EV = EV * sj(jj) + (1 - sj(jj)) * WG_eval;
        end
    end

    valid_EV = isfinite(EV) & (EV ~= 0);
    if ezc6(jj) ~= 1; EV(valid_EV) = max(EV(valid_EV), 0).^ezc6(jj); end
    if ezc8(jj) ~= 1; EV(valid_EV) = max(EV(valid_EV), 0).^ezc8(jj); end
    EV_flat_ze = reshape(EV, [N_a, N_ze, N_dsemiz]);

    V_j_max        = zeros(N_a, N_ze, 'like', V_next);
    Pol_apr_max    = zeros(N_a, N_ze, 'like', V_next);
    Pol_d_max      = zeros(N_a, N_ze, 'like', V_next);
    Pol_L2idx_max  = zeros(N_a, N_ze, 'like', V_next);
    Pol_L2flag_max = zeros(N_a, N_ze, 'like', V_next);

    if N_dsemiz > 1
        if isfield(vfoptions, 'l_dsemiz'); N_d_prefix = max(1, prod(n_d(1:end-vfoptions.l_dsemiz))); else; N_d_prefix = max(1, prod(n_d(1:end-1))); end
        dsemiz_idx = ceil((1:N_d_safe)' / N_d_prefix); dsemiz_idx_tensor = reshape(dsemiz_idx, [N_d_safe, 1, 1, 1]);
    else
        dsemiz_idx_tensor = ones(N_d_safe, 1, 1, 1);
    end

    if vfoptions.divideandconquer == 1
        for i_ze = 1:length(ze_chunks)
            meta = chunk_meta{i_ze}; n_z_loc = meta.n_z_loc; n_e_loc = meta.n_e_loc;
            curr_ze = ze_chunks{i_ze}; N_ze_local = length(curr_ze);
            EV_local = EV_flat_ze(:, curr_ze, :);
            if has_semiz || has_z
                num_z_vars = length(n_combined_z); Z_cells_local = cell(1, num_z_vars);
                if size(z_gridvals_J, 2) ~= num_z_vars
                    z_inflated = reshape(z_gridvals_J, [N_z, num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                    for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
                else
                    for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
                end
            else; Z_cells_local = {}; end
            if has_e
                num_e_vars = size(e_work, 2); E_cells_local = cell(1, num_e_vars);
                for ie_var = 1:num_e_vars; E_cells_local{ie_var} = reshape(e_work(meta.e_vals, ie_var), [1, 1, 1, 1, n_e_loc]); end
            else; E_cells_local = {}; end

            if vfoptions.gridinterplayer(1) == 1
                N_cols = N_ze_local * N_dsemiz; zero_weights = (interp_weights == 0); one_weights = (interp_weights == 1);
                N_a2_rem = N_a1_other * N_a2;
                EV_2d = reshape(EV_local, [N_a1_dc, N_a2_rem * N_cols]);
                EV_left_val = EV_2d(interp_left_idx, :); EV_right_val = EV_2d(interp_right_idx, :);
                EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);
                EV_interp_flat(zero_weights, :) = EV_left_val(zero_weights, :); EV_interp_flat(one_weights, :) = EV_right_val(one_weights, :);
                EV_interp_flat(isnan(EV_interp_flat)) = -Inf;
                EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_a2_rem, N_ze_local, N_dsemiz]);
            else; EV_interp_local = []; end

            if l_a_exp == 0
                % Reshape to expose N_dsemiz, then slice by dsemiz_idx_tensor to safely
                % broadcast EV across endogenous choices and semi-exogenous transitions
                EV_reshaped = reshape(EV_local, [N_a1_dc * N_a1_other, n_z_loc, n_e_loc, N_dsemiz]);
                EV_d_sliced = EV_reshaped(:, :, :, dsemiz_idx_tensor(:));
                EV_bounded_pre = beta_j .* permute(EV_d_sliced, [4, 1, 5, 2, 3]);

                d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]);
                z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1_dc * N_a1_other), [1, 1, 1, n_z_loc, 1]);
                e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1_dc * N_a1_other * n_z_loc), [1, 1, 1, 1, n_e_loc]);
                static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
            else; EV_bounded_pre = []; static_EV_offset = []; end

            vfoptions.level1n = vfoptions.level1n(1);
            LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar, d_gap, dc_mode_override) Evaluate_Case1_TensorBlock(...
                state_idx, loweredge_matrix, maxgap_scalar, d_gap, N_a1_dc, N_a1_other, max(1, N_a2), N_d_safe, N_ze_local, ...
                Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, a2_grids_1d, ...
                vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, dc_mode_override);

            SlicerWrapper = @(a1_idx, low_mat, mg, dc_mode) Helper_SlicerWrapper(a1_idx, low_mat, mg, dc_mode, N_a1_dc, N_a1_other, max(1, N_a2), N_ze_local, max(1, N_d_safe), LocalBlockFn);

            if vfoptions.gridinterplayer(1) == 1
                % --- VRAM Protection: Cartesian Chunking for the COARSE Pass ---
                flat_choices_coarse = max(1, N_d_safe) * N_a1_dc * N_a1_other;
                N_other = N_a1_other * max(1, N_a2);
                max_a1_per_chunk_coarse = max(1, floor(5e8 / (flat_choices_coarse * N_other * N_ze_local)));

                loweredge_pass = zeros(max(1, N_d_safe), N_a1_other, N_a1_dc * max(1, N_a2), N_ze_local, 'like', EV_local);

                for chunk_start = 1:max_a1_per_chunk_coarse:N_a1_dc
                    chunk_end = min(N_a1_dc, chunk_start + max_a1_per_chunk_coarse - 1);
                    a1_chunk = (chunk_start:chunk_end)';
                    state_chunk_mat = a1_chunk + (0:N_other-1) * N_a1_dc;
                    state_chunk = state_chunk_mat(:)';

                    [~, ~, ~, ~, ~, p_a1_per_a2] = LocalBlockFn(state_chunk, [], 0, 0, 2);
                    loweredge_pass(:, :, state_chunk, :) = reshape(p_a1_per_a2, [max(1, N_d_safe), N_a1_other, length(state_chunk), N_ze_local]);
                end

                % --- VRAM Protection: Cartesian Chunking for the Grid Interp Fine Pass ---
                flat_choices = max(1, N_d_safe) * n2long * max(1, N_a1_other);
                max_a1_per_chunk = max(1, floor(5e8 / (flat_choices * N_other * N_ze_local)));

                v = zeros(N_a, N_ze_local, 'like', EV_local);
                p_apr = zeros(N_a, N_ze_local, 'like', EV_local);
                p_d = zeros(N_a, N_ze_local, 'like', EV_local);
                p_l2idx = zeros(N_a, N_ze_local, 'like', EV_local);
                p_l2flag = zeros(N_a, N_ze_local, 'like', EV_local);

                for chunk_start = 1:max_a1_per_chunk:N_a1_dc
                    chunk_end = min(N_a1_dc, chunk_start + max_a1_per_chunk - 1);
                    a1_chunk = (chunk_start:chunk_end)';

                    state_chunk_mat = a1_chunk + (0:N_other-1) * N_a1_dc;
                    state_chunk = state_chunk_mat(:)';

                    loweredge_chunk = loweredge_pass(:, :, state_chunk, :);
                    [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c] = LocalBlockFn(state_chunk, loweredge_chunk, n2long - 1, 0, 1);

                    v(state_chunk, :) = v_c;
                    p_apr(state_chunk, :) = p_apr_c;
                    p_d(state_chunk, :) = p_d_c;
                    p_l2idx(state_chunk, :) = p_l2idx_c;
                    p_l2flag(state_chunk, :) = p_l2flag_c;
                end
            else
                LocalBlockFn_Standard = @(state_idx, loweredge_matrix, maxgap_scalar) SlicerWrapper(state_idx, loweredge_matrix, maxgap_scalar, 0);
                if l_a1 == 1
                    [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(N_a1_dc, N_a1_dc, max(1, N_a2), N_ze_local, vfoptions, LocalBlockFn_Standard);
                else
                    [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC2A_Slicer(N_a1_dc, N_a1_other, N_a1_other * max(1, N_a2), N_a1_dc, N_ze_local, vfoptions, LocalBlockFn_Standard);
                end
            end

            V_j_max(:, curr_ze)     = reshape(v,     [N_a, N_ze_local]);
            Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a, N_ze_local]);
            Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a, N_ze_local]);
            if vfoptions.gridinterplayer(1) == 1
                Pol_L2idx_max(:, curr_ze)  = reshape(p_l2idx,  [N_a, N_ze_local]);
                Pol_L2flag_max(:, curr_ze) = reshape(p_l2flag, [N_a, N_ze_local]);
            end
        end
    else
        % Non-DC block
        for i_a2 = 1:length(a2_chunks)
            curr_a2 = a2_chunks{i_a2}; N_a2_local = length(curr_a2);
            start_a_idx = (min(curr_a2) - 1) * (N_a1_dc * N_a1_other) + 1; end_a_idx   = max(curr_a2) * (N_a1_dc * N_a1_other);
            for i_ze = 1:length(ze_chunks)
                curr_ze = ze_chunks{i_ze}; N_ze_local = length(curr_ze);
                meta = chunk_meta{i_ze}; n_z_loc = meta.n_z_loc; n_e_loc = meta.n_e_loc;
                if l_a_exp > 0; A2_local = A2_mat(curr_a2, :); else; A2_local = []; end
                EV_local = EV_flat_ze(:, curr_ze, :);
                if has_semiz || has_z
                    num_z_vars = length(n_combined_z); Z_cells_local = cell(1, num_z_vars);
                    if size(z_gridvals_J, 2) ~= num_z_vars
                        z_inflated = reshape(z_gridvals_J, [N_z, num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                        for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
                    else
                        for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
                    end
                else; Z_cells_local = {}; end
                if has_e
                    num_e_vars = size(e_work, 2); E_cells_local = cell(1, num_e_vars);
                    for ie = 1:num_e_vars; E_cells_local{ie} = reshape(e_work(meta.e_vals, ie), [1, 1, 1, 1, n_e_loc]); end
                else; E_cells_local = {}; end

                if vfoptions.gridinterplayer(1) == 1
                    N_cols = N_ze_local * N_dsemiz; zero_weights = (interp_weights == 0); one_weights = (interp_weights == 1);
                    if l_a_exp > 0
                        EV_2d = reshape(EV_local, [N_a1_dc, N_a1_other * N_a2_local * N_cols]);
                        EV_left_val = EV_2d(interp_left_idx, :); EV_right_val = EV_2d(interp_right_idx, :);
                        EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);
                        EV_interp_flat(zero_weights, :) = EV_left_val(zero_weights, :); EV_interp_flat(one_weights, :) = EV_right_val(one_weights, :);
                        EV_interp_flat(isnan(EV_interp_flat)) = -Inf;
                        EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_a1_other, N_a2_local, N_ze_local, N_dsemiz]);
                    else
                        EV_2d = reshape(EV_local, [N_a1_dc, N_a1_other * N_cols]);
                        EV_left_val = EV_2d(interp_left_idx, :); EV_right_val = EV_2d(interp_right_idx, :);
                        EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);
                        EV_interp_flat(zero_weights, :) = EV_left_val(zero_weights, :); EV_interp_flat(one_weights, :) = EV_right_val(one_weights, :);
                        EV_interp_flat(isnan(EV_interp_flat)) = -Inf;
                        EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_a1_other, N_ze_local, N_dsemiz]);
                    end
                else; EV_interp_local = []; end

                if l_a_exp == 0
                    % Reshape to expose N_dsemiz, then slice by dsemiz_idx_tensor to safely
                    % broadcast EV across endogenous choices and semi-exogenous transitions
                    EV_reshaped = reshape(EV_local, [N_a1_dc * N_a1_other, n_z_loc, n_e_loc, N_dsemiz]);
                    EV_d_sliced = EV_reshaped(:, :, :, dsemiz_idx_tensor(:));
                    EV_bounded_pre = beta_j .* permute(EV_d_sliced, [4, 1, 5, 2, 3]);

                    d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]);
                    z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1_dc * N_a1_other), [1, 1, 1, n_z_loc, 1]);
                    e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1_dc * N_a1_other * n_z_loc), [1, 1, 1, 1, n_e_loc]);
                    static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
                else; EV_bounded_pre = []; static_EV_offset = []; end

                LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar, d_gap, dc_mode_override) Evaluate_Case1_TensorBlock(...
                    state_idx, loweredge_matrix, maxgap_scalar, d_gap, N_a1_dc, N_a1_other, max(1, N_a2_local), N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, A1_grids_1d, a2_grids_1d, ...
                    vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, dc_mode_override);

                state_list = start_a_idx:end_a_idx; total_states = length(state_list);
                flat_choices = max(1, N_d_safe) * N_a1_dc * N_a1_other;
                max_states_per_chunk = max(1, floor(50000000 / (flat_choices * n_z_loc * n_e_loc)));
                v_concat = []; p_apr_concat = []; p_d_concat = []; p_l2idx_concat = []; p_l2flag_concat = [];
                for chunk_start = 1:max_states_per_chunk:total_states
                    chunk_end = min(total_states, chunk_start + max_states_per_chunk - 1);
                    state_chunk = state_list(chunk_start:chunk_end);
                    if vfoptions.gridinterplayer(1) == 1
                        [~, ~, ~, ~, ~, p_a1_per_a2] = LocalBlockFn(state_chunk, [], 0, 0, 2);
                        loweredge_chunk = reshape(p_a1_per_a2, [max(1, N_d_safe), N_a1_other, length(state_chunk), N_ze_local]);
                        [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c] = LocalBlockFn(state_chunk, loweredge_chunk, n2long - 1, 0, 0);
                    else
                        [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c] = LocalBlockFn(state_chunk, [], 0, 0, 0);
                    end
                    v_concat = [v_concat; v_c]; p_apr_concat = [p_apr_concat; p_apr_c]; p_d_concat = [p_d_concat; p_d_c];
                    if vfoptions.gridinterplayer(1) == 1; p_l2idx_concat = [p_l2idx_concat; p_l2idx_c]; p_l2flag_concat = [p_l2flag_concat; p_l2flag_c]; end
                end
                V_j_max(start_a_idx:end_a_idx, curr_ze)     = reshape(v_concat,     [length(state_list), N_ze_local]);
                Pol_apr_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_apr_concat, [length(state_list), N_ze_local]);
                Pol_d_max(start_a_idx:end_a_idx, curr_ze)   = reshape(p_d_concat,   [length(state_list), N_ze_local]);
                if vfoptions.gridinterplayer(1) == 1
                    Pol_L2idx_max(start_a_idx:end_a_idx, curr_ze)  = reshape(p_l2idx_concat,  [length(state_list), N_ze_local]);
                    Pol_L2flag_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_l2flag_concat, [length(state_list), N_ze_local]);
                end
            end
        end
    end

    V_j_max     = reshape(V_j_max,     [N_a, n_z_work, n_e_work]);
    Pol_apr_max = reshape(Pol_apr_max, [N_a, n_z_work, n_e_work]);
    Pol_d_max   = reshape(Pol_d_max,   [N_a, n_z_work, n_e_work]);
    if vfoptions.gridinterplayer(1) == 1
        Pol_L2idx_max  = reshape(Pol_L2idx_max,  [N_a, n_z_work, n_e_work]);
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_a, n_z_work, n_e_work]);
        if N_d > 0; PolicyKron(1, :, :, :, jj) = (Pol_apr_max - 1) * N_d + Pol_d_max; else; PolicyKron(1, :, :, :, jj) = Pol_apr_max; end
        PolicyKron(2, :, :, :, jj) = Pol_L2idx_max; PolicyKron(3, :, :, :, jj) = Pol_L2flag_max;
    else
        if N_d > 0; PolicyKron(:, :, :, jj) = (Pol_apr_max - 1) * N_d + Pol_d_max; else; PolicyKron(:, :, :, jj) = Pol_apr_max; end
    end
    V(:, :, :, jj) = V_j_max; V_next = V_j_max;
end

if N_z == 0; V = squeeze(V); end
if N_d == 0; n_daprime = n_a(1:l_a1); else; n_daprime = [n_d, n_a(1:l_a1)]; end
if vfoptions.gridinterplayer(1) ~= 1; PolicyKron = shiftdim(PolicyKron, -1); end

if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    varargout{1} = V; varargout{2} = PolicyKron; return
end

disp('Unpacking Policy tensor to System RAM...');
num_pol_vars = length(n_daprime); n_daprime_col = n_daprime(:); divisors = cumprod([1; n_daprime_col(1:end-1)]);
MAX_INT32 = 2147483647;

if vfoptions.gridinterplayer(1) == 1
    total_elements = (num_pol_vars + 2) * n_a_work * n_z_work * n_e_work * N_j;
    if total_elements < (MAX_INT32 * 0.9)
        BaseIndexKron = PolicyKron(1, :, :, :, :);
        P_base_gpu = mod(floor((BaseIndexKron - 1) ./ divisors), n_daprime_col) + 1;
        P_gpu = [P_base_gpu; PolicyKron(2:3, :, :, :, :)]; Policy_flat = gather(P_gpu);
    else
        disp('Using memory-safe iterative unpacking due to massive array size...');
        Policy_flat = zeros([num_pol_vars + 2, n_a_work, n_z_work, n_e_work, N_j], vfoptions.precision);
        for jj = 1:N_j
            PK_j = PolicyKron(1, :, :, :, jj); P_base = mod(floor((PK_j - 1) ./ divisors), n_daprime_col) + 1;
            Policy_flat(:, :, :, :, jj) = gather([P_base; PolicyKron(2:3, :, :, :, jj)]);
        end
    end
else
    total_elements = num_pol_vars * n_a_work * n_z_work * n_e_work * N_j;
    if total_elements < (MAX_INT32 * 0.9)
        P_gpu = mod(floor((PolicyKron - 1) ./ divisors), n_daprime_col) + 1; Policy_flat = gather(P_gpu);
    else
        disp('Using memory-safe iterative unpacking due to massive array size...');
        Policy_flat = zeros([num_pol_vars, n_a_work, n_z_work, n_e_work, N_j], vfoptions.precision);
        for jj = 1:N_j
            PK_j = PolicyKron(:, :, :, :, jj); P_j_gpu = mod(floor((PK_j - 1) ./ divisors), n_daprime_col) + 1;
            Policy_flat(:, :, :, :, jj) = gather(P_j_gpu);
        end
    end
end

V_cpu = gather(V); out_pol_vars = size(Policy_flat, 1);
out_n_a = n_a(n_a > 0); if isempty(out_n_a); out_n_a = 1; end
out_n_all_z = n_all_z(n_all_z > 0); if isempty(out_n_all_z); out_n_all_z = 1; end
state_shape = out_n_a;
if has_z || has_semiz; state_shape = [state_shape, out_n_all_z]; end
if has_e; state_shape = [state_shape, n_e_pass]; end
state_shape = [state_shape, N_j];
Policy = reshape(Policy_flat, [out_pol_vars, state_shape]);
V = reshape(V_cpu, state_shape);
varargout{1} = V; varargout{2} = Policy;
end

function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max, Pol_a1_per_a2] = Evaluate_Case1_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, d_gap, N_a1_dc, N_a1_other, N_a2, N_d_safe, N_ze_local, ...
    Z_cells_block, E_cells_block, D_cells_block, A1_mat, A2_mat, A1_grids_1d, a2_grids_1d, ...
    gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
    TensorReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j, ...
    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, is_dc_mode)

N_states = length(state_idx);

% Cast indices to GPU to lock all memory extraction on the device (Prevents PCIe thrashing)
state_idx = cast(state_idx, 'like', EV_local);
if ~isempty(loweredge_matrix)
    loweredge_matrix = cast(loweredge_matrix, 'like', EV_local);
end

l_a1 = length(A1_grids_1d);
l_a2 = sum(size(A2_mat)>1);

if N_a2 > 1
    [a1_sub, a2_sub] = ind2sub([N_a1_dc * N_a1_other, N_a2], state_idx);
else
    a1_sub = state_idx;
end

A1_cells = cell(1, l_a1);
for ia = 1:l_a1
    A1_cells{ia} = cast(reshape(A1_mat(a1_sub, ia), [1, 1, N_states, 1, 1]), 'like', EV_local);
end

if l_a2 > 0
    A2_cells = cell(1, l_a2);
    for ia = 1:l_a2
        A2_cells{ia} = cast(reshape(A2_mat(a2_sub, ia), [1, 1, N_states, 1, 1]), 'like', EV_local);
    end
else
    A2_cells = {};
end

if isempty(loweredge_matrix)
    if gridinterplayer(1) == 0 || is_dc_mode == 2
        % =================================================================
        % BRANCH 1A: COARSE EVALUATION
        % =================================================================
        grids_for_choices = A1_grids_1d;
        if l_a1 > 1; [mesh_out{1:l_a1}] = ndgrid(grids_for_choices{:}); else; mesh_out{1} = grids_for_choices{1}; end
        num_choices_total = numel(mesh_out{1});
        Apr_cells = cell(1, l_a1); for ia = 1:l_a1; Apr_cells{ia} = reshape(mesh_out{ia}(:), [1, num_choices_total, 1, 1, 1]); end

        if N_a2 > 1
            for i_a = 1:length(Apr_cells)
                Apr_cells{i_a} = cast(Apr_cells{i_a}, 'like', EV_local);
            end
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
            A2_prime = TensoraprimeFn(D_cells_block, A2_cells, Z_cells_block, E_cells_block, aprimeFnParamsCell);
            a2_grid_1d_vec = a2_grids_1d{1}; a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));
            idx = discretize(a2_prime_clipped, a2_grid_1d_vec); idx(isnan(idx)) = N_a2 - 1; idx = max(1, min(idx, N_a2 - 1));
            a2_left = reshape(a2_grid_1d_vec(idx), size(idx)); a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
            weight(a2_right == a2_left) = 0;
            weight(abs(weight) < 1e-12) = 0;
            weight(abs(weight - 1) < 1e-12) = 1;

            A1pr_idx = reshape(1:(N_a1_dc * N_a1_other), [1, (N_a1_dc * N_a1_other), 1, 1, 1]); ZE_idx = reshape(1:N_ze_local, [1, 1, 1, n_z_loc, n_e_loc]);
            N_a2_global = max(1, prod(cellfun(@length, a2_grids_1d)));
            idx_left  = A1pr_idx + (idx - 1) * (N_a1_dc * N_a1_other) + (ZE_idx - 1) * (N_a1_dc * N_a1_other * N_a2_global);
            idx_right = A1pr_idx + (idx) * (N_a1_dc * N_a1_other) + (ZE_idx - 1) * (N_a1_dc * N_a1_other * N_a2_global);
            max_idx = numel(EV_local);
            dsemiz_stride = size(EV_local, 1) * size(EV_local, 2);
            linear_idx_left  = min(max_idx, max(1, idx_left  + (dsemiz_idx_tensor - 1) * dsemiz_stride));
            linear_idx_right = min(max_idx, max(1, idx_right + (dsemiz_idx_tensor - 1) * dsemiz_stride));

            EV_bounded = EV_local(linear_idx_left) + weight .* (EV_local(linear_idx_right) - EV_local(linear_idx_left));
            weight_full = weight + zeros(1, num_choices_total, 1, n_z_loc, n_e_loc, 'like', weight);
            EV_bounded(weight_full == 0) = EV_local(linear_idx_left(weight_full == 0));
            EV_bounded(weight_full == 1) = EV_local(linear_idx_right(weight_full == 1));
            EV_bounded(isnan(EV_bounded)) = -Inf;
            EV_bounded = beta_j .* EV_bounded;
        else
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
            choice_idx_linear = reshape(1:num_choices_total, [1, num_choices_total, 1, 1, 1]);
            a_offset = (choice_idx_linear - 1) * N_d_safe;
            EV_bounded = EV_bounded_pre(static_EV_offset + a_offset);
        end
        FLAT_CHOICES = max(1, N_d_safe) * num_choices_total; FLAT_STATES  = N_states * N_ze_local;
        RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
        clear F_tensor EV_bounded; % Memory Hoist

        RHS_flat = reshape(RHS, [FLAT_CHOICES, FLAT_STATES]);
        clear RHS; % Memory Hoist

        [V_sub_coarse, Pol_sub_idx] = max(RHS_flat, [], 1);

        if nargout > 5
            num_choices_a1 = num_choices_total / N_a1_other;
            RHS_for_d = reshape(RHS_flat, [max(1, N_d_safe), num_choices_a1, N_a1_other, FLAT_STATES]);
            [~, max_a1_idx_per_d] = max(RHS_for_d, [], 2);
            clear RHS_for_d; % Memory Hoist
            Pol_a1_per_a2 = reshape(max_a1_idx_per_d, [max(1, N_d_safe), N_a1_other, N_states, N_ze_local]);
        else
            Pol_a1_per_a2 = [];
        end
        clear RHS_flat;

        d_idx_local   = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1; apr_idx_local = ceil(Pol_sub_idx / max(1, N_d_safe));
        V_j_max        = reshape(V_sub_coarse,  [N_states, N_ze_local]);
        Pol_apr_max    = reshape(apr_idx_local, [N_states, N_ze_local]);
        Pol_d_max      = reshape(d_idx_local,   [N_states, N_ze_local]);
        Pol_L2idx_max  = []; Pol_L2flag_max = [];

    else
        % =================================================================
        % BRANCH 1B: FULL FINE GRID EVALUATION (1-Step Brute Force)
        % =================================================================
        grids_for_choices = A1_grids_1d; grids_for_choices{1} = a1prime_grid;
        if l_a1 > 1; [mesh_out{1:l_a1}] = ndgrid(grids_for_choices{:}); else; mesh_out{1} = grids_for_choices{1}; end
        num_choices_total = numel(mesh_out{1});
        Apr_cells = cell(1, l_a1); for ia = 1:l_a1; Apr_cells{ia} = reshape(mesh_out{ia}(:), [1, num_choices_total, 1, 1, 1]); end

        if N_a2 > 1
            for i_a = 1:length(Apr_cells)
                Apr_cells{i_a} = cast(Apr_cells{i_a}, 'like', EV_local);
            end
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
            A2_prime = TensoraprimeFn(D_cells_block, A2_cells, Z_cells_block, E_cells_block, aprimeFnParamsCell);
            a2_grid_1d_vec = a2_grids_1d{1}; a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));
            idx = discretize(a2_prime_clipped, a2_grid_1d_vec); idx(isnan(idx)) = N_a2 - 1; idx = max(1, min(idx, N_a2 - 1));
            a2_left = reshape(a2_grid_1d_vec(idx), size(idx)); a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
            weight(a2_right == a2_left) = 0;
            weight(abs(weight) < 1e-12) = 0;
            weight(abs(weight - 1) < 1e-12) = 1;

            A1pr_idx = reshape(1:num_choices_total, [1, num_choices_total, 1, 1, 1]);
            ZE_offset = reshape((0:N_ze_local-1) * (length(a1prime_grid) * N_a1_other * N_a2), [1, 1, 1, n_z_loc, n_e_loc]);
            lin_idx_left = A1pr_idx + (idx - 1) * (length(a1prime_grid) * N_a1_other) + ZE_offset;
            lin_idx_right = A1pr_idx + (idx) * (length(a1prime_grid) * N_a1_other) + ZE_offset;
            if N_dsemiz > 1
                dsemiz_offset = (dsemiz_idx_tensor - 1) * (length(a1prime_grid) * N_a1_other * N_a2 * N_ze_local);
                lin_idx_left = lin_idx_left + dsemiz_offset; lin_idx_right = lin_idx_right + dsemiz_offset;
            end
            EV_left = EV_interp_local(lin_idx_left); EV_right = EV_interp_local(lin_idx_right);
            EV_bounded = EV_local(linear_idx_left) + weight .* (EV_local(linear_idx_right) - EV_local(linear_idx_left));
            weight_full = weight + zeros(1, num_choices_total, 1, n_z_loc, n_e_loc, 'like', weight);
            EV_bounded(weight_full == 0) = EV_local(linear_idx_left(weight_full == 0));
            EV_bounded(weight_full == 1) = EV_local(linear_idx_right(weight_full == 1));
            EV_bounded(isnan(EV_bounded)) = -Inf;
            EV_bounded = beta_j .* EV_bounded;
        else
            for i_a = 1:length(Apr_cells)
                Apr_cells{i_a} = cast(Apr_cells{i_a}, 'like', EV_local);
            end
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
            choice_idx_linear = reshape(1:num_choices_total, [1, num_choices_total, 1, 1, 1]);
            stride_z = length(a1prime_grid) * N_a1_other;
            ze_offset = reshape((0:N_ze_local-1) * stride_z, [1, 1, 1, n_z_loc, n_e_loc]);
            L2_linear_idx = choice_idx_linear + ze_offset;
            if N_dsemiz > 1; L2_linear_idx = L2_linear_idx + (dsemiz_idx_tensor - 1) * (stride_z * N_ze_local); end
            EV_bounded = beta_j .* EV_interp_local(L2_linear_idx);
        end
        FLAT_CHOICES = max(1, N_d_safe) * num_choices_total; FLAT_STATES  = N_states * N_ze_local;
        RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
        clear F_tensor EV_bounded; % Memory Hoist

        RHS_flat = reshape(RHS, [FLAT_CHOICES, FLAT_STATES]);
        clear RHS; % Memory Hoist

        [V_sub_fine, Pol_sub_idx] = max(RHS_flat, [], 1);
        Pol_a1_per_a2 = [];
        d_idx_local = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1; apr_offset  = ceil(Pol_sub_idx / max(1, N_d_safe));
        V_j_max   = reshape(V_sub_fine,  [N_states, N_ze_local]); Pol_d_max = reshape(d_idx_local, [N_states, N_ze_local]);

        a1_apr_offset = mod(apr_offset - 1, length(a1prime_grid)) + 1;
        a2_offset_factor = ceil(apr_offset / length(a1prime_grid));

        Pol_apr_max = floor((a1_apr_offset - 1) / (n2short + 1)) + 1; Pol_apr_max = min(Pol_apr_max, N_a1_dc - 1);
        Pol_L2idx_max = a1_apr_offset - (Pol_apr_max - 1) * (n2short + 1);
        Pol_apr_max = Pol_apr_max + (a2_offset_factor - 1) * N_a1_dc;

        Pol_apr_max    = reshape(Pol_apr_max, [N_states, N_ze_local]); Pol_L2idx_max  = reshape(Pol_L2idx_max, [N_states, N_ze_local]);
        Pol_L2flag_max = 2 * ones(1, FLAT_STATES, 'like', V_j_max);

        idx_lower_coarse = (a1_apr_offset(:)' - 1) * (n2short + 1) + 1;
        idx_upper_coarse = min(length(a1prime_grid), idx_lower_coarse + (n2short + 1));
        lin_lower = d_idx_local(:)' + (idx_lower_coarse - 1 + (a2_offset_factor(:)' - 1) * length(a1prime_grid)) * max(1, N_d_safe) + (0:FLAT_STATES-1) * size(RHS_flat, 1);
        lin_upper = d_idx_local(:)' + (idx_upper_coarse - 1 + (a2_offset_factor(:)' - 1) * length(a1prime_grid)) * max(1, N_d_safe) + (0:FLAT_STATES-1) * size(RHS_flat, 1);
        isInfLower = (RHS_flat(lin_lower) == -Inf); isInfUpper = (RHS_flat(lin_upper) == -Inf);
        clear RHS_flat; % Memory Hoist

        isInnerOrUpper = (Pol_L2idx_max(:)' > 1); isInnerOrLower = (Pol_L2idx_max(:)' < n2short + 2);
        Pol_L2flag_max(isInnerOrUpper & isInfLower) = 3; Pol_L2flag_max(isInnerOrLower & isInfUpper) = 1;
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_states, N_ze_local]);
    end

else
    % =================================================================
    % BRANCH 2: ZOOM PHASE (loweredge_matrix provided)
    % =================================================================
    Pol_a1_per_a2 = [];

    % Legacy bounds can be expanded < 1 or > N. Safely clip before extracting the a1 dimension
    % to prevent negative modulo math from wrapping the bounds to the opposite end of the grid.
    loweredge_matrix = max(1, min(loweredge_matrix, N_a1_dc * N_a1_other));
    loweredge_matrix = mod(loweredge_matrix - 1, N_a1_dc) + 1;

    % Robust Geometry Normalization
    % Legacy Slicers provide loweredge_matrix in various collapsed 1D/2D forms.
    % We map the elements to the core states and use implicit broadcasting to fill D and A_other.
    num_val = numel(loweredge_matrix);
    target_shape = zeros(max(1, N_d_safe), N_a1_other, N_states, n_z_loc, n_e_loc, 'like', loweredge_matrix);
    target_states_ze = N_states * n_z_loc * n_e_loc;

    if num_val == target_states_ze
        low_reshaped = reshape(loweredge_matrix, [1, 1, N_states, n_z_loc, n_e_loc]);
    elseif num_val == N_a1_other * target_states_ze
        low_reshaped = reshape(loweredge_matrix, [1, N_a1_other, N_states, n_z_loc, n_e_loc]);
    elseif num_val == max(1, N_d_safe) * target_states_ze
        low_reshaped = reshape(loweredge_matrix, [max(1, N_d_safe), 1, N_states, n_z_loc, n_e_loc]);
    elseif num_val == max(1, N_d_safe) * N_a1_other * target_states_ze
        low_reshaped = reshape(loweredge_matrix, [max(1, N_d_safe), N_a1_other, N_states, n_z_loc, n_e_loc]);
    else
        % Ultimate Fallback: Truncate or pad to exactly match the state space, averting a hard crash
        low_flat = loweredge_matrix(:);
        if length(low_flat) < target_states_ze
            low_flat = repmat(low_flat, ceil(target_states_ze / max(1, length(low_flat))), 1);
        end
        low_reshaped = reshape(low_flat(1:target_states_ze), [1, 1, N_states, n_z_loc, n_e_loc]);
    end
    loweredge_matrix = low_reshaped + target_shape;

    if gridinterplayer(1) == 0 || is_dc_mode == 2
        % -------------------------------------------------------------
        % SCENARIO 2A: Standard DC Segment Zoom
        % -------------------------------------------------------------
        total_gap = maxgap_scalar;
        if l_a1 == 1
            base_idx_a1 = reshape(loweredge_matrix, [max(1, N_d_safe), 1, N_states, n_z_loc, n_e_loc]);
            offsets_a1 = reshape(0:total_gap, [1, total_gap + 1, 1, 1, 1]);
            choice_idx_a1_base = max(1, min(base_idx_a1 + offsets_a1, length(A1_grids_1d{1})));
            Apr_cells = { A1_grids_1d{1}(choice_idx_a1_base) };
            choice_idx_linear = choice_idx_a1_base;
            num_choices_total = total_gap + 1;
        else
            num_choices_total = (total_gap + 1) * N_a1_other;
            base_idx_a1 = reshape(loweredge_matrix, [max(1, N_d_safe), N_a1_other, N_states, n_z_loc, n_e_loc]);
            offsets_a1 = reshape(0:total_gap, [1, 1, 1, 1, 1, total_gap + 1]);
            choice_idx_a1_matrix = max(1, min(base_idx_a1 + offsets_a1, length(A1_grids_1d{1})));
            choice_idx_a1_matrix = permute(choice_idx_a1_matrix, [1, 6, 2, 3, 4, 5]);
            choice_idx_a1 = reshape(choice_idx_a1_matrix, [max(1, N_d_safe), num_choices_total, N_states, n_z_loc, n_e_loc]);

            a2_base_vec = reshape(1:N_a1_other, [1, 1, N_a1_other]);
            a2_mesh = repmat(a2_base_vec, [max(1, N_d_safe), total_gap + 1, 1]);
            choice_idx_a2 = cast(reshape(a2_mesh, [max(1, N_d_safe), num_choices_total, 1, 1, 1]), 'like', choice_idx_a1);

            Apr_cells = cell(1, l_a1);
            Apr_cells{1} = A1_grids_1d{1}(choice_idx_a1);
            if l_a1 > 2; [mesh_a2{1:l_a1-1}] = ndgrid(A1_grids_1d{2:end}); else; mesh_a2{1} = A1_grids_1d{2}; end
            for ia = 2:l_a1; flat_grid = mesh_a2{ia-1}(:); Apr_cells{ia} = reshape(flat_grid(choice_idx_a2), [max(1, N_d_safe), num_choices_total, 1, 1, 1]); end
            choice_idx_linear = choice_idx_a1 + (choice_idx_a2 - 1) * length(A1_grids_1d{1});
        end

        if N_a2 > 1
            for i_a = 1:length(Apr_cells)
                Apr_cells{i_a} = cast(Apr_cells{i_a}, 'like', EV_local);
            end
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
            A2_prime = TensoraprimeFn(D_cells_block, A2_cells, Z_cells_block, E_cells_block, aprimeFnParamsCell);

            a2_grid_1d_vec = a2_grids_1d{1};
            a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));
            idx = discretize(a2_prime_clipped, a2_grid_1d_vec);
            idx(isnan(idx)) = N_a2 - 1;
            idx = max(1, min(idx, N_a2 - 1));
            a2_left = reshape(a2_grid_1d_vec(idx), size(idx));
            a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
            weight(a2_right == a2_left) = 0;
            weight(abs(weight) < 1e-12) = 0;
            weight(abs(weight - 1) < 1e-12) = 1;

            ZE_idx = reshape(1:N_ze_local, [1, 1, 1, n_z_loc, n_e_loc]);
            N_a2_global = max(1, prod(cellfun(@length, a2_grids_1d)));
            idx_left  = choice_idx_linear + (idx - 1) * (N_a1_dc * N_a1_other) + (ZE_idx - 1) * (N_a1_dc * N_a1_other * N_a2_global);
            idx_right = choice_idx_linear + (idx) * (N_a1_dc * N_a1_other) + (ZE_idx - 1) * (N_a1_dc * N_a1_other * N_a2_global);

            max_idx = numel(EV_local);
            dsemiz_stride = size(EV_local, 1) * size(EV_local, 2);
            linear_idx_left  = min(max_idx, max(1, idx_left  + (dsemiz_idx_tensor - 1) * dsemiz_stride));
            linear_idx_right = min(max_idx, max(1, idx_right + (dsemiz_idx_tensor - 1) * dsemiz_stride));

            EV_bounded = EV_local(linear_idx_left) + weight .* (EV_local(linear_idx_right) - EV_local(linear_idx_left));
            weight_full = weight + zeros(1, num_choices_total, 1, n_z_loc, n_e_loc, 'like', weight);
            EV_bounded(weight_full == 0) = EV_local(linear_idx_left(weight_full == 0));
            EV_bounded(weight_full == 1) = EV_local(linear_idx_right(weight_full == 1));
            EV_bounded(isnan(EV_bounded)) = -Inf;
            EV_bounded = beta_j .* EV_bounded;
        else
            for i_a = 1:length(Apr_cells)
                Apr_cells{i_a} = cast(Apr_cells{i_a}, 'like', EV_local);
            end
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
            EV_bounded = EV_bounded_pre(static_EV_offset + (choice_idx_linear - 1) * N_d_safe);
        end

    else
        % -------------------------------------------------------------
        % SCENARIO 2B: Grid Interpolation Zoom
        % -------------------------------------------------------------
        loweredge_matrix_bounds = max(2, min(loweredge_matrix, length(A1_grids_1d{1}) - 1));
        L2_base = (loweredge_matrix_bounds - 1) * (n2short + 1) + 1;
        start_offset = -(n2short + 1);
        end_offset = (n2short + 1);
        num_choices_total_a1 = end_offset - start_offset + 1;

        if l_a1 == 1
            base_idx_a1 = reshape(L2_base, [max(1, N_d_safe), 1, N_states, n_z_loc, n_e_loc]);
            offsets_a1 = reshape(start_offset:end_offset, [1, num_choices_total_a1, 1, 1, 1]);
            raw_choice_idx_a1 = base_idx_a1 + offsets_a1;
            out_of_bounds = (raw_choice_idx_a1 < 1) | (raw_choice_idx_a1 > length(a1prime_grid));
            choice_idx_a1_base = max(1, min(raw_choice_idx_a1, length(a1prime_grid)));
            Apr_cells = { reshape(a1prime_grid(choice_idx_a1_base), size(choice_idx_a1_base)) };
            choice_idx_linear = choice_idx_a1_base;
            num_choices_total = num_choices_total_a1;
        else
            num_choices_total = num_choices_total_a1 * N_a1_other;
            base_idx_a1 = reshape(L2_base, [max(1, N_d_safe), N_a1_other, N_states, n_z_loc, n_e_loc]);
            offsets_a1 = reshape(start_offset:end_offset, [1, 1, 1, 1, 1, num_choices_total_a1]);
            raw_choice_idx_a1_matrix = base_idx_a1 + offsets_a1;
            raw_choice_idx_a1_matrix = permute(raw_choice_idx_a1_matrix, [1, 6, 2, 3, 4, 5]);
            raw_choice_idx_a1 = reshape(raw_choice_idx_a1_matrix, [max(1, N_d_safe), num_choices_total, N_states, n_z_loc, n_e_loc]);

            out_of_bounds = (raw_choice_idx_a1 < 1) | (raw_choice_idx_a1 > length(a1prime_grid));
            choice_idx_a1 = max(1, min(raw_choice_idx_a1, length(a1prime_grid)));

            a2_base_vec = reshape(1:N_a1_other, [1, 1, N_a1_other]);
            a2_mesh = repmat(a2_base_vec, [max(1, N_d_safe), num_choices_total_a1, 1]);
            choice_idx_a2 = cast(reshape(a2_mesh, [max(1, N_d_safe), num_choices_total, 1, 1, 1]), 'like', choice_idx_a1);

            Apr_cells = cell(1, l_a1);
            Apr_cells{1} = reshape(a1prime_grid(choice_idx_a1), size(choice_idx_a1));
            if l_a1 > 2; [mesh_a2{1:l_a1-1}] = ndgrid(A1_grids_1d{2:end}); else; mesh_a2{1} = A1_grids_1d{2}; end
            for ia = 2:l_a1; flat_grid = mesh_a2{ia-1}(:); Apr_cells{ia} = reshape(flat_grid(choice_idx_a2), [max(1, N_d_safe), num_choices_total, 1, 1, 1]); end
            choice_idx_linear = choice_idx_a1 + (choice_idx_a2 - 1) * length(a1prime_grid);
        end

        % Update the mask format for Branch 2B since it now inherently has N_d_safe

        if N_a2 > 1
            for i_a = 1:length(Apr_cells)
                Apr_cells{i_a} = cast(Apr_cells{i_a}, 'like', EV_local);
            end
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
            A2_prime = TensoraprimeFn(D_cells_block, A2_cells, Z_cells_block, E_cells_block, aprimeFnParamsCell);

            a2_grid_1d_vec = a2_grids_1d{1};
            a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));
            idx = discretize(a2_prime_clipped, a2_grid_1d_vec);
            idx(isnan(idx)) = N_a2 - 1;
            idx = max(1, min(idx, N_a2 - 1));
            a2_left = reshape(a2_grid_1d_vec(idx), size(idx));
            a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
            weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
            weight(a2_right == a2_left) = 0;
            weight(abs(weight) < 1e-12) = 0;
            weight(abs(weight - 1) < 1e-12) = 1;

            ZE_offset = reshape((0:N_ze_local-1) * (length(a1prime_grid) * N_a1_other * N_a2), [1, 1, 1, n_z_loc, n_e_loc]);
            lin_idx_left = choice_idx_linear + (idx - 1) * (length(a1prime_grid) * N_a1_other) + ZE_offset;
            lin_idx_right = choice_idx_linear + (idx) * (length(a1prime_grid) * N_a1_other) + ZE_offset;

            if N_dsemiz > 1
                dsemiz_offset = (dsemiz_idx_tensor - 1) * (length(a1prime_grid) * N_a1_other * N_a2 * N_ze_local);
                lin_idx_left = lin_idx_left + dsemiz_offset;
                lin_idx_right = lin_idx_right + dsemiz_offset;
            end

            EV_left = EV_interp_local(lin_idx_left);
            EV_right = EV_interp_local(lin_idx_right);
            EV_bounded = EV_left + weight .* (EV_right - EV_left);
            clear EV_left EV_right; % Memory Hoist

            weight_full = weight + zeros(1, num_choices_total, 1, n_z_loc, n_e_loc, 'like', weight);
            EV_bounded(weight_full == 0) = EV_interp_local(lin_idx_left(weight_full == 0));
            EV_bounded(weight_full == 1) = EV_interp_local(lin_idx_right(weight_full == 1));
            EV_bounded(isnan(EV_bounded)) = -Inf;
            EV_bounded = beta_j .* EV_bounded;
        else
            for i_a = 1:length(Apr_cells)
                Apr_cells{i_a} = cast(Apr_cells{i_a}, 'like', EV_local);
            end
            F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
            stride_z = length(a1prime_grid) * N_a1_other;
            ze_offset = reshape((0:N_ze_local-1) * stride_z, [1, 1, 1, n_z_loc, n_e_loc]);
            L2_linear_idx = choice_idx_linear + ze_offset;
            if N_dsemiz > 1; L2_linear_idx = L2_linear_idx + (dsemiz_idx_tensor - 1) * (stride_z * N_ze_local); end
            EV_bounded = EV_interp_local(L2_linear_idx);
            EV_bounded(out_of_bounds) = -Inf;
            EV_bounded = beta_j .* EV_bounded;
        end
    end

    % --- Universal RHS Evaluation ---
    FLAT_STATES = N_states * N_ze_local;
    RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    clear F_tensor EV_bounded; % Memory Hoist

    RHS_flat = reshape(RHS, [max(1, N_d_safe) * num_choices_total, FLAT_STATES]);
    clear RHS; % Memory Hoist
    [V_sub_fine, Pol_sub_idx] = max(RHS_flat, [], 1);

    if nargout > 5
        num_choices_a1 = num_choices_total / N_a1_other;
        RHS_for_d = reshape(RHS_flat, [max(1, N_d_safe), num_choices_a1, N_a1_other, FLAT_STATES]);
        [~, max_a1_idx_rel] = max(RHS_for_d, [], 2);
        clear RHS_for_d; % Memory Hoist
        if gridinterplayer(1) == 0 || is_dc_mode == 2
            max_a1_idx_rel = reshape(max_a1_idx_rel, [max(1, N_d_safe), N_a1_other, N_states, N_ze_local]);
            Pol_a1_per_a2 = min(loweredge_matrix + max_a1_idx_rel - 1, N_a1_dc);
        else
            Pol_a1_per_a2 = [];
        end
    end

    d_idx_local = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1;
    apr_offset  = ceil(Pol_sub_idx / max(1, N_d_safe));

    V_j_max   = reshape(V_sub_fine,  [N_states, N_ze_local]);
    Pol_d_max = reshape(d_idx_local, [N_states, N_ze_local]);

    if gridinterplayer(1) == 0 || is_dc_mode == 2
        clear RHS_flat; % Memory Hoist
        a1_apr_offset = mod(apr_offset(:)' - 1, total_gap + 1) + 1;
        a2_offset_factor = ceil(apr_offset(:)' / (total_gap + 1));

        loweredge_matrix_2d = reshape(loweredge_matrix, [max(1, N_d_safe), N_a1_other, FLAT_STATES]);
        lin_idx_loweredge = d_idx_local(:)' + (a2_offset_factor(:)' - 1) * max(1, N_d_safe) + (0:FLAT_STATES-1) * (max(1, N_d_safe) * N_a1_other);
        chosen_loweredge = loweredge_matrix_2d(lin_idx_loweredge);

        a1_Pol_apr = chosen_loweredge(:)' + a1_apr_offset(:)' - 1;
        a1_Pol_apr = min(a1_Pol_apr, N_a1_dc); % Clip to ensure out-of-bound evaluations aren't returned as policy
        Pol_apr_max = a1_Pol_apr(:)' + (a2_offset_factor(:)' - 1) * N_a1_dc;
        Pol_apr_max = reshape(Pol_apr_max, [N_states, N_ze_local]);
        Pol_L2idx_max = [];
        Pol_L2flag_max = [];
    else
        a1_apr_offset = mod(apr_offset(:)' - 1, num_choices_total_a1) + 1;
        a2_offset_factor = ceil(apr_offset(:)' / num_choices_total_a1);
        chosen_offset = start_offset + a1_apr_offset(:)' - 1;

        loweredge_matrix_2d = reshape(loweredge_matrix_bounds, [max(1, N_d_safe), N_a1_other, FLAT_STATES]);
        lin_idx_loweredge = d_idx_local(:)' + (a2_offset_factor(:)' - 1) * max(1, N_d_safe) + (0:FLAT_STATES-1) * (max(1, N_d_safe) * N_a1_other);
        chosen_loweredge = loweredge_matrix_2d(lin_idx_loweredge);

        abs_fine_idx_flat = (chosen_loweredge(:)' - 1) * (n2short + 1) + 1 + chosen_offset(:)';
        a1_Pol_apr = floor((abs_fine_idx_flat(:)' - 1) / (n2short + 1)) + 1;
        a1_Pol_apr = min(a1_Pol_apr, N_a1_dc - 1);
        Pol_L2idx_max = abs_fine_idx_flat(:)' - (a1_Pol_apr(:)' - 1) * (n2short + 1);
        Pol_apr_max = a1_Pol_apr(:)' + (a2_offset_factor(:)' - 1) * N_a1_dc;
        Pol_apr_max = reshape(Pol_apr_max, [N_states, N_ze_local]);
        Pol_L2idx_max = reshape(Pol_L2idx_max, [N_states, N_ze_local]);

        lin_lower = d_idx_local(:)' + (1 - 1) * max(1, N_d_safe) + (a2_offset_factor(:)' - 1) * (num_choices_total_a1 * max(1, N_d_safe)) + (0:FLAT_STATES-1) * size(RHS_flat, 1);
        lin_upper = d_idx_local(:)' + (num_choices_total_a1 - 1) * max(1, N_d_safe) + (a2_offset_factor(:)' - 1) * (num_choices_total_a1 * max(1, N_d_safe)) + (0:FLAT_STATES-1) * size(RHS_flat, 1);

        isInfLower = (RHS_flat(lin_lower) == -Inf);
        isInfUpper = (RHS_flat(lin_upper) == -Inf);
        clear RHS_flat; % Memory Hoist

        inLowerStrict = (a1_apr_offset(:)' >= 2) & (a1_apr_offset(:)' <= n2short + 1);
        inUpperStrict = (a1_apr_offset(:)' >= n2short + 3 + d_gap * (n2short + 1)) & (a1_apr_offset(:)' <= num_choices_total_a1 - 1);

        Pol_L2flag_max = 2 * ones(1, FLAT_STATES, 'like', V_j_max);
        Pol_L2flag_max(inLowerStrict & isInfLower) = 3;
        Pol_L2flag_max(inUpperStrict & isInfUpper) = 1;
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_states, N_ze_local]);
    end
end


end


function [v, p_apr, p_d, p_l2, p_l2f, p_a1] = Helper_SlicerWrapper(a1_idx, low_mat, mg, dc_mode, N_a1_dc, N_a1_other, N_a2, N_ze, N_d, CoreFn)
% Translates 1D index chunks from DC Slicers into absolute 1D states for the Tensor Block,
% and reshapes the 2D tensor outputs back into the multi-dimensional geometry expected by the Slicers.

N_other = N_a1_other * max(1, N_a2);
state_chunk = reshape(a1_idx(:) + (0:N_other-1) * N_a1_dc, 1, []);
num_a1 = length(a1_idx);

if isempty(low_mat)
    low_chunk = [];
    d_gap = 0;
else
    % Legacy DC slicer passes loweredge_matrix with N_d as the LAST dimension
    % We must permute it to [N_d, num_a1, N_other, N_ze] before handing it to the Tensor Block
    if numel(low_mat) == num_a1 * N_other * N_ze * max(1, N_d)
        low_chunk = reshape(low_mat, [num_a1, N_other, N_ze, max(1, N_d)]);
        low_chunk = permute(low_chunk, [4, 1, 2, 3]);
    else
        low_chunk = low_mat;
    end
    d_gap = 0;
end

% Strictly respect the caller's requested outputs to prevent generating unused geometry
if nargout > 5
    [v_c, p_apr_c, p_d_c, p_l2_c, p_l2f_c, p_a1_c] = CoreFn(state_chunk, low_chunk, mg, d_gap, dc_mode);
else
    [v_c, p_apr_c, p_d_c, p_l2_c, p_l2f_c] = CoreFn(state_chunk, low_chunk, mg, d_gap, dc_mode);
    p_a1_c = [];
end

out_shape = [num_a1, N_other, N_ze];

v     = reshape(v_c, out_shape);
p_apr = reshape(p_apr_c, out_shape);
p_d   = reshape(p_d_c, out_shape);

if ~isempty(p_l2_c)
    p_l2  = reshape(p_l2_c, out_shape);
else
    p_l2  = [];
end

if ~isempty(p_l2f_c)
    p_l2f = reshape(p_l2f_c, out_shape);
else
    p_l2f = [];
end

if ~isempty(p_a1_c)
    % Tensor returns p_a1_c with N_d FIRST: [N_d, N_a1_other, N_states, N_ze]
    % We must permute it back to [num_a1, ..., N_d] for the legacy Slicers
    if N_a1_other > 1
        p_a1 = reshape(p_a1_c, [max(1, N_d), N_a1_other, num_a1, max(1, N_a2), N_ze]);
        p_a1 = permute(p_a1, [3, 2, 4, 5, 1]); % -> [num_a1, N_a1_other, N_a2, N_ze, N_d]
    else
        p_a1 = reshape(p_a1_c, [max(1, N_d), num_a1, N_other, N_ze]);
        p_a1 = permute(p_a1, [2, 3, 4, 1]); % -> [num_a1, N_other, N_ze, N_d]
    end
else
    p_a1 = [];
end


end
