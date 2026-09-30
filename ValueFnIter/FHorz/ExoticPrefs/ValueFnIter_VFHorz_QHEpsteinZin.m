function [V, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% =========================================================================
% PHASE 1: PRE-COMPUTATION & SETUP
% =========================================================================
% 1.1 Preference & Parameter Initialization
proto = a_grid;
vfoptions.precision = underlyingType(proto);

if strcmp(vfoptions.exoticpreferences, 'QHEpsteinZin')
    vfoptions = EpsteinZinSetup_VFHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions);
    ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4;
    ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; ezc8 = vfoptions.ezc8;
else
    % Neutral CRRA Fallbacks
    ezc2 = ones(N_j, 1, 'like', proto); ezc3 = 1; ezc4 = 1;
    ezc5 = ones(N_j, 1, 'like', proto); ezc6 = ones(N_j, 1, 'like', proto);
    ezc7 = ones(N_j, 1, 'like', proto); ezc8 = ones(N_j, 1, 'like', proto);
end

if isfield(vfoptions, 'ezc9'); ezc9 = vfoptions.ezc9; else; ezc9 = ones(N_j, 1, 'like', proto); end

% Extract present-bias parameter beta0 universally
if isfield(vfoptions, 'QHadditionaldiscount') && isfield(Parameters, vfoptions.QHadditionaldiscount)
    beta0_val = Parameters.(vfoptions.QHadditionaldiscount);
    if isscalar(beta0_val); beta0_j = beta0_val * ones(N_j, 1, 'like', proto); else; beta0_j = beta0_val; end
else
    beta0_j = ones(N_j, 1, 'like', proto);
end

% 1.2 Dual-Pass (Naive vs Sophisticated) Dispatcher
if isfield(vfoptions, 'quasi_hyperbolic') && strcmpi(vfoptions.quasi_hyperbolic, 'Sophisticated')
    is_naive = false;
else
    is_naive = true; % VFIToolkit default
end

% 1.3 Universal Grid Packing & Sizing
l_a2 = 0;
if vfoptions.experienceasset > 0; l_a2 = vfoptions.experienceasset; end
if vfoptions.experienceassetz > 0; l_a2 = vfoptions.experienceassetz; end
if l_a2 > 0; n_a1 = n_a(1:end-l_a2); n_a2 = n_a(end-l_a2+1:end); else; n_a1 = n_a; n_a2 = []; end

N_a1 = prod(max(1, n_a1));
N_a2 = prod(max(1, n_a2));
N_a = prod(max(1, n_a));
N_d = prod(n_d);
N_d_safe = prod(max(1, n_d));
N_z = prod(n_z);
N_z_safe = prod(max(1, n_z));

a1_grid_len = sum(n_a1);
a1_grid_vals = a_grid(1:a1_grid_len);
a2_grid_vals = a_grid(a1_grid_len+1:end);

has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
if has_e; n_e_pass = vfoptions.n_e; e_work = vfoptions.e_grid; else; n_e_pass = 0; e_work = ones(1, 1, 'like', proto); end

[TensorReturnFn, D_cells_block, A1_cells, ~, ~] = CreateTensorFnAndCells(ReturnFn, n_d, n_a1, n_z, n_e_pass, d_grid, a1_grid_vals, [], []);
if l_a2 > 0; [TensoraprimeFn, ~, A2_cells, ~, ~] = CreateTensorFnAndCells(vfoptions.aprimeFn, 0, n_a2, 0, 0, [], a2_grid_vals, [], []); else; TensoraprimeFn = []; A2_cells = {}; end

A1_mat = zeros(N_a1, length(n_a1), 'like', proto);
for i_a = 1:length(n_a1); A1_mat(:, i_a) = A1_cells{i_a}(:); end

A2_mat = zeros(N_a2, length(n_a2), 'like', proto);
a2_grids_1d = cell(1, length(n_a2)); offset = 0;
for i_a = 1:length(n_a2); A2_mat(:, i_a) = A2_cells{i_a}(:); a2_grids_1d{i_a} = a2_grid_vals((offset + 1):(offset + n_a2(i_a))); offset = offset + n_a2(i_a); end

for i_d = 1:length(D_cells_block); D_cells_block{i_d} = reshape(D_cells_block{i_d}, [N_d_safe, 1, 1, 1, 1]); end

aprimeFnParamsCell = {};
if l_a2 > 0; aprimeFnParamNames = vfoptions.aprimeFnParamNames(isfield(Parameters, vfoptions.aprimeFnParamNames)); end

has_semiz = isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0;
if has_semiz; N_semiz = prod(vfoptions.n_semiz); N_z_exog = max(1, prod(n_z) / N_semiz); else; N_semiz = 1; N_z_exog = max(1, prod(n_z)); end
n_z_work = N_semiz * N_z_exog;
n_e_work = max(1, prod(n_e_pass));
N_ze = n_z_work * n_e_work;

V = zeros(N_a, n_z_work, n_e_work, N_j, 'like', proto);
Valt = zeros(N_a, n_z_work, n_e_work, N_j, 'like', proto);
Valt_next = zeros(N_a, n_z_work, n_e_work, 'like', proto);
if is_naive; V_exp_next = zeros(N_a, n_z_work, n_e_work, 'like', proto); end
Policyalt = []; % Tracker for alternative beliefs if required later

% 1.4 Interpolation & Memory Chunking Setup
if vfoptions.gridinterplayer(1) == 1
    PolicyKron = zeros(3, N_a, n_z_work, n_e_work, N_j, 'like', proto);
    n2short = vfoptions.ngridinterp; n2long = n2short * 2 + 3;
    a1_work = A1_cells{1}(:);
    a1prime_grid = interp1(1:1:N_a1, a1_work, linspace(1, N_a1, N_a1 + (N_a1 - 1) * n2short))';
    idx = discretize(a1prime_grid, a1_work);
    idx(isnan(idx) | idx == length(a1_work)) = length(a1_work) - 1;
    interp_left_idx = idx(:); interp_right_idx = idx(:) + 1;
    a1_left = a1_work(interp_left_idx); a1_right = a1_work(interp_right_idx);
    interp_weights = (a1prime_grid(:) - a1_left) ./ (a1_right - a1_left);
    interp_weights(a1_right == a1_left) = 0;
    if vfoptions.parallel == 2
        interp_left_idx = gpuArray(interp_left_idx); interp_right_idx = gpuArray(interp_right_idx); interp_weights = gpuArray(interp_weights);
    end
else
    PolicyKron = zeros(N_a, n_z_work, n_e_work, N_j, 'like', proto);
    n2short = 0; n2long = 0; a1prime_grid = []; interp_left_idx = []; interp_right_idx = []; interp_weights = [];
end

ze_chunks = {1:N_ze};
if ismember(vfoptions.lowmemory, [4, 5]) && l_a2 > 0; a2_chunks = num2cell(1:N_a2); else; a2_chunks = {1:N_a2}; end

chunk_meta = cell(1, length(ze_chunks));
for i_ze = 1:length(ze_chunks)
    c_ze = ze_chunks{i_ze}; c_ze_cpu = gather(c_ze);
    [z_ind, e_ind] = ind2sub([n_z_work, n_e_work], c_ze_cpu);
    meta.z_vals = unique(z_ind); meta.e_vals = unique(e_ind);
    meta.n_z_loc = length(meta.z_vals); meta.n_e_loc = length(meta.e_vals); meta.N_ze_local = length(c_ze);
    chunk_meta{i_ze} = meta;
end

% Keep STATIC scalars on the CPU so they bake into PTX as ultra-fast constants. Only push arrays to the GPU.
base_ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);
is_age_dependent = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j; is_age_dependent(ip) = true; end
    if vfoptions.parallel == 2 && isnumeric(base_ReturnFnParamsCell{ip}) && ~isa(base_ReturnFnParamsCell{ip}, 'gpuArray')
        if ~isscalar(base_ReturnFnParamsCell{ip}); base_ReturnFnParamsCell{ip} = gpuArray(base_ReturnFnParamsCell{ip}); end
    end
end

if l_a2 > 0
    base_aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, 1, vfoptions.precision);
    aprimeFnParam_is_age_dependent = false(1, length(aprimeFnParamNames));
    for ip = 1:length(aprimeFnParamNames)
        if numel(Parameters.(aprimeFnParamNames{ip})) == N_j; aprimeFnParam_is_age_dependent(ip) = true; end
        if vfoptions.parallel == 2 && isnumeric(base_aprimeFnParamsCell{ip}) && ~isa(base_aprimeFnParamsCell{ip}, 'gpuArray')
            if ~isscalar(base_aprimeFnParamsCell{ip}); base_aprimeFnParamsCell{ip} = gpuArray(base_aprimeFnParamsCell{ip}); end
        end
    end
else
    base_aprimeFnParamsCell = {}; aprimeFnParam_is_age_dependent = [];
end

is_EZ = strcmp(vfoptions.exoticpreferences, 'EpsteinZin') || strcmp(vfoptions.exoticpreferences, 'QHEpsteinZin');

% --- STRICT EZ GATEKEEPER ---
if is_EZ && isfield(vfoptions,'survivalprobability'); sj = Parameters.(vfoptions.survivalprobability);
elseif isfield(vfoptions,'WarmGlowBequestsFn'); sj = ones(N_j, 1); sj(end) = 0;
else; sj = ones(N_j, 1); end

warmglow = isfield(vfoptions,'WarmGlowBequestsFn');
N_dsemiz = 1; if has_semiz && N_d > 0; N_dsemiz = n_d(end); end

% =========================================================================
% PHASE 2: REVERSE TIME LOOP (j = N_j down to 1)
% =========================================================================
% --- Dynamic VRAM Profiling (Hoisted) ---
if vfoptions.parallel == 2
    gpu_device_info = gpuDevice(); safe_elements = max(1e7, floor((gpu_device_info.AvailableMemory / 8) / 8));
else
    safe_elements = 50000000; % CPU fallback
end

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose == 1; fprintf('Finite horizon QHEZ: %i of %i \n', jj, N_j); end

    % 2.1 Age-Dependent Parameter Updates
    % Push DYNAMIC scalars to GPU to preserve JIT cache recompilation limits
    ReturnFnParamsCell = base_ReturnFnParamsCell;
    for ip = find(is_age_dependent)
        val = Parameters.(ReturnFnParamNames{ip})(jj);
        if vfoptions.parallel == 2; ReturnFnParamsCell{ip} = gpuArray(val); else; ReturnFnParamsCell{ip} = val; end
    end

    if l_a2 > 0
        aprimeFnParamsCell = base_aprimeFnParamsCell;
        for ip = find(aprimeFnParam_is_age_dependent)
            val = Parameters.(aprimeFnParamNames{ip})(jj);
            if vfoptions.parallel == 2; aprimeFnParamsCell{ip} = gpuArray(val); else; aprimeFnParamsCell{ip} = val; end
        end
    else
        aprimeFnParamsCell = {};
    end

    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision);
    delta_j = prod(DiscountFactorParamsVec);
    sj_val = sj(jj);

    % --- EZ Weight Extraction ---
    if isfield(vfoptions, 'EZoneminusbeta') && vfoptions.EZoneminusbeta == 1
        ezc1_j = 1 - delta_j;
    elseif isfield(vfoptions, 'EZoneminusbeta') && vfoptions.EZoneminusbeta == 2
        ezc1_j = 1 - sj_val * delta_j;
    else
        ezc1_j = 1;
    end
    ezc9_j = ezc9(min(jj, length(ezc9)));

    pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    if has_e; pi_e_j = vfoptions.pi_e_J(:, min(jj + 1, size(vfoptions.pi_e_J, 2))); end
    if vfoptions.parallel == 2 && has_e && ~isa(pi_e_j, 'gpuArray'); pi_e_j = gpuArray(pi_e_j); end

    % 2.2 Terminal Conditions
    if jj == N_j && (~isfield(vfoptions, 'V_Jplus1') || isempty(vfoptions.V_Jplus1))
        Valt_next(:) = 0; if is_naive; V_exp_next(:) = 0; end
    elseif jj == N_j
        Valt_next = reshape(vfoptions.V_Jplus1, [N_a, n_z_work, n_e_work]);
        if vfoptions.parallel == 2 && ~isa(Valt_next, 'gpuArray'); Valt_next = gpuArray(Valt_next); end
        if is_naive; V_exp_next = Valt_next; end % Terminal beliefs align
    end

    % =========================================================================
    % PHASE 3: THE EXPECTATION BLOCK (Dual-Pass Vectorized)
    % =========================================================================
    num_EVs = 1 + is_naive;
    EV_pack = cell(1, num_EVs);
    for i_ev = 1:num_EVs
        if i_ev == 1; V_target = Valt_next; else; V_target = V_exp_next; end

        valid_V = isfinite(V_target) & (V_target ~= 0);
        V_transformed = V_target;
        if ezc5(jj) == 1; V_transformed(valid_V) = ezc4 * V_target(valid_V); else; V_transformed(valid_V) = max(ezc4 * V_target(valid_V), 0).^ezc5(jj); end
        V_transformed(V_target == 0) = 0;

        if has_e
            V_trans_flat = reshape(V_transformed, [N_a * n_z_work, n_e_work]);
            V_inf_mask = (V_trans_flat == -Inf);
            V_safe = V_trans_flat; V_safe(V_inf_mask) = -1e250;
            V_expected_e = V_safe * pi_e_j(:);
            inf_restore = (V_inf_mask * (pi_e_j(:) > 0)) > 0;
            V_expected_e(inf_restore) = -Inf;
            V_transformed = repmat(reshape(V_expected_e, [N_a, n_z_work, 1]), [1, 1, n_e_work]);
        end

        EV_base = zeros(N_a, N_semiz * N_z_exog, n_e_work, N_dsemiz, 'like', Valt_next);

        for ie = 1:n_e_work
            V_curr = V_transformed(:,:,ie);
            if N_z_exog > 1 && prod(n_z) > 0
                V_slice = reshape(V_curr, [N_a * N_semiz, N_z_exog]);
                V_inf_mask = (V_slice == -Inf);
                V_safe = V_slice; V_safe(V_inf_mask) = -1e250;
                V_z_eval = V_safe * pi_z_j';
                inf_restore = (V_inf_mask * (pi_z_j' > 0)) > 0;
                V_z_eval(inf_restore) = -Inf;
                V_z_eval = reshape(V_z_eval, [N_a, N_semiz, N_z_exog]);
            else; V_z_eval = reshape(V_curr, [N_a, N_semiz, N_z_exog]); end

            if has_semiz
                pi_semiz_j = vfoptions.pi_semiz_J(:, :, :, min(jj, size(vfoptions.pi_semiz_J, 4)));
                V_perm = reshape(permute(V_z_eval, [2, 1, 3]), [N_semiz, N_a * N_z_exog]);
                V_inf_mask = (V_perm == -Inf); V_safe = V_perm; V_safe(V_inf_mask) = -1e250;
                for idsemiz = 1:N_dsemiz
                    pi_semiz_d = pi_semiz_j(:, :, idsemiz);
                    EV_perm = pi_semiz_d * V_safe;
                    inf_restore = (pi_semiz_d > 0) * V_inf_mask > 0; EV_perm(inf_restore) = -Inf;
                    EV_d = permute(reshape(EV_perm, [N_semiz, N_a, N_z_exog]), [2, 1, 3]);
                    EV_base(:,:,ie,idsemiz) = reshape(EV_d, [N_a, N_semiz * N_z_exog]);
                end
            else; EV_base(:,:,ie,1) = reshape(V_z_eval, [N_a, N_semiz * N_z_exog]); end
        end

        if warmglow == 1
            wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            WG_eval = vfoptions.WarmGlowBequestsFn(a_grid, wg_params{:});
            if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a_grid), 'like', proto); end

            valid_wg = isfinite(WG_eval) & (WG_eval ~= 0);
            WG_transformed = WG_eval;
            if ezc5(jj) == 1; WG_transformed(valid_wg) = ezc4 * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4 * WG_eval(valid_wg), 0).^ezc5(jj); end
            WG_transformed(WG_eval == 0) = 0;
            EV_base = EV_base * sj_val + (1 - sj_val) * reshape(WG_transformed, [N_a, 1, 1, 1]);
        end

        % The Crucial Missing Transformation: Epstein-Zin Certainty Equivalent Outer Power
        valid_EV = isfinite(EV_base) & (EV_base ~= 0);
        if ezc6(jj) ~= 1; EV_base(valid_EV) = max(EV_base(valid_EV), 0).^ezc6(jj); end
        if ezc8(jj) ~= 1; EV_base(valid_EV) = max(EV_base(valid_EV), 0).^ezc8(jj); end

        % --- SOLIDIFY LAZY TREE (Prevents massive memory reallocation in VRAM) ---
        EV_base_lazy = EV_base;
        EV_base = zeros(size(EV_base_lazy), 'like', EV_base_lazy);
        EV_base(:) = EV_base_lazy(:);

        EV_pack{i_ev} = reshape(EV_base, [N_a, N_ze, N_dsemiz]);
    end

    EV_Valt_flat_ze = EV_pack{1};
    if is_naive; EV_belief_flat_ze = EV_pack{2}; else; EV_belief_flat_ze = EV_Valt_flat_ze; end

    V_j_max = zeros(N_a, N_ze, 'like', Valt_next);
    Valt_j_max = zeros(N_a, N_ze, 'like', Valt_next);
    if is_naive; V_exp_j_max = zeros(N_a, N_ze, 'like', Valt_next); end

    Pol_apr_max = zeros(N_a, N_ze, 'like', Valt_next);
    Pol_d_max = zeros(N_a, N_ze, 'like', Valt_next);
    Pol_L2idx_max = zeros(N_a, N_ze, 'like', Valt_next);
    Pol_L2flag_max = zeros(N_a, N_ze, 'like', Valt_next);

    if N_dsemiz > 1
        if isfield(vfoptions, 'l_dsemiz'); N_d_prefix = prod(max(1, n_d(1:end-vfoptions.l_dsemiz))); else; N_d_prefix = prod(max(1, n_d(1:end-1))); end
        dsemiz_idx = ceil((1:N_d_safe)' / N_d_prefix);
        dsemiz_idx_tensor = reshape(dsemiz_idx, [N_d_safe, 1, 1, 1]);
    else; dsemiz_idx_tensor = ones(N_d_safe, 1, 1, 1); end

    % =========================================================================
    % PHASE 4: THE MASTER ORCHESTRATOR
    % =========================================================================
    if vfoptions.divideandconquer == 1
        % --- SCENARIO 4A: Divide and Conquer Active ---
        for i_ze = 1:length(ze_chunks)
            meta = chunk_meta{i_ze}; n_z_loc = meta.n_z_loc; n_e_loc = meta.n_e_loc;
            curr_ze = ze_chunks{i_ze}; N_ze_local = length(curr_ze);

            EV_belief_local = EV_belief_flat_ze(:, curr_ze, :);
            EV_Valt_local = EV_Valt_flat_ze(:, curr_ze, :);

            num_z_vars = length(n_z); Z_cells_local = cell(1, num_z_vars);
            if size(z_gridvals_J, 2) ~= num_z_vars
                z_inflated = reshape(z_gridvals_J, [prod(n_z), num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
            else
                for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
            end

            if has_e
                num_e_vars = size(e_work, 2); E_cells_local = cell(1, num_e_vars);
                for ie_var = 1:num_e_vars; E_cells_local{ie_var} = reshape(e_work(meta.e_vals, ie_var), [1, 1, 1, 1, n_e_loc]); end
            else; E_cells_local = {}; end

            if vfoptions.gridinterplayer(1) == 1
                N_cols = N_ze_local * N_dsemiz; zero_weights = (interp_weights == 0); one_weights = (interp_weights == 1);

                EV_2d_b = reshape(EV_belief_local, [N_a1, N_cols]);
                EV_left_b = EV_2d_b(interp_left_idx, :); EV_right_b = EV_2d_b(interp_right_idx, :);
                EV_interp_flat_b = EV_left_b + interp_weights .* (EV_right_b - EV_left_b);
                EV_interp_flat_b(zero_weights, :) = EV_left_b(zero_weights, :); EV_interp_flat_b(one_weights, :) = EV_right_b(one_weights, :);
                EV_interp_flat_b(isnan(EV_interp_flat_b)) = -Inf;
                EV_belief_interp = reshape(EV_interp_flat_b, [length(a1prime_grid), N_ze_local, N_dsemiz]);

                EV_2d_v = reshape(EV_Valt_local, [N_a1, N_cols]);
                EV_left_v = EV_2d_v(interp_left_idx, :); EV_right_v = EV_2d_v(interp_right_idx, :);
                EV_interp_flat_v = EV_left_v + interp_weights .* (EV_right_v - EV_left_v);
                EV_interp_flat_v(zero_weights, :) = EV_left_v(zero_weights, :); EV_interp_flat_v(one_weights, :) = EV_right_v(one_weights, :);
                EV_interp_flat_v(isnan(EV_interp_flat_v)) = -Inf;
                EV_Valt_interp = reshape(EV_interp_flat_v, [length(a1prime_grid), N_ze_local, N_dsemiz]);
            else
                EV_belief_interp = []; EV_Valt_interp = [];
            end

            if l_a2 == 0
                EV_b_slice = reshape(EV_belief_local, [N_a1, n_z_loc, n_e_loc, N_dsemiz]);
                EV_belief_pre = permute(EV_b_slice(:, :, :, dsemiz_idx_tensor(:)), [4, 1, 5, 2, 3]);
                EV_v_slice = reshape(EV_Valt_local, [N_a1, n_z_loc, n_e_loc, N_dsemiz]);
                EV_Valt_pre = permute(EV_v_slice(:, :, :, dsemiz_idx_tensor(:)), [4, 1, 5, 2, 3]);

                d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]);
                z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1), [1, 1, 1, n_z_loc, 1]);
                e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1 * n_z_loc), [1, 1, 1, 1, n_e_loc]);
                static_EV_offset = d_vec + 1 + z_vec + e_vec;
            else
                EV_belief_pre = []; EV_Valt_pre = []; static_EV_offset = [];
            end

            % --- PASS 1: The Exponential Belief Pass (Valt expectation, evaluated at beta = 1.0) ---
            if is_naive
                LocalBlockFn_Exp = @(state_idx, loweredge_matrix, maxgap_scalar) QHEZ_SlicerWrapper(...
                    state_idx, loweredge_matrix, maxgap_scalar, N_ze_local, ...
                    @(s, l, m) Evaluate_QHEZ_TensorBlock(...
                    s, l, m, N_a1, N_a2, N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
                    0, n2short, n2long, 1.0, delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                    EV_belief_local, EV_belief_pre, EV_belief_interp, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 1));

                [v_exp_c, ~, ~] = ValueFnIter_DC1_Slicer(N_a1 * N_a2, N_a, 1, N_ze_local, vfoptions, LocalBlockFn_Exp);
                V_exp_j_max(:, curr_ze) = reshape(v_exp_c, [N_a1 * N_a2, N_ze_local]);
            end

            % --- PASS 2: The Actual Reality Pass (Valt expectation, evaluated at present-bias beta0) ---
            full_state_chunk = 1:(N_a1 * N_a2);
            if vfoptions.gridinterplayer(1) == 1
                LocalBlockFn_Actual_Coarse_Flat = @(state_idx, loweredge_matrix, maxgap_scalar) QHEZ_SlicerWrapper(...
                    state_idx, loweredge_matrix, maxgap_scalar, N_ze_local, ...
                    @(s, l, m) Evaluate_QHEZ_TensorBlock(...
                    s, l, m, N_a1, N_a2, N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
                    0, n2short, n2long, beta0_j(jj), delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                    EV_Valt_local, EV_Valt_pre, EV_Valt_interp, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 2));

                LocalBlockFn_Actual_Zoom_Flat = @(state_idx, loweredge_matrix, maxgap_scalar) QHEZ_SlicerWrapper(...
                    state_idx, loweredge_matrix, maxgap_scalar, N_ze_local, ...
                    @(s, l, m) Evaluate_QHEZ_TensorBlock(...
                    s, l, m, N_a1, N_a2, N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
                    vfoptions.gridinterplayer, n2short, n2long, beta0_j(jj), delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                    EV_Valt_local, EV_Valt_pre, EV_Valt_interp, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 1));

                if ~isfield(vfoptions, 'level1n')
                    vfoptions.level1n = max(2, floor(sqrt(N_a1)));
                end
                level1n_scalar = vfoptions.level1n(1);
                level1ii = round(linspace(1, N_a1, level1n_scalar)); level1iidiff = level1ii(2:end) - level1ii(1:end-1) - 1;

                [~, ~, ~, ~, ~, p_a1_per_a2_L1] = LocalBlockFn_Actual_Coarse_Flat(level1ii(:)', [], 0);
                maxindex1 = reshape(p_a1_per_a2_L1, [N_d_safe, 1, length(level1ii), 1, N_ze_local]);

                loweredge_pass = zeros(N_d_safe, 1, N_a1, 1, N_ze_local, 'like', EV_Valt_local);
                loweredge_pass(:, :, level1ii, :, :) = maxindex1;
                maxgap = squeeze(max(max(max(max(maxindex1(:,:,2:end,:,:) - maxindex1(:,:,1:end-1,:,:), [], 5), [], 4), [], 2), [], 1));
                if isempty(maxgap); maxgap = 0; end

                for ii = 1:(level1n_scalar - 1)
                    curraindex = (level1ii(ii)+1 : level1ii(ii+1)-1)'; if isempty(curraindex); continue; end
                    if maxgap(ii) > 0
                        loweredge = min(maxindex1(:, :, ii, :, :), N_a1); upper_bound_req = loweredge + maxgap(ii);
                        mg_eval = max(maxgap(ii), max(upper_bound_req - loweredge, [], 'all')); mg_eval = min(mg_eval, N_a1 - 1);
                        loweredge = min(loweredge, N_a1 - mg_eval); loweredge_rep = repmat(loweredge, [1, 1, length(curraindex), 1, 1]);
                        [~, ~, ~, ~, ~, p_a1_per_a2_L2] = LocalBlockFn_Actual_Coarse_Flat(curraindex(:)', loweredge_rep(:), mg_eval);
                        maxindex_L2 = reshape(p_a1_per_a2_L2, [N_d_safe, 1, length(curraindex), 1, N_ze_local]);
                        loweredge_pass(:, :, curraindex, :, :) = maxindex_L2;
                    else
                        loweredge_pass(:, :, curraindex, :, :) = repmat(maxindex1(:, :, ii, :, :), [1, 1, level1iidiff(ii), 1, 1]);
                    end
                end
                loweredge_pass = reshape(loweredge_pass, [N_d_safe, 1, N_a1, N_ze_local]);

                v = zeros(N_a1, N_ze_local, 'like', EV_Valt_local); valt = zeros(N_a1, N_ze_local, 'like', EV_Valt_local);
                p_apr = zeros(N_a1, N_ze_local, 'like', EV_Valt_local); p_d = zeros(N_a1, N_ze_local, 'like', EV_Valt_local);
                p_l2idx = zeros(N_a1, N_ze_local, 'like', EV_Valt_local); p_l2flag = zeros(N_a1, N_ze_local, 'like', EV_Valt_local);

                flat_choices = N_d_safe * n2long;
                max_a1_per_chunk = max(1, floor(safe_elements / (flat_choices * N_ze_local)));

                for chunk_start = 1:max_a1_per_chunk:N_a1
                    chunk_end = min(N_a1, chunk_start + max_a1_per_chunk - 1); state_chunk = (chunk_start:chunk_end)';
                    loweredge_chunk = loweredge_pass(:, :, state_chunk, :);
                    [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c, ~, valt_c] = LocalBlockFn_Actual_Zoom_Flat(state_chunk(:)', loweredge_chunk, n2long - 1);
                    v(state_chunk, :) = v_c; valt(state_chunk, :) = valt_c; p_apr(state_chunk, :) = p_apr_c;
                    p_d(state_chunk, :) = p_d_c; p_l2idx(state_chunk, :) = p_l2idx_c; p_l2flag(state_chunk, :) = p_l2flag_c;
                end
            else
                LocalBlockFn_Actual_Coarse = @(state_idx, loweredge_matrix, maxgap_scalar) QHEZ_SlicerWrapper(...
                    state_idx, loweredge_matrix, maxgap_scalar, N_ze_local, ...
                    @(s, l, m) Evaluate_QHEZ_TensorBlock(...
                    s, l, m, N_a1, N_a2, N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
                    0, n2short, n2long, beta0_j(jj), delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                    EV_Valt_local, EV_Valt_pre, EV_Valt_interp, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 1));

                [v, p_apr, p_d] = ValueFnIter_DC1_Slicer(N_a1 * N_a2, N_a, 1, N_ze_local, vfoptions, LocalBlockFn_Actual_Coarse);
                [~, ~, ~, p_l2idx, p_l2flag, ~, valt] = LocalBlockFn_Actual_Coarse(full_state_chunk, p_apr, 0);
            end

            V_j_max(:, curr_ze)     = reshape(v,     [N_a1 * N_a2, N_ze_local]);
            Valt_j_max(:, curr_ze)  = reshape(valt,  [N_a1 * N_a2, N_ze_local]);
            Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a1 * N_a2, N_ze_local]);
            Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a1 * N_a2, N_ze_local]);
            if vfoptions.gridinterplayer(1) == 1
                Pol_L2idx_max(:, curr_ze)  = reshape(p_l2idx,  [N_a1 * N_a2, N_ze_local]);
                Pol_L2flag_max(:, curr_ze) = reshape(p_l2flag, [N_a1 * N_a2, N_ze_local]);
            end
        end
    else
        % --- SCENARIO 4B: Full Tensor Evaluation Active ---
        for i_a2 = 1:length(a2_chunks)
            curr_a2 = a2_chunks{i_a2}; N_a2_local = length(curr_a2);
            start_a_idx = (min(curr_a2) - 1) * N_a1 + 1; end_a_idx   = max(curr_a2) * N_a1;

            for i_ze = 1:length(ze_chunks)
                meta = chunk_meta{i_ze}; n_z_loc = meta.n_z_loc; n_e_loc = meta.n_e_loc;
                curr_ze = ze_chunks{i_ze}; N_ze_local = length(curr_ze);

                if l_a2 > 0; A2_local = A2_mat(curr_a2, :); else; A2_local = []; end
                EV_belief_local = EV_belief_flat_ze(:, curr_ze, :); EV_Valt_local = EV_Valt_flat_ze(:, curr_ze, :);

                num_z_vars = length(n_z); Z_cells_local = cell(1, num_z_vars);
                if size(z_gridvals_J, 2) ~= num_z_vars
                    z_inflated = reshape(z_gridvals_J, [prod(n_z), num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                    for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
                else
                    for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
                end

                if has_e
                    num_e_vars = size(e_work, 2); E_cells_local = cell(1, num_e_vars);
                    for ie_var = 1:num_e_vars; E_cells_local{ie_var} = reshape(e_work(meta.e_vals, ie_var), [1, 1, 1, 1, n_e_loc]); end
                else; E_cells_local = {}; end

                if l_a2 == 0
                    EV_b_slice = reshape(EV_belief_local, [N_a1, n_z_loc, n_e_loc, N_dsemiz]);
                    EV_belief_pre = permute(EV_b_slice(:, :, :, dsemiz_idx_tensor(:)), [4, 1, 5, 2, 3]);
                    EV_v_slice = reshape(EV_Valt_local, [N_a1, n_z_loc, n_e_loc, N_dsemiz]);
                    EV_Valt_pre = permute(EV_v_slice(:, :, :, dsemiz_idx_tensor(:)), [4, 1, 5, 2, 3]);
                    d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]);
                    z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1), [1, 1, 1, n_z_loc, 1]);
                    e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1 * n_z_loc), [1, 1, 1, 1, n_e_loc]);
                    static_EV_offset = d_vec + 1 + z_vec + e_vec;
                else
                    EV_belief_pre = []; EV_Valt_pre = []; static_EV_offset = [];
                end

                if vfoptions.gridinterplayer(1) == 1
                    % --- GI PRECOMPUTATION ---
                    N_cols = N_ze_local * N_dsemiz; zero_weights = (interp_weights == 0); one_weights = (interp_weights == 1);
                    if l_a2 > 0
                        EV_2d_b = reshape(EV_belief_local, [N_a1, N_a2_local * N_cols]);
                        EV_left_b = EV_2d_b(interp_left_idx, :); EV_right_b = EV_2d_b(interp_right_idx, :);
                        EV_interp_flat_b = EV_left_b + interp_weights .* (EV_right_b - EV_left_b);
                        EV_interp_flat_b(zero_weights, :) = EV_left_b(zero_weights, :); EV_interp_flat_b(one_weights, :) = EV_right_b(one_weights, :);
                        EV_interp_flat_b(isnan(EV_interp_flat_b)) = -Inf;
                        EV_belief_interp = reshape(EV_interp_flat_b, [length(a1prime_grid), N_a2_local, N_ze_local, N_dsemiz]);

                        EV_2d_v = reshape(EV_Valt_local, [N_a1, N_a2_local * N_cols]);
                        EV_left_v = EV_2d_v(interp_left_idx, :); EV_right_v = EV_2d_v(interp_right_idx, :);
                        EV_interp_flat_v = EV_left_v + interp_weights .* (EV_right_v - EV_left_v);
                        EV_interp_flat_v(zero_weights, :) = EV_left_v(zero_weights, :); EV_interp_flat_v(one_weights, :) = EV_right_v(one_weights, :);
                        EV_interp_flat_v(isnan(EV_interp_flat_v)) = -Inf;
                        EV_Valt_interp = reshape(EV_interp_flat_v, [length(a1prime_grid), N_a2_local, N_ze_local, N_dsemiz]);
                    else
                        EV_2d_b = reshape(EV_belief_local, [N_a1, N_cols]);
                        EV_left_b = EV_2d_b(interp_left_idx, :); EV_right_b = EV_2d_b(interp_right_idx, :);
                        EV_interp_flat_b = EV_left_b + interp_weights .* (EV_right_b - EV_left_b);
                        EV_interp_flat_b(zero_weights, :) = EV_left_b(zero_weights, :); EV_interp_flat_b(one_weights, :) = EV_right_b(one_weights, :);
                        EV_interp_flat_b(isnan(EV_interp_flat_b)) = -Inf;
                        EV_belief_interp = reshape(EV_interp_flat_b, [length(a1prime_grid), N_ze_local, N_dsemiz]);

                        EV_2d_v = reshape(EV_Valt_local, [N_a1, N_cols]);
                        EV_left_v = EV_2d_v(interp_left_idx, :); EV_right_v = EV_2d_v(interp_right_idx, :);
                        EV_interp_flat_v = EV_left_v + interp_weights .* (EV_right_v - EV_left_v);
                        EV_interp_flat_v(zero_weights, :) = EV_left_v(zero_weights, :); EV_interp_flat_v(one_weights, :) = EV_right_v(one_weights, :);
                        EV_interp_flat_v(isnan(EV_interp_flat_v)) = -Inf;
                        EV_Valt_interp = reshape(EV_interp_flat_v, [length(a1prime_grid), N_ze_local, N_dsemiz]);
                    end
                else
                    EV_belief_interp = []; EV_Valt_interp = [];
                end

                state_list = start_a_idx:end_a_idx; total_states = length(state_list);
                flat_choices = max(1, N_d_safe) * N_a1;
                max_states_per_chunk = max(1, floor(safe_elements / (flat_choices * n_z_loc * n_e_loc)));
                v_concat = []; valt_concat = []; p_apr_concat = []; p_d_concat = []; p_l2idx_concat = []; p_l2flag_concat = [];

                for chunk_start = 1:max_states_per_chunk:total_states
                    chunk_end = min(total_states, chunk_start + max_states_per_chunk - 1); state_chunk = state_list(chunk_start:chunk_end);

                    % --- PASS 1: The Exponential Belief Pass (Valt expectation, evaluated at beta = 1.0) ---
                    if is_naive
                        if vfoptions.gridinterplayer(1) == 1
                            [~, ~, ~, ~, ~, p_a1_per_a2_exp] = Evaluate_QHEZ_TensorBlock(...
                                state_chunk, [], 0, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                                Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                                0, n2short, n2long, 1.0, delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                                EV_belief_local, EV_belief_pre, EV_belief_interp, a1prime_grid, ...
                                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                                TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 2);

                            loweredge_chunk_exp = reshape(p_a1_per_a2_exp, [N_d_safe, max(1, N_a2_local), length(state_chunk), N_ze_local]);
                            v_exp_c = Evaluate_QHEZ_TensorBlock(...
                                state_chunk, loweredge_chunk_exp, n2long - 1, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                                Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                                vfoptions.gridinterplayer, n2short, n2long, 1.0, delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                                EV_belief_local, EV_belief_pre, EV_belief_interp, a1prime_grid, ...
                                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                                TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 0);
                        else
                            v_exp_c = Evaluate_QHEZ_TensorBlock(...
                                state_chunk, [], 0, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                                Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                                0, n2short, n2long, 1.0, delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                                EV_belief_local, EV_belief_pre, EV_belief_interp, a1prime_grid, ...
                                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                                TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 0);
                        end
                        V_exp_j_max(state_chunk, curr_ze) = reshape(v_exp_c, [length(state_chunk), N_ze_local]);
                    end

                    % --- PASS 2: The Actual Reality Pass (Valt expectation, evaluated at present-bias beta0) ---
                    if vfoptions.gridinterplayer(1) == 1
                        [~, ~, ~, ~, ~, p_a1_per_a2] = Evaluate_QHEZ_TensorBlock(...
                            state_chunk, [], 0, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                            Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                            0, n2short, n2long, beta0_j(jj), delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                            EV_Valt_local, EV_Valt_pre, EV_Valt_interp, a1prime_grid, ...
                            TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                            TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 2);

                        loweredge_chunk = reshape(p_a1_per_a2, [N_d_safe, max(1, N_a2_local), length(state_chunk), N_ze_local]);
                        [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c, ~, valt_c] = Evaluate_QHEZ_TensorBlock(...
                            state_chunk, loweredge_chunk, n2long - 1, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                            Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                            vfoptions.gridinterplayer, n2short, n2long, beta0_j(jj), delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                            EV_Valt_local, EV_Valt_pre, EV_Valt_interp, a1prime_grid, ...
                            TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                            TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 0);
                    else
                        [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c, ~, valt_c] = Evaluate_QHEZ_TensorBlock(...
                            state_chunk, [], 0, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                            Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                            0, n2short, n2long, beta0_j(jj), delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
                            EV_Valt_local, EV_Valt_pre, EV_Valt_interp, a1prime_grid, ...
                            TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                            TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, 0);
                    end
                    v_concat = [v_concat; v_c]; valt_concat = [valt_concat; valt_c]; p_apr_concat = [p_apr_concat; p_apr_c]; p_d_concat = [p_d_concat; p_d_c];
                    if vfoptions.gridinterplayer(1) == 1; p_l2idx_concat = [p_l2idx_concat; p_l2idx_c]; p_l2flag_concat = [p_l2flag_concat; p_l2flag_c]; end
                end
                if l_a2 > 0; N_a_local = N_a1 * N_a2_local; else; N_a_local = N_a1; end
                V_j_max(start_a_idx:end_a_idx, curr_ze)     = reshape(v_concat,     [N_a_local, N_ze_local]);
                Valt_j_max(start_a_idx:end_a_idx, curr_ze)  = reshape(valt_concat,  [N_a_local, N_ze_local]);
                Pol_apr_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_apr_concat, [N_a_local, N_ze_local]);
                Pol_d_max(start_a_idx:end_a_idx, curr_ze)   = reshape(p_d_concat,   [N_a_local, N_ze_local]);
                if vfoptions.gridinterplayer(1) == 1
                    Pol_L2idx_max(start_a_idx:end_a_idx, curr_ze)  = reshape(p_l2idx_concat,  [N_a_local, N_ze_local]);
                    Pol_L2flag_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_l2flag_concat, [N_a_local, N_ze_local]);
                end
            end
        end
    end

    % Prepare for next iteration
    V_j_max     = reshape(V_j_max,     [N_a, n_z_work, n_e_work]);
    Valt_j_max  = reshape(Valt_j_max,  [N_a, n_z_work, n_e_work]);
    Pol_apr_max = reshape(Pol_apr_max, [N_a, n_z_work, n_e_work]);
    Pol_d_max   = reshape(Pol_d_max,   [N_a, n_z_work, n_e_work]);

    if vfoptions.gridinterplayer(1) == 1
        Pol_L2idx_max  = reshape(Pol_L2idx_max,  [N_a, n_z_work, n_e_work]);
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_a, n_z_work, n_e_work]);
        lower_grid_pt = Pol_apr_max; subgrid_step  = Pol_L2idx_max;
        if N_d > 0; PolicyKron(1, :, :, :, jj) = (lower_grid_pt - 1) * N_d + Pol_d_max; else; PolicyKron(1, :, :, :, jj) = lower_grid_pt; end
        PolicyKron(2, :, :, :, jj) = subgrid_step; PolicyKron(3, :, :, :, jj) = Pol_L2flag_max;
    else
        if N_d > 0; PolicyKron(:, :, :, jj) = (Pol_apr_max - 1) * N_d + Pol_d_max; else; PolicyKron(:, :, :, jj) = Pol_apr_max; end
    end

    V(:, :, :, jj) = V_j_max;
    Valt(:, :, :, jj) = Valt_j_max; Valt_next = Valt_j_max;
    if is_naive; V_exp_next = V_exp_j_max; end
end

% =========================================================================
% PHASE 5: SYSTEM RAM HANDOFF & POLICY UNPACKING
% =========================================================================
if N_z == 0; V = squeeze(V); Valt = squeeze(Valt); end
if N_d == 0; n_daprime = n_a(1:length(n_a)); else; n_daprime = [n_d, n_a(1:length(n_a))]; end

if vfoptions.gridinterplayer(1) ~= 1; PolicyKron = shiftdim(PolicyKron, -1); end

disp('Unpacking QHEZ Policy tensor to System RAM...');
num_pol_vars = length(n_daprime); n_daprime_col = n_daprime(:); divisors = cumprod([1; n_daprime_col(1:end-1)]);

if vfoptions.gridinterplayer(1) == 1
    BaseIndexKron = PolicyKron(1, :, :, :, :);
    P_base_gpu = mod(floor((BaseIndexKron - 1) ./ divisors), n_daprime_col) + 1;
    Policy_flat = gather([P_base_gpu; PolicyKron(2:3, :, :, :, :)]);
else
    Policy_flat = gather(mod(floor((PolicyKron - 1) ./ divisors), n_daprime_col) + 1);
end

out_pol_vars = size(Policy_flat, 1);
out_n_a = n_a(n_a > 0); if isempty(out_n_a); out_n_a = 1; end
out_n_z = n_z(n_z > 0); if isempty(out_n_z); out_n_z = 1; end

state_shape = out_n_a;
if prod(n_z) > 0; state_shape = [state_shape, out_n_z]; end
if has_e; state_shape = [state_shape, n_e_pass]; end
state_shape = [state_shape, N_j];

Policy = reshape(Policy_flat, [out_pol_vars, state_shape]);
V = reshape(gather(V), state_shape);
Valt = reshape(gather(Valt), state_shape);


end


function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max, Pol_a1_per_a2, Valt_j_max] = Evaluate_QHEZ_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d_safe, N_ze_local, ...
    Z_cells_block, E_cells_block, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
    gridinterplayer, n2short, n2long, beta_j, delta_j, ezc1_j, ezc9_j, EV_belief_local, EV_belief_pre, EV_belief_interp, ...
    EV_Valt_local, EV_Valt_pre, EV_Valt_interp, a1prime_grid, ...
    TensorReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j, ...
    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, is_dc_mode)

% --- VALT BYPASS OPTIMIZATION ---
compute_valt = (nargout > 6);
N_states = length(state_idx);

if l_a2 > 0; [a1_sub, a2_sub] = ind2sub([N_a1, size(A2_mat, 1)], state_idx); else; a1_sub = state_idx; a2_sub = []; end

num_a1 = size(A1_mat, 2);
% --- REBUILD LOCAL 5D ORTHOGONAL STATE CELLS FOR NATIVE EXECUTION ---
N_a2_len = N_states / N_a1_total;
is_cartesian = (N_a2_len == floor(N_a2_len)) && isempty(loweredge_matrix) && (a1_sub(1) == 1) && (a1_sub(end) == N_a1_total);

% Set safe defaults for all standard branches
Z_cells_eval = Z_cells_block;
E_cells_eval = E_cells_block;
A2_cells = {};

if is_cartesian
    A1_cells = cell(1, l_a1);
    for ia = 1:l_a1; A1_cells{ia} = reshape(A1_mat(1:N_a1_total, ia), [1, 1, N_a1_total, 1, 1, 1]); end
    if N_a2 > 1
        A2_cells = cell(1, l_a2);
        a2_unique_idx = a2_sub(1:N_a1_total:end);
        for ia = 1:l_a2; A2_cells{ia} = reshape(A2_mat(a2_unique_idx, ia), [1, 1, 1, N_a2_len, 1, 1]); end

        % Only override Z and E if building a full 6D tensor mesh
        Z_cells_eval = cell(1, length(Z_cells_block));
        for iz = 1:length(Z_cells_block); Z_cells_eval{iz} = reshape(Z_cells_block{iz}(1,1,1,:,1), [1, 1, 1, 1, n_z_loc, 1]); end
        E_cells_eval = cell(1, length(E_cells_block));
        for ie = 1:length(E_cells_block); E_cells_eval{ie} = reshape(E_cells_block{ie}(1,1,1,1,:), [1, 1, 1, 1, 1, n_e_loc]); end
    end
else
    A1_cells = cell(1, l_a1);
    for ia = 1:l_a1; A1_cells{ia} = reshape(A1_mat(a1_sub, ia), [1, 1, N_states, 1, 1]); end
    if N_a2 > 1
        A2_cells = cell(1, l_a2);
        for ia = 1:l_a2; A2_cells{ia} = reshape(A2_mat(a2_sub, ia), [1, 1, N_states, 1, 1]); end
    end
end

is_coarse = isempty(loweredge_matrix) && (gridinterplayer(1) == 0 || is_dc_mode == 2);

% =========================================================================
% PHASE 1: EVALUATION (CHOICE MATRIX GENERATION)
% =========================================================================
is_coarse = (gridinterplayer(1) == 0 || is_dc_mode == 2);

if isempty(loweredge_matrix)
    if is_coarse
        num_choices = N_a1;
        Apr_cells = cell(1, num_a1);
        for ia = 1:num_a1; Apr_cells{ia} = reshape(A1_mat(:, ia), [1, num_choices, 1, 1, 1]); end
        EV_source_b = EV_belief_pre; EV_source_v = EV_Valt_pre; target_offset = static_EV_offset;
        a1_target_grid = A1_grids_1d{1};
    else
        num_choices = length(a1prime_grid);
        Apr_cells = cell(1, num_a1);
        Apr_cells{1} = reshape(a1prime_grid, [1, num_choices, 1, 1, 1]);
        if num_a1 > 1; for ia = 2:num_a1; Apr_cells{ia} = reshape(A1_mat(a1_sub, ia), [1, 1, N_states, 1, 1]); end; end
        EV_source_b = EV_belief_interp; EV_source_v = EV_Valt_interp; target_offset = static_EV_offset_fine;
        a1_target_grid = a1prime_grid;
    end

    if N_a2 > 1
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_eval{:}, E_cells_eval{:}, ReturnFnParamsCell{:});
    else
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_eval{:}, E_cells_eval{:}, ReturnFnParamsCell{:});
    end

    if ~is_cartesian
        if isa(target_offset, 'gpuArray')
            choice_idx_linear = gpuArray(reshape(1:num_choices, [1, num_choices, 1, 1, 1]));
        else
            choice_idx_linear = reshape(1:num_choices, [1, num_choices, 1, 1, 1]);
        end
        if N_a2 > 1
            a2_offset = reshape(a2_sub - 1, [1, 1, N_states, 1, 1]) * (N_d_safe * length(a1_target_grid));
            lin_idx_compact = target_offset + ((choice_idx_linear - 1) * N_d_safe + a2_offset);
            EV_belief_bounded = EV_source_b(lin_idx_compact);
            if compute_valt; EV_Valt_bounded = EV_source_v(lin_idx_compact); end
        else
            if is_coarse
                lin_idx_compact = target_offset + (choice_idx_linear - 1) * N_d_safe;
                EV_belief_bounded = EV_source_b(lin_idx_compact);
                if compute_valt; EV_Valt_bounded = EV_source_v(lin_idx_compact); end
            else
                stride_z = num_choices;
                if isa(EV_source_b, 'gpuArray')
                    ze_offset = gpuArray(reshape((0:N_ze_local-1) * stride_z, [1, 1, 1, n_z_loc, n_e_loc]));
                else
                    ze_offset = reshape((0:N_ze_local-1) * stride_z, [1, 1, 1, n_z_loc, n_e_loc]);
                end
                if N_dsemiz > 1; ze_offset = ze_offset + (dsemiz_idx_tensor - 1) * (stride_z * N_ze_local); end
                EV_belief_bounded = EV_source_b(choice_idx_linear + ze_offset);
                if compute_valt; EV_Valt_bounded = EV_source_v(choice_idx_linear + ze_offset); end
            end
        end
    end
else
    % ZOOM PHASE
    [choice_idx_linear, out_of_bounds, num_choices, Apr_cells, num_choices_total_a1, start_offset, loweredge_matrix_bounds] = Helper_SlicerBounds_QHEZ(...
        loweredge_matrix, N_a1, N_states, n_z_loc, n_e_loc, N_d_safe, n2short, a1prime_grid, num_a1, A1_mat, a1_sub, EV_belief_local);

    if is_coarse
        target_offset = static_EV_offset;
        EV_source_b = EV_belief_pre; EV_source_v = EV_Valt_pre;
        a1_target_grid = A1_grids_1d{1};
    else
        target_offset = static_EV_offset_fine;
        EV_source_b = EV_belief_interp; EV_source_v = EV_Valt_interp;
        a1_target_grid = a1prime_grid;
    end

    if N_a2 > 1
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_eval{:}, E_cells_eval{:}, ReturnFnParamsCell{:});
        a2_offset = reshape(a2_sub - 1, [1, 1, N_states, 1, 1]) * (N_d_safe * length(a1_target_grid));
        lin_idx_compact = target_offset + ((choice_idx_linear - 1) * N_d_safe + a2_offset);
        EV_belief_bounded = EV_source_b(lin_idx_compact);
        if compute_valt; EV_Valt_bounded = EV_source_v(lin_idx_compact); end
    else
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_eval{:}, E_cells_eval{:}, ReturnFnParamsCell{:});
        if is_coarse
            lin_idx_compact = target_offset + (choice_idx_linear - 1) * N_d_safe;
            EV_belief_bounded = EV_source_b(lin_idx_compact);
            if compute_valt; EV_Valt_bounded = EV_source_v(lin_idx_compact); end
        else
            stride_z = length(a1_target_grid);
            if isa(EV_source_b, 'gpuArray')
                ze_offset = gpuArray(reshape((0:N_ze_local-1) * stride_z, [1, 1, 1, n_z_loc, n_e_loc]));
            else
                ze_offset = reshape((0:N_ze_local-1) * stride_z, [1, 1, 1, n_z_loc, n_e_loc]);
            end
            if N_dsemiz > 1; ze_offset = ze_offset + (dsemiz_idx_tensor - 1) * (stride_z * N_ze_local); end
            EV_belief_bounded = EV_source_b(choice_idx_linear + ze_offset);
            if compute_valt; EV_Valt_bounded = EV_source_v(choice_idx_linear + ze_offset); end
        end
    end
    EV_belief_bounded(out_of_bounds) = -Inf;
    if compute_valt
        EV_Valt_bounded(out_of_bounds) = -Inf;
    end
end

% =========================================================================
% PHASE 2: UNIVERSAL RHS EVALUATION & ADDITION
% =========================================================================
FLAT_CHOICES = max(1, N_d_safe) * num_choices;
FLAT_STATES  = N_states * N_ze_local;
is_EZ = ~(all(ezc2_j == 1) && ezc3 == 1 && ezc4 == 1 && all(ezc7_j == 1));

weight_belief = beta_j * delta_j * ezc9_j;
weight_valt   = delta_j * ezc9_j;

if is_cartesian
    if isempty(loweredge_matrix) && (gridinterplayer(1) == 0 || is_dc_mode == 2)
        EV_slice_b = reshape(EV_belief_pre, [N_d_safe, num_choices, 1, N_a2_len, N_ze_local]);
        if compute_valt; EV_slice_v = reshape(EV_Valt_pre, [N_d_safe, num_choices, 1, N_a2_len, N_ze_local]); end
    else
        EV_slice_b = reshape(EV_belief_interp, [N_d_safe, num_choices, 1, N_a2_len, N_ze_local]);
        if compute_valt; EV_slice_v = reshape(EV_Valt_interp, [N_d_safe, num_choices, 1, N_a2_len, N_ze_local]); end
    end

    if size(F_tensor, 3) ~= N_a1 || size(F_tensor, 4) ~= max(1, N_a2_len)
        F_tensor = F_tensor + zeros([1, 1, N_a1, max(1, N_a2_len), 1], 'like', F_tensor);
    end
    F_tensor_native = reshape(F_tensor, [N_d_safe, num_choices, N_a1, max(1, N_a2_len), N_ze_local]);

    EV_expanded_b = EV_slice_b + zeros(size(F_tensor_native), 'like', EV_slice_b);

    if is_EZ
        RHS_belief_native = Evaluate_Universal_RHS_VFHorz(F_tensor_native, EV_expanded_b, ezc1_j, weight_belief, ezc2_j, ezc3, ezc4, ezc7_j);
    else
        RHS_belief_native = F_tensor_native + weight_belief .* EV_expanded_b;
    end
    RHS_belief_flat = reshape(RHS_belief_native, [FLAT_CHOICES, FLAT_STATES]);

    if compute_valt
        EV_expanded_v = EV_slice_v + zeros(size(F_tensor_native), 'like', EV_slice_v);
        if is_EZ
            RHS_Valt_native = Evaluate_Universal_RHS_VFHorz(F_tensor_native, EV_expanded_v, ezc1_j, weight_valt, ezc2_j, ezc3, ezc4, ezc7_j);
        else
            RHS_Valt_native = F_tensor_native + weight_valt .* EV_expanded_v;
        end
        RHS_Valt_flat = reshape(RHS_Valt_native, [FLAT_CHOICES, FLAT_STATES]);
    else
        RHS_Valt_flat = [];
    end
else
    EV_expanded_b = EV_belief_bounded + zeros(size(F_tensor), 'like', EV_belief_bounded);

    if is_EZ
        RHS_belief_native = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_expanded_b, ezc1_j, weight_belief, ezc2_j, ezc3, ezc4, ezc7_j);
    else
        RHS_belief_native = F_tensor + weight_belief .* EV_expanded_b;
    end

    if size(RHS_belief_native, 3) ~= N_states
        RHS_belief_native = RHS_belief_native + zeros([1, 1, N_states, 1, 1], 'like', RHS_belief_native);
    end
    RHS_belief_flat = reshape(RHS_belief_native, [FLAT_CHOICES, FLAT_STATES]);

    if compute_valt
        EV_expanded_v = EV_Valt_bounded + zeros(size(F_tensor), 'like', EV_Valt_bounded);
        if is_EZ
            RHS_Valt_native = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_expanded_v, ezc1_j, weight_valt, ezc2_j, ezc3, ezc4, ezc7_j);
        else
            RHS_Valt_native = F_tensor + weight_valt .* EV_expanded_v;
        end

        if size(RHS_Valt_native, 3) ~= N_states
            RHS_Valt_native = RHS_Valt_native + zeros([1, 1, N_states, 1, 1], 'like', RHS_Valt_native);
        end
        RHS_Valt_flat = reshape(RHS_Valt_native, [FLAT_CHOICES, FLAT_STATES]);
    else
        RHS_Valt_flat = [];
    end
end

% =========================================================================
% PHASE 3: OUTPUT MAPPING
% =========================================================================
if isempty(loweredge_matrix)
    start_offset = 0; num_choices_total_a1 = 0; loweredge_matrix_bounds = [];
end
[V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max, Pol_a1_per_a2, Valt_j_max] = Helper_QHEZ_OutputMapping(...
    RHS_belief_flat, RHS_Valt_flat, compute_valt, is_dc_mode, isempty(loweredge_matrix), N_d_safe, num_choices, 1, ...
    FLAT_STATES, N_states, N_ze_local, N_a1, 0, gridinterplayer, loweredge_matrix_bounds, ...
    start_offset, num_choices_total_a1, n2short, 0, EV_belief_local);


end


function [v, p_apr, p_d, p_l2, p_l2f, p_a1, valt] = QHEZ_SlicerWrapper(state_chunk_idx, low_mat, mg, N_ze_local, CoreFn)
state_chunk = state_chunk_idx(:)';
if isempty(low_mat); low_chunk = []; else; low_chunk = low_mat; end
if nargout > 6
    [v, p_apr, p_d, p_l2, p_l2f, p_a1, valt] = CoreFn(state_chunk, low_chunk, mg);
elseif nargout > 5
    [v, p_apr, p_d, p_l2, p_l2f, p_a1] = CoreFn(state_chunk, low_chunk, mg);
else
    [v, p_apr, p_d, p_l2, p_l2f] = CoreFn(state_chunk, low_chunk, mg);
end


end


function [choice_idx_linear, out_of_bounds, num_choices_total, Apr_cells, num_choices_total_a1, start_offset, loweredge_matrix_bounds] = Helper_SlicerBounds_QHEZ(...
    loweredge_matrix, N_a1_dc, N_states, n_z_loc, n_e_loc, N_d_safe, n2short, a1prime_grid, l_a1, A1_mat, a1_sub, EV_local)

loweredge_matrix = max(1, min(loweredge_matrix, N_a1_dc));
loweredge_matrix = mod(loweredge_matrix - 1, N_a1_dc) + 1;
num_val = numel(loweredge_matrix);
target_shape = zeros(N_d_safe, 1, N_states, n_z_loc, n_e_loc, 'like', loweredge_matrix);
target_states_ze = N_states * n_z_loc * n_e_loc;

if num_val == target_states_ze
    low_reshaped = reshape(loweredge_matrix, [1, 1, N_states, n_z_loc, n_e_loc]);
elseif num_val == N_d_safe * target_states_ze
    low_reshaped = reshape(loweredge_matrix, [N_d_safe, 1, N_states, n_z_loc, n_e_loc]);
else
    low_flat = loweredge_matrix(:);
    if length(low_flat) < target_states_ze; low_flat = repmat(low_flat, ceil(target_states_ze / max(1, length(low_flat))), 1); end
    low_reshaped = reshape(low_flat(1:target_states_ze), [1, 1, N_states, n_z_loc, n_e_loc]);
end
loweredge_matrix = low_reshaped + target_shape;

% FIX: Safely clamp to N_a1_dc - 1 to guarantee the Zoom pass can reach the absolute boundary!
loweredge_matrix_bounds = max(2, min(loweredge_matrix, N_a1_dc - 1));

L2_base = (loweredge_matrix_bounds - 1) * (n2short + 1) + 1;
start_offset = -(n2short + 1);
end_offset = (n2short + 1);
num_choices_total_a1 = end_offset - start_offset + 1;
grid_len = length(a1prime_grid);

choice_idx_linear = reshape(L2_base, [N_d_safe, 1, N_states, n_z_loc, n_e_loc]) ...
    + reshape(start_offset:end_offset, [1, num_choices_total_a1, 1, 1, 1]);
out_of_bounds = (choice_idx_linear < 1) | (choice_idx_linear > grid_len);
choice_idx_linear = max(1, min(choice_idx_linear, grid_len));
Apr_cells = { a1prime_grid(choice_idx_linear) };
if l_a1 > 1
    for ia = 2:l_a1; Apr_cells{ia} = reshape(A1_mat(a1_sub, ia), [1, 1, N_states, 1, 1]); end
end
num_choices_total = num_choices_total_a1;


end


function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max, Pol_a1_per_a2, Valt_j_max] = Helper_QHEZ_OutputMapping(...
    RHS_belief_flat, RHS_Valt_flat, compute_valt, is_dc_mode, is_coarse_mapping, N_d_safe, num_choices_total, N_a1_other, ...
    FLAT_STATES, N_states, N_ze_local, N_a1_dc, d_override, gridinterplayer, loweredge_matrix_bounds, ...
    start_offset, num_choices_total_a1, n2short, d_gap, EV_local)

Pol_L2idx_max = []; Pol_L2flag_max = []; Pol_a1_per_a2 = []; Valt_j_max = [];

if is_coarse_mapping
    if is_dc_mode == 3
        if N_d_safe == 1
            [V_sub_coarse, apr_idx_local] = max(RHS_belief_flat, [], 1);
            V_sub_coarse = reshape(V_sub_coarse, [1, 1, FLAT_STATES]);
            apr_idx_local = reshape(apr_idx_local, [1, 1, FLAT_STATES]);
        else
            RHS_for_d = reshape(RHS_belief_flat, [N_d_safe, num_choices_total, FLAT_STATES]);
            [V_sub_coarse, apr_idx_local] = max(RHS_for_d, [], 2);
            V_sub_coarse = reshape(V_sub_coarse, [N_d_safe, 1, FLAT_STATES]);
            apr_idx_local = reshape(apr_idx_local, [N_d_safe, 1, FLAT_STATES]);
        end
        d_idx_local = repmat(reshape(1:N_d_safe, [N_d_safe, 1]), [1, FLAT_STATES]);
        V_j_max     = reshape(V_sub_coarse,  [N_d_safe, N_states, N_ze_local]);
        Pol_apr_max = reshape(apr_idx_local, [N_d_safe, N_states, N_ze_local]);
        Pol_d_max   = reshape(d_idx_local,   [N_d_safe, N_states, N_ze_local]);

        num_choices_a1 = num_choices_total / N_a1_other;
        if N_d_safe == 1
            RHS_a1 = reshape(RHS_belief_flat, [num_choices_a1, N_a1_other * FLAT_STATES]);
            [~, max_a1_idx_per_d] = max(RHS_a1, [], 1);
        else
            RHS_a1 = reshape(RHS_belief_flat, [N_d_safe, num_choices_a1, N_a1_other, FLAT_STATES]);
            [~, max_a1_idx_per_d] = max(RHS_a1, [], 2);
        end
        Pol_a1_per_a2 = reshape(max_a1_idx_per_d, [N_d_safe, N_a1_other, N_states, N_ze_local]);

        if compute_valt
            if N_d_safe == 1
                Valt_sub_coarse = RHS_Valt_flat;
            else
                lin_idx_valt = repmat(1:N_d_safe, [1, FLAT_STATES])' + (apr_idx_local(:) - 1) * N_d_safe + (0:FLAT_STATES-1)' * (N_d_safe * num_choices_total);
                Valt_sub_coarse = RHS_Valt_flat(lin_idx_valt);
            end
            Valt_j_max = reshape(Valt_sub_coarse, [N_d_safe, N_states, N_ze_local]);
        end
    else
        [V_sub_coarse, Pol_sub_idx] = max(RHS_belief_flat, [], 1);
        num_choices_a1 = num_choices_total / N_a1_other;
        RHS_for_d = reshape(RHS_belief_flat, [N_d_safe, num_choices_a1, N_a1_other, FLAT_STATES]);
        [~, max_a1_idx_per_d] = max(RHS_for_d, [], 2);
        Pol_a1_per_a2 = reshape(max_a1_idx_per_d, [N_d_safe, N_a1_other, N_states, N_ze_local]);

        d_idx_local = mod(Pol_sub_idx - 1, N_d_safe) + 1;
        apr_idx_local  = ceil(Pol_sub_idx / N_d_safe);
        V_j_max        = reshape(V_sub_coarse,  [N_states, N_ze_local]);
        Pol_apr_max    = reshape(apr_idx_local, [N_states, N_ze_local]);
        Pol_d_max      = reshape(d_idx_local,   [N_states, N_ze_local]);

        if compute_valt
            lin_idx_valt = Pol_sub_idx + (0:FLAT_STATES-1) * (N_d_safe * num_choices_total);
            Valt_sub_coarse = RHS_Valt_flat(lin_idx_valt);
            Valt_j_max = reshape(Valt_sub_coarse, [N_states, N_ze_local]);
        end
    end
else
    % Fine Zoom Mapping
    if is_dc_mode == 3
        if N_d_safe == 1
            [V_sub_fine, apr_offset] = max(RHS_belief_flat, [], 1);
            V_sub_fine = reshape(V_sub_fine, [1, 1, FLAT_STATES]);
            apr_offset = reshape(apr_offset, [1, 1, FLAT_STATES]);
        else
            RHS_for_d = reshape(RHS_belief_flat, [N_d_safe, num_choices_total, FLAT_STATES]);
            [V_sub_fine, apr_offset] = max(RHS_for_d, [], 2);
            V_sub_fine = reshape(V_sub_fine, [N_d_safe, 1, FLAT_STATES]);
            apr_offset = reshape(apr_offset, [N_d_safe, 1, FLAT_STATES]);
        end
        d_idx_local = repmat(reshape(1:N_d_safe, [N_d_safe, 1]), [1, FLAT_STATES]);
        V_j_max   = reshape(V_sub_fine,  [N_d_safe, N_states, N_ze_local]);
        Pol_d_max = reshape(d_idx_local, [N_d_safe, N_states, N_ze_local]);

        apr_offset_2d = reshape(apr_offset, [N_d_safe, N_a1_other, FLAT_STATES]);
        a1_apr_offset = mod(apr_offset_2d - 1, num_choices_total_a1) + 1;
        a2_offset_factor = ceil(apr_offset_2d / num_choices_total_a1);
        loweredge_matrix_2d = reshape(loweredge_matrix_bounds, [N_d_safe, N_a1_other, FLAT_STATES]);

        d_vec_row = (1:N_d_safe)'; s_vec = shiftdim((0:FLAT_STATES-1) * (N_d_safe * N_a1_other), -1);
        lin_idx_loweredge = d_vec_row + (a2_offset_factor - 1) * N_d_safe + s_vec;
        chosen_loweredge = loweredge_matrix_2d(lin_idx_loweredge);

        a1_Pol_apr = min(chosen_loweredge + a1_apr_offset - 1, N_a1_dc);
        Pol_apr_max = a1_Pol_apr + (a2_offset_factor - 1) * N_a1_dc;
        Pol_apr_max = reshape(Pol_apr_max, [N_d_safe, N_states, N_ze_local]);
    else
        [V_sub_fine, Pol_sub_idx] = max(RHS_belief_flat, [], 1);
        num_choices_a1 = num_choices_total / N_a1_other;
        RHS_for_d = reshape(RHS_belief_flat, [N_d_safe, num_choices_a1, N_a1_other, FLAT_STATES]);
        [~, max_a1_idx_rel] = max(RHS_for_d, [], 2);

        if gridinterplayer(1) == 0 || is_dc_mode == 2
            max_a1_idx_rel = reshape(max_a1_idx_rel, [N_d_safe, N_a1_other, N_states, N_ze_local]);
            low_mat_4d = reshape(loweredge_matrix_bounds, [N_d_safe, N_a1_other, N_states, N_ze_local]);
            Pol_a1_per_a2 = min(low_mat_4d + max_a1_idx_rel - 1, N_a1_dc);
        end

        d_idx_local = mod(Pol_sub_idx - 1, N_d_safe) + 1;
        apr_offset  = ceil(Pol_sub_idx / N_d_safe);

        V_j_max   = reshape(V_sub_fine,  [N_states, N_ze_local]);
        Pol_d_max = reshape(d_idx_local, [N_states, N_ze_local]);

        a1_apr_offset = mod(apr_offset(:) - 1, num_choices_total_a1) + 1;
        a2_offset_factor = ceil(apr_offset(:) / num_choices_total_a1);
        chosen_offset = start_offset + a1_apr_offset(:) - 1;

        loweredge_matrix_2d = reshape(loweredge_matrix_bounds, [N_d_safe, N_a1_other, FLAT_STATES]);
        lin_idx_loweredge = d_idx_local(:) + (a2_offset_factor(:) - 1) * N_d_safe + (0:FLAT_STATES-1)' * (N_d_safe * N_a1_other);
        chosen_loweredge = loweredge_matrix_2d(lin_idx_loweredge);

        abs_fine_idx_flat = (chosen_loweredge(:) - 1) * (n2short + 1) + 1 + chosen_offset(:);
        a1_Pol_apr = floor((abs_fine_idx_flat(:) - 1) / (n2short + 1)) + 1;
        a1_Pol_apr = min(a1_Pol_apr, N_a1_dc - 1);
        Pol_L2idx_max = abs_fine_idx_flat(:) - (a1_Pol_apr(:) - 1) * (n2short + 1);

        Pol_apr_max = a1_Pol_apr(:) + (a2_offset_factor(:) - 1) * N_a1_dc;
        Pol_apr_max = reshape(Pol_apr_max, [N_states, N_ze_local]);
        Pol_L2idx_max = reshape(Pol_L2idx_max, [N_states, N_ze_local]);

        lin_lower = d_idx_local(:) + (1 - 1) * N_d_safe + (a2_offset_factor(:) - 1) * (num_choices_total_a1 * N_d_safe) + (0:FLAT_STATES-1)' * size(RHS_belief_flat, 1);
        lin_upper = d_idx_local(:) + (num_choices_total_a1 - 1) * N_d_safe + (a2_offset_factor(:) - 1) * (num_choices_total_a1 * N_d_safe) + (0:FLAT_STATES-1)' * size(RHS_belief_flat, 1);

        isInfLower = (RHS_belief_flat(lin_lower) == -Inf);
        isInfUpper = (RHS_belief_flat(lin_upper) == -Inf);
        inLowerStrict = (a1_apr_offset(:) >= 2) & (a1_apr_offset(:) <= n2short + 1);
        inUpperStrict = (a1_apr_offset(:) >= n2short + 3 + d_gap * (n2short + 1)) & (a1_apr_offset(:) <= num_choices_total_a1 - 1);

        Pol_L2flag_max = 2 * ones(1, FLAT_STATES, 'like', EV_local);
        Pol_L2flag_max(inLowerStrict & isInfLower) = 3;
        Pol_L2flag_max(inUpperStrict & isInfUpper) = 1;
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_states, N_ze_local]);

        if compute_valt
            lin_idx_valt = Pol_sub_idx + (0:FLAT_STATES-1) * (N_d_safe * num_choices_total);
            Valt_sub_fine = RHS_Valt_flat(lin_idx_valt);
            Valt_j_max = reshape(Valt_sub_fine, [N_states, N_ze_local]);
        end
    end
end


end