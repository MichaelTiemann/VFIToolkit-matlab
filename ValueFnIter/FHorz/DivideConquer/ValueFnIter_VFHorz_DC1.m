function varargout = ValueFnIter_VFHorz_DC1(n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% ValueFnIter_VFHorz_DC1 - Specialized Divide & Conquer High-Performance Engine
% Streamlined exclusively for DC models, combining flat-pack GPU vectorization
% with n-monotonicity slicing.

%% Standard Options & Validation Setup is performed by VFHorz orchestrator

N_d = prod(n_d); N_a = prod(n_a); N_z = prod(n_z);

if size(d_grid, 2) == 1
    d_gridvals = CreateGridvals(n_d, d_grid, 1);
else
    d_gridvals = d_grid;
end

% Default level1n configuration for DC -- done by orchestrator, but leave here for documentation
if ~isfield(vfoptions, 'level1n')
    if isscalar(n_a); vfoptions.level1n = floor(sqrt(n_a(1)));
    else; vfoptions.level1n = [floor(sqrt(n_a(1))), n_a(2:end)]; end
end

% done by orchestrator, but leave here for documentation
[z_gridvals_J, pi_z_J, vfoptions] = ExogShockSetup_FHorz(n_z, z_grid, pi_z, N_j, Parameters, vfoptions, 3, 0);
z_gridvals_J = gpuArray(z_gridvals_J);
pi_z_J = gpuArray(pi_z_J);

% --- MULTI-AXIS STATE PARSER ---
has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
n_e_pass = 0;
if has_e
    n_e_pass = vfoptions.n_e;
    if size(vfoptions.e_grid, 1) == sum(n_e_pass) && length(n_e_pass) > 1
        e_work = CreateGridvals(n_e_pass, vfoptions.e_grid, 1);
    else
        e_work = vfoptions.e_grid;
    end
else
    e_work = ones(1, 1, 'like', a_grid);
end

l_exp_base  = vfoptions.experienceasset >= 1;
l_exp_u     = vfoptions.experienceassetu >= 1;
l_exp_z     = vfoptions.experienceassetz >= 1;
l_exp_e     = vfoptions.experienceassete >= 1;
l_exp_ze    = vfoptions.experienceassetze >= 1;
l_exp_semiz = vfoptions.experienceassetsemiz >= 1;
is_exp_asset = l_exp_base || l_exp_u || l_exp_z || l_exp_e || l_exp_ze || l_exp_semiz;

if is_exp_asset || vfoptions.riskyasset == 1 || vfoptions.residualasset == 1
    a1_endo_grid_vals = vfoptions.a1_grid;
    a2_exp_grid_vals  = vfoptions.a2_grid;
    n_a1_dc = vfoptions.n_a1(1);
    l_a1 = length(vfoptions.n_a1);
    if l_a1 > 1; n_a1_other = vfoptions.n_a1(2:end); else; n_a1_other = []; end
    n_a2 = vfoptions.n_a2; l_a2 = length(n_a2);
else
    a1_endo_grid_vals = a_grid;
    a2_exp_grid_vals  = [];
    n_a1_dc = n_a(1);
    l_a1 = length(n_a);
    if l_a1 > 1; n_a1_other = n_a(2:end); else; n_a1_other = []; end
    n_a2 = []; l_a2 = 0;
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

[TensorReturnFn, ~, ~, ~, ~] = CreateTensorFnAndCells(ReturnFn, [n_d, n_a(1:l_a1)], n_a, n_z, n_e_pass, [], [], [], []);
[~, D_cells_block, A1_cells, Z_cells_block, E_cells_block] = CreateTensorFnAndCells(ReturnFn, n_d, n_a(1:l_a1), n_z, n_e_pass, d_grid, a1_endo_grid_vals, [], []);

if l_a2 > 0
    [~, ~, A2_cells, ~, ~] = CreateTensorFnAndCells(vfoptions.aprimeFn, n_d, n_a2, 0, 0, [], a2_exp_grid_vals, [], []);
else
    A2_cells = {};
end

A1_mat = zeros(N_a1_dc * N_a1_other, l_a1, 'like', a_grid);
for i_a = 1:l_a1; A1_mat(:, i_a) = A1_cells{i_a}(:); end

A2_mat = zeros(N_a2, l_a2, 'like', a_grid);
a2_grids_1d = cell(1, l_a2);
offset = 0;
for i_a = 1:l_a2
    A2_mat(:, i_a) = A2_cells{i_a}(:);
    a2_grids_1d{i_a} = a2_exp_grid_vals((offset + 1):(offset + n_a2(i_a)));
    offset = offset + n_a2(i_a);
end

for i_d = 1:length(D_cells_block)
    D_cells_block{i_d} = reshape(D_cells_block{i_d}, [max(1, prod(n_d)), 1, 1, 1, 1]);
end

N_d_safe = max(1, N_d);

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
has_z = prod(n_z) > 0; n_z_work = N_semiz * N_z_exog; n_e_work = max(1, prod(n_e_pass));
N_ze = n_z_work * n_e_work;

V = zeros(N_a, N_ze, N_j, 'gpuArray');
if N_d > 0
    Policy = zeros(2, N_a, N_ze, N_j, 'gpuArray');
else
    Policy = zeros(N_a, N_ze, N_j, 'gpuArray');
end
V_next = zeros(N_a, N_ze, 'gpuArray');

base_ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);
is_age_dependent = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j; is_age_dependent(ip) = true; end
    if ~isa(base_ReturnFnParamsCell{ip}, 'gpuArray'); base_ReturnFnParamsCell{ip} = gpuArray(base_ReturnFnParamsCell{ip}); end
end

N_semiz_local = 1; N_dsemiz = 1;
if has_semiz && length(n_d) > 0
    N_semiz_local = max(1, prod(vfoptions.n_semiz));
    if isfield(vfoptions, 'l_dsemiz'); N_dsemiz = prod(n_d(end-vfoptions.l_dsemiz+1:end)); else; N_dsemiz = n_d(end); end
end
N_z_exog = max(1, n_z_work / N_semiz_local);

% Pre-compute z and e values for calls to TensorFn and aprimeFn
ze_chunks = {1:N_ze};
chunk_meta = cell(1, length(ze_chunks));
d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]);

for i_ze = 1:length(ze_chunks)
    c_ze = ze_chunks{i_ze};
    if isa(c_ze, 'gpuArray'), c_ze_cpu = gather(c_ze); else, c_ze_cpu = c_ze; end
    [z_ind, e_ind] = ind2sub([n_z_work, n_e_work], c_ze_cpu);
    meta.z_vals = gpuArray(unique(z_ind));
    meta.e_vals = gpuArray(unique(e_ind));
    meta.n_z_loc = length(meta.z_vals);
    meta.n_e_loc = length(meta.e_vals);
    meta.N_ze_local = length(c_ze);

    % CORRECTED STATIC OFFSET:
    % The offset stride must span the local chunk size (meta.N_ze_local),
    % not the global n_z_work, to keep lin_idx_compact inside EV_local bounds.
    N_a_local = N_a1_dc * N_a1_other;
    z_vec = reshape((0:meta.n_z_loc-1) * N_a_local, [1, 1, 1, meta.n_z_loc, 1]);
    e_vec = reshape((0:meta.n_e_loc-1) * (N_a_local * n_z_work), [1, 1, 1, 1, meta.n_e_loc]);

    meta.static_EV_offset = [];
    meta.static_EV_offset_fine = [];

    chunk_meta{i_ze} = meta;
end

%% Finite Horizon Backward Induction Loop
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose == 1; fprintf('Finite horizon: %i of %i \n', jj, N_j); end
    ReturnFnParamsCell = base_ReturnFnParamsCell;
    for ip = find(is_age_dependent)
        ReturnFnParamsCell{ip} = cast(Parameters.(ReturnFnParamNames{ip})(jj), 'like', a_grid);
    end
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision);
    beta_j = prod(DiscountFactorParamsVec);
    if is_exp_asset; aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj); else; aprimeFnParamsCell = {}; end

    % All copied across from ValueFnIter_Case1_VHorz...setting up EV_flat_ze and others
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

        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
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

    for i_ze = 1:length(ze_chunks)
        meta = chunk_meta{i_ze};
        n_z_loc = meta.n_z_loc;
        n_e_loc = meta.n_e_loc;
        curr_ze = ze_chunks{i_ze};
        N_ze_local = length(curr_ze);
        EV_local = EV_flat_ze(:, curr_ze, :);

        % Extract precomputed static offset for this chunk
        static_EV_offset = meta.static_EV_offset;
        static_EV_offset_fine = meta.static_EV_offset_fine;

        % Build Z and E cells locally per chunk
        if has_semiz || has_z
            num_z_vars = length(n_all_z);
            Z_cells_local = cell(1, num_z_vars);
            if size(z_gridvals_J, 2) ~= num_z_vars
                z_inflated = reshape(z_gridvals_J, [N_z, num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
            else
                for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
            end
        else; Z_cells_local = {}; end
        if has_e
            num_e_vars = size(e_work, 2);
            E_cells_local = cell(1, num_e_vars);
            for ie_var = 1:num_e_vars; E_cells_local{ie_var} = reshape(e_work(meta.e_vals, ie_var), [1, 1, 1, 1, n_e_loc]); end
        else; E_cells_local = {}; end

        if ~is_exp_asset
            EV_reshaped = reshape(EV_local, [N_a1_dc * N_a1_other, n_z_loc, n_e_loc, N_dsemiz]);
            EV_d_sliced = EV_reshaped(:, :, :, dsemiz_idx_tensor(:));
            EV_bounded_pre = beta_j .* permute(EV_d_sliced, [4, 1, 5, 2, 3]);
        else
            % (Keep your Experience Asset EV prep here)
            error("exp asset not implemented yet")
        end

        % Ensure EV_local has the correct shape for stride indexing
        if N_a2 > 1
            EV_local = reshape(EV_local, [N_a1_dc * N_a1_other, N_a2, n_z_loc, n_e_loc]);
        else
            EV_local = reshape(EV_local, [N_a1_dc * N_a1_other, n_z_loc, n_e_loc]);
        end

        % Define Evaluation Block for DC Slicer using Meta-Trick cells
        LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_DC_TensorBlock(...
            state_idx, loweredge_matrix, maxgap_scalar, N_a1_dc, N_a1_other, max(1, N_a2), N_d_safe, N_ze_local, ...
            Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, ...
            EV_local, static_EV_offset, TensorReturnFn, ReturnFnParamsCell, n_z_loc, n_e_loc);

        % Define Evaluation Block for DC Slicer
        % LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar, d_gap, dc_mode_override) Evaluate_Case1_TensorBlock(...
        %     state_idx, loweredge_matrix, maxgap_scalar, d_gap, N_a1_dc, N_a1_other, max(1, N_a2), N_d_safe, N_ze_local, ...
        %     Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, a2_grids_1d, ...
        %     vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
        %     TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
        %     TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, dc_mode_override, 0, static_EV_offset_fine);

        % Dispatch Universal DC1 Slicer [ValueFnIter_DC1_Slicer.m](https://github.com/MichaelTiemann/VFIToolkit-matlab/raw/refs/heads/tensor-branch2/ValueFnIter/FHorz/DivideConquer/ValueFnIter_DC1_Slicer.m)
        if l_a1 == 1
            [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(N_a1_dc, N_a1_dc, max(1, N_a2), N_ze_local, vfoptions, LocalBlockFn_Base, N_d_safe);
        else
            % CRITICAL FIX: Pass N_a1_other * max(1, N_a2) as N_other_states so V_max allocates correctly
            [v, p_apr, p_d, p_l2idx, p_l2flag] = ValueFnIter_DC2A_Slicer(N_a1_dc, N_a1_other, N_a1_other * N_a2, N_d_safe, N_ze_local, vfoptions, LocalBlockFn_Base);
        end

        V_j_max(:, curr_ze)     = reshape(v,     [N_a, N_ze_local]);
        Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a, N_ze_local]);
        Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a, N_ze_local]);
        if vfoptions.gridinterplayer(1) == 1
            Pol_L2idx_max(:, curr_ze)  = reshape(p_l2idx,  [N_a, N_ze_local]);
            Pol_L2flag_max(:, curr_ze) = reshape(p_l2flag, [N_a, N_ze_local]);
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


function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx, Pol_L2flag] = Evaluate_DC_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a1_dc, N_a1_other, N_a2, N_d_safe, N_z, ...
    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, ...
    EV_local, static_EV_offset, TensorReturnFn, ReturnFnParamsCell, n_z_loc, n_e_loc)
% Evaluate_DC_TensorBlock - Pure Flat-Pack Tensor Core using Meta-Trick Cells

N_states = length(state_idx);
state_idx = cast(state_idx, 'like', EV_local);
if ~isempty(loweredge_matrix); loweredge_matrix = cast(loweredge_matrix, 'like', EV_local); end
l_a1 = length(A1_grids_1d);
l_a2 = size(A2_mat, 2);

if N_a2 > 1
    N_a1_total = N_a1_dc * N_a1_other;
    a2_sub = ceil(state_idx / N_a1_total);
    a1_sub = state_idx - (a2_sub - 1) * N_a1_total;
else
    a1_sub = state_idx;
    a2_sub = [];
end

A1_cells = cell(1, l_a1);
for ia = 1:l_a1
    A1_cells{ia} = cast(reshape(A1_mat(a1_sub, ia), [1, 1, N_states, 1, 1]), 'like', EV_local);
end

if N_a2 > 1
    A2_cells = cell(1, l_a2);
    for ia = 1:l_a2
        A2_cells{ia} = cast(reshape(A2_mat(a2_sub, ia), [1, 1, N_states, 1, 1]), 'like', EV_local);
    end
else
    A2_cells = {};
end

local_ze = n_z_loc * n_e_loc;

if isempty(loweredge_matrix)
    grids_for_choices = A1_grids_1d;
    [mesh_out{1:l_a1}] = ndgrid(grids_for_choices{:});
    num_choices_total = numel(mesh_out{1});

    Apr_cells = cell(1, l_a1);
    for ia = 1:l_a1
        Apr_cells{ia} = cast(reshape(mesh_out{ia}(:), [1, num_choices_total, 1, 1, 1]), 'like', EV_local);
    end

    if N_a2 > 1
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_local{:}, E_cells_local{:}, ReturnFnParamsCell{:});
    else
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_local{:}, E_cells_local{:}, ReturnFnParamsCell{:});
    end

    FLAT_CHOICES = N_d_safe * num_choices_total;
    F_tensor = reshape(F_tensor, [FLAT_CHOICES, N_states, local_ze]);

    % Generate choice indices across choices and local shocks
    choice_idx_linear = reshape(1:num_choices_total, [1, num_choices_total, 1, 1, 1]);
    N_a_total = N_a1_dc * N_a1_other;

    a1_indices = reshape(choice_idx_linear, [1, num_choices_total, 1, 1, 1]);
    a1_indices = max(1, min(a1_indices, N_a_total));

    ze_indices = reshape(1:local_ze, [1, 1, 1, local_ze, 1]);

    % Broadcast indices to match [N_d_safe, num_choices_total, N_states, local_ze]
    % Note: FLAT_CHOICES = N_d_safe * num_choices_total
    d_idx = reshape(1:N_d_safe, [N_d_safe, 1, 1, 1, 1]);

    % Compute linear indices matching EV_local [N_a, local_ze]
    % Row = a1_indices, Col = ze_indices
    lin_idx_compact = a1_indices + (ze_indices - 1) * N_a_total;

    EV_bounded_raw = EV_local(lin_idx_compact); % Slices based on a1 and ze
    % Replicate/Expand across decision choices (N_d_safe) and states (N_states)
    EV_bounded = repmat(EV_bounded_raw, [N_d_safe, 1, N_states, 1]);
    EV_bounded = reshape(EV_bounded, [FLAT_CHOICES, N_states, local_ze]);
else
    total_gap = maxgap_scalar;
    if l_a1 == 1
        base_idx_a1 = reshape(loweredge_matrix, [N_d_safe, 1, N_states, n_z_loc, n_e_loc]);
        offsets_a1 = reshape(0:total_gap, [1, total_gap + 1, 1, 1, 1]);
        choice_idx_a1_base = max(1, min(base_idx_a1 + offsets_a1, length(A1_grids_1d{1})));
        Apr_cells = { A1_grids_1d{1}(choice_idx_a1_base) };
        choice_idx_linear = choice_idx_a1_base;
        num_choices_total = total_gap + 1;
    else
        num_choices_total = (total_gap + 1) * N_a1_other;
        base_idx_a1 = reshape(loweredge_matrix, [N_d_safe, N_a1_other, N_states, n_z_loc, n_e_loc]);
        offsets_a1 = reshape(0:total_gap, [1, 1, 1, 1, 1, total_gap + 1]);
        choice_idx_a1_matrix = max(1, min(base_idx_a1 + offsets_a1, length(A1_grids_1d{1})));
        choice_idx_a1_matrix = permute(choice_idx_a1_matrix, [1, 6, 2, 3, 4, 5]);
        choice_idx_a1 = reshape(choice_idx_a1_matrix, [N_d_safe, num_choices_total, N_states, n_z_loc, n_e_loc]);

        a2_base_vec = reshape(1:N_a1_other, [1, 1, N_a1_other]);
        a2_mesh = repmat(a2_base_vec, [N_d_safe, total_gap + 1, 1]);
        choice_idx_a2 = cast(reshape(a2_mesh, [N_d_safe, num_choices_total, 1, 1, 1]), 'like', choice_idx_a1);

        Apr_cells = cell(1, l_a1);
        Apr_cells{1} = A1_grids_1d{1}(choice_idx_a1);
        if l_a1 > 2; [mesh_a2{1:l_a1-1}] = ndgrid(A1_grids_1d{2:end}); else; mesh_a2{1} = A1_grids_1d{2}; end
        for ia = 2:l_a1
            flat_grid = mesh_a2{ia-1}(:);
            Apr_cells{ia} = reshape(flat_grid(choice_idx_a2), [N_d_safe, num_choices_total, 1, 1, 1]);
        end
        choice_idx_linear = choice_idx_a1 + (choice_idx_a2 - 1) * length(A1_grids_1d{1});
    end

    if N_a2 > 1
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_local{:}, E_cells_local{:}, ReturnFnParamsCell{:});
    else
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_local{:}, E_cells_local{:}, ReturnFnParamsCell{:});
    end

    FLAT_CHOICES = N_d_safe * num_choices_total;
    F_tensor = reshape(F_tensor, [FLAT_CHOICES, N_states, local_ze]);

    N_a_total = N_a1_dc * N_a1_other;
    a1_indices = max(1, min(choice_idx_linear, N_a_total));

    % Decouple shock indices into separate z and e dimensions for multi-axis compatibility
    z_indices = reshape(1:n_z_loc, [1, 1, 1, n_z_loc, 1]);
    e_indices = reshape(1:n_e_loc, [1, 1, 1, 1, n_e_loc]);

    % Compute multi-axis linear offset matching EV_local layout
    ze_flat_idx = z_indices + (e_indices - 1) * n_z_loc;
    lin_idx_compact = a1_indices + (ze_flat_idx - 1) * N_a_total;

    EV_bounded = EV_local(lin_idx_compact);
    EV_bounded = reshape(EV_bounded, [FLAT_CHOICES, N_states, local_ze]);
end

FLAT_CHOICES = N_d_safe * num_choices_total;
FLAT_STATES = N_states * local_ze;

RHS = F_tensor + EV_bounded;
RHS_flat = reshape(RHS, [FLAT_CHOICES, FLAT_STATES]);

if N_d_safe == 1
    [V_sub, apr_idx_local] = max(RHS_flat, [], 1);
    V_sub = reshape(V_sub, [1, 1, FLAT_STATES]);
    apr_idx_local = reshape(apr_idx_local, [1, 1, FLAT_STATES]);
else
    RHS_for_d = reshape(RHS_flat, [N_d_safe, num_choices_total, FLAT_STATES]);
    RHS_perm = permute(RHS_for_d, [2, 1, 3]);
    [V_sub, apr_idx_local] = max(RHS_perm, [], 1);
    V_sub = permute(V_sub, [2, 1, 3]);
    apr_idx_local = permute(apr_idx_local, [2, 1, 3]);
end

d_idx_local = repmat(reshape(1:N_d_safe, [N_d_safe, 1]), [1, FLAT_STATES]);

V_j_max   = reshape(V_sub,       [N_d_safe, N_states, local_ze]);
Pol_d_max = reshape(d_idx_local, [N_d_safe, N_states, local_ze]);

if isempty(loweredge_matrix)
    Pol_apr_max = reshape(apr_idx_local, [N_d_safe, N_states, local_ze]);
else
    % Force apr_idx_local to flatten to [N_d_safe, FLAT_STATES] to match loweredge_2d
    apr_idx_flat = reshape(apr_idx_local, [N_d_safe, FLAT_STATES]);

    a1_apr_offset = mod(apr_idx_flat - 1, total_gap + 1) + 1;
    a2_offset_factor = ceil(apr_idx_flat / (total_gap + 1));

    loweredge_2d = reshape(loweredge_matrix, [N_d_safe, FLAT_STATES]);
    d_vec = cast((1:N_d_safe)', 'like', apr_idx_flat);
    s_vec = cast((0:FLAT_STATES-1) * N_d_safe, 'like', apr_idx_flat);

    lin_idx_low = d_vec + s_vec;
    chosen_low = loweredge_2d(lin_idx_low);

    a1_Pol = min(chosen_low + a1_apr_offset - 1, N_a1_dc);
    Pol_apr_max = a1_Pol + (a2_offset_factor - 1) * N_a1_dc;

    % Final safe reshape matching [N_d_safe, N_states, local_ze]
    Pol_apr_max = reshape(Pol_apr_max, [N_d_safe, N_states, local_ze]);
end

Pol_L2idx = zeros(N_d_safe, N_states, local_ze, 'like', EV_local);
Pol_L2flag = 2 * ones(N_d_safe, N_states, local_ze, 'like', EV_local);


end