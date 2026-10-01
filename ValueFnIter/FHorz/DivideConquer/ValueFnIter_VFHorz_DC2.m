function varargout = ValueFnIter_VFHorz_DC2(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
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

% done by orchestrator, but leave here for documentation; we receive properly aged z and pi_z grids
% [z_gridvals_J, pi_z_J, vfoptions] = ExogShockSetup_FHorz(n_z, z_grid, pi_z, N_j, Parameters, vfoptions, 3, 0);
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

if vfoptions.riskyasset == 1 || vfoptions.residualasset == 1
    error("riskyasset and residualasset not supported in DC2")
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
[~, D_cells_block, A1_cells, ~, ~] = CreateTensorFnAndCells(ReturnFn, n_d, n_a(1:l_a1), n_z, n_e_pass, d_grid, a1_endo_grid_vals, [], []);

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
n_a_work = prod(n_a);

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

V = zeros(n_a_work, n_z_work, n_e_work, N_j, 'gpuArray'); V_next = zeros(n_a_work, n_z_work, n_e_work, 'gpuArray');
PolicyKron = zeros(N_a, n_z_work, n_e_work, N_j, 'gpuArray');

base_ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);
is_age_dependent = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j; is_age_dependent(ip) = true; end
    if ~isa(base_ReturnFnParamsCell{ip}, 'gpuArray'); base_ReturnFnParamsCell{ip} = gpuArray(base_ReturnFnParamsCell{ip}); end
end

warmglow = int32(isfield(vfoptions,'WarmGlowBequestsFn'));
ezc2 = ones(N_j,1); ezc3 = 1; ezc4 = 1; ezc5 = ones(N_j,1); ezc6 = ones(N_j,1); ezc7 = ones(N_j,1); ezc8 = ones(N_j,1);

N_semiz_local = 1; N_dsemiz = 1;
if has_semiz && ~isempty(n_d)
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

        EV_reshaped = reshape(EV_local, [N_a1_dc * N_a1_other, n_z_loc, n_e_loc, N_dsemiz]);
        EV_d_sliced = EV_reshaped(:, :, :, dsemiz_idx_tensor(:));
        EV_bounded_pre = beta_j .* permute(EV_d_sliced, [4, 1, 5, 2, 3]);

        % Define Evaluation Block for DC Slicer using Meta-Trick cells
        LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_DC2_TensorBlock(...
            state_idx, loweredge_matrix, maxgap_scalar, N_a1_dc, N_a1_other, max(1, N_a2), N_d_safe, N_ze_local, ...
            Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, ...
            EV_bounded_pre, TensorReturnFn, ReturnFnParamsCell, n_z_loc, n_e_loc);

        % Define Evaluation Block for DC Slicer
        % LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar, d_gap, dc_mode_override) Evaluate_Case1_TensorBlock(...
        %     state_idx, loweredge_matrix, maxgap_scalar, d_gap, N_a1_dc, N_a1_other, max(1, N_a2), N_d_safe, N_ze_local, ...
        %     Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, a2_grids_1d, ...
        %     vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
        %     TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
        %     TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, dc_mode_override, 0, static_EV_offset_fine);

        % DC Mode 3 requests [N_d, N_states, N_ze] from the Tensor Block
        if l_a1 == 1
            error("must call DC2 with 2-Asset grid")
        else
            N_other_states = N_a1_other * max(1, N_a2);
            [v, p_apr, p_d] = ValueFnIter_DC2_Slicer(N_a1_dc, N_a1_other, N_other_states, N_ze_local, vfoptions, LocalBlockFn, N_d_safe);
        end

        V_j_max(:, curr_ze)     = reshape(v,     [N_a, N_ze_local]);
        Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a, N_ze_local]);
        Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a, N_ze_local]);
    end
    V_j_max     = reshape(V_j_max,     [N_a, n_z_work, n_e_work]);
    Pol_apr_max = reshape(Pol_apr_max, [N_a, n_z_work, n_e_work]);
    Pol_d_max   = reshape(Pol_d_max,   [N_a, n_z_work, n_e_work]);
    if N_d > 0
        PolicyKron(:, :, :, jj) = (Pol_apr_max - 1) * N_d + Pol_d_max;
    else
        PolicyKron(:, :, :, jj) = Pol_apr_max;
    end

    V(:, :, :, jj) = V_j_max;
    V_next = V_j_max;
end

if N_z == 0; V = squeeze(V); end
if N_d == 0; n_daprime = n_a(1:l_a1); else; n_daprime = [n_d, n_a(1:l_a1)]; end
% No gridinterplayer, so need to make room for default 1 dimension
PolicyKron = shiftdim(PolicyKron, -1);

if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    varargout{1} = V; varargout{2} = PolicyKron; return
end

disp('Unpacking Policy tensor to System RAM...');
num_pol_vars = length(n_daprime); n_daprime_col = n_daprime(:); divisors = cumprod([1; n_daprime_col(1:end-1)]);
MAX_INT32 = 2147483647;

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


function [V_j_max, Pol_apr_max, Pol_d_max, Pol_a1_per_a2] = Evaluate_DC2_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a1_dc, N_a1_other, N_a2, N_d_safe, N_ze_local, ...
    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, ...
    EV_bounded_pre, TensorReturnFn, ReturnFnParamsCell, n_z_loc, n_e_loc)

N_states = length(state_idx);
state_idx = cast(state_idx, 'like', EV_bounded_pre);
if ~isempty(loweredge_matrix); loweredge_matrix = cast(loweredge_matrix, 'like', EV_bounded_pre); end
is_coarse = isempty(loweredge_matrix);

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

% =========================================================================
% --- ORTHOGONAL 3D FLAT-PACK GEOMETRY (D, A1_prime, A2_prime) ---
% =========================================================================
if is_coarse; num_a1_choices = N_a1_dc;
else;         num_a1_choices = maxgap_scalar + 1; end

num_a2_choices = N_a1_other;
num_choices_total = num_a1_choices * num_a2_choices;

if N_d_safe > 1
    dim_D = 1; dim_C1 = 2; dim_C2 = 3; dim_S = 4; dim_Z = 5; dim_E = 6;
    d_shape = [N_d_safe, 1, 1, 1, 1, 1];
else
    dim_C1 = 1; dim_C2 = 2; dim_S = 3; dim_Z = 4; dim_E = 5;
    d_shape = [1, 1];
end

c1_shape = ones(1, 6); c1_shape(dim_C1) = num_a1_choices;
c2_shape = ones(1, 6); c2_shape(dim_C2) = num_a2_choices;
s_shape  = ones(1, 6); s_shape(dim_S)  = N_states;
z_shape  = ones(1, 6); z_shape(dim_Z)  = n_z_loc;
e_shape  = ones(1, 6); e_shape(dim_E)  = n_e_loc;

% --- BUILD PURELY ORTHOGONAL STATE CELLS ---
A1_cells = cell(1, l_a1);
for ia = 1:l_a1; A1_cells{ia} = reshape(A1_mat(a1_sub, ia), s_shape); end
if N_a2 > 1
    A2_cells = cell(1, l_a2);
    for ia = 1:l_a2; A2_cells{ia} = reshape(A2_mat(a2_sub, ia), s_shape); end
else; A2_cells = {}; end

Z_cells_eval = cell(1, length(Z_cells_local));
for iz = 1:length(Z_cells_local); Z_cells_eval{iz} = reshape(Z_cells_local{iz}(:), z_shape); end
E_cells_eval = cell(1, length(E_cells_local));
for ie = 1:length(E_cells_local); E_cells_eval{ie} = reshape(E_cells_local{ie}(:), e_shape); end
for i_d = 1:length(D_cells_block); D_cells_block{i_d} = reshape(D_cells_block{i_d}(1:N_d_safe), d_shape); end

% --- PHASE 1: EVALUATION ---
Apr_cells = cell(1, l_a1);
choice_idx_a2 = gpuArray(reshape(1:num_a2_choices, c2_shape));

if is_coarse
    choice_idx_a1 = gpuArray(reshape(1:num_a1_choices, c1_shape));
    Apr_cells{1} = cast(A1_grids_1d{1}(choice_idx_a1), 'like', EV_bounded_pre);
else
    % We reshape the conditional loweredge natively into Orthogonal Dim 3 (a2prime)
    if N_d_safe > 1; low_shape = [N_d_safe, 1, num_a2_choices, N_states, n_z_loc, n_e_loc];
    else;            low_shape = [1, num_a2_choices, N_states, n_z_loc, n_e_loc, 1]; end

    base_idx_a1 = reshape(loweredge_matrix, low_shape);
    offsets_a1 = gpuArray(reshape(0:maxgap_scalar, c1_shape));

    choice_idx_a1 = base_idx_a1 + offsets_a1;
    choice_idx_a1 = max(1, min(choice_idx_a1, length(A1_grids_1d{1})));
    Apr_cells{1} = cast(A1_grids_1d{1}(choice_idx_a1), 'like', EV_bounded_pre);
end

if l_a1 > 1
    if l_a1 > 2; [mesh_a2{1:l_a1-1}] = ndgrid(A1_grids_1d{2:end}); else; mesh_a2{1} = A1_grids_1d{2}; end
    for ia = 2:l_a1
        flat_grid = mesh_a2{ia-1}(:);
        Apr_cells{ia} = cast(flat_grid(choice_idx_a2), 'like', EV_bounded_pre);
    end
end

DAprime_cells = [D_cells_block, Apr_cells];

if N_a2 > 1
    F_tensor = TensorReturnFn(DAprime_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_eval{:}, E_cells_eval{:}, ReturnFnParamsCell{:});
else
    F_tensor = TensorReturnFn(DAprime_cells{:}, A1_cells{:}, Z_cells_eval{:}, E_cells_eval{:}, ReturnFnParamsCell{:});
end

% --- PHASE 2: UNIVERSAL RHS EVALUATION & EXPECTED VALUE RECONSTRUCTION ---
stride_C1 = N_d_safe;
stride_C2 = N_d_safe * N_a1_dc;
stride_Z  = N_d_safe * N_a1_dc * N_a1_other * max(1, N_a2);
stride_E  = stride_Z * n_z_loc;

d_offset  = gpuArray(reshape(0:N_d_safe-1, d_shape));
c1_offset = (choice_idx_a1 - 1) * stride_C1;
c2_offset = (choice_idx_a2 - 1) * stride_C2;
c_offset  = c1_offset + c2_offset;

z_offset = gpuArray(reshape(0:n_z_loc-1, z_shape) * stride_Z);
e_offset = gpuArray(reshape(0:n_e_loc-1, e_shape) * stride_E);

lin_idx_compact = 1 + d_offset + c_offset + z_offset + e_offset;
EV_bounded = EV_bounded_pre(lin_idx_compact);

F_tensor = F_tensor + EV_bounded;

FLAT_CHOICES = N_d_safe * num_choices_total;
FLAT_STATES  = N_states * n_z_loc * n_e_loc;

% --- PHASE 3: OUTPUT MAPPING ---
if is_coarse
    RHS_4D = reshape(F_tensor, [N_d_safe, num_a1_choices, num_a2_choices, FLAT_STATES]);
    [~, Pol_a1_per_a2] = max(RHS_4D, [], 2);
    Pol_a1_per_a2 = reshape(Pol_a1_per_a2, [N_d_safe, num_a2_choices, FLAT_STATES]);

    RHS_flat = reshape(RHS_4D, [FLAT_CHOICES, FLAT_STATES]);
    [V_j_max, Pol_idx] = max(RHS_flat, [], 1);

    Pol_d_max = mod(Pol_idx - 1, N_d_safe) + 1;
    C1C2_idx = floor((Pol_idx - 1) / N_d_safe);
    Pol_a1_idx = mod(C1C2_idx, num_a1_choices) + 1;
    Pol_a2_idx = floor(C1C2_idx / num_a1_choices) + 1;
    Pol_apr_max = Pol_a1_idx + (Pol_a2_idx - 1) * N_a1_dc;
else
    Pol_a1_per_a2 = [];

    RHS_flat = reshape(F_tensor, [FLAT_CHOICES, FLAT_STATES]);
    [V_j_max, Pol_idx] = max(RHS_flat, [], 1);

    Pol_d_max = mod(Pol_idx - 1, N_d_safe) + 1;
    C1C2_idx = floor((Pol_idx - 1) / N_d_safe);

    c1_local = mod(C1C2_idx, num_a1_choices);
    c2_local = floor(C1C2_idx / num_a1_choices);

    low_flat = reshape(loweredge_matrix, [N_d_safe, num_a2_choices, FLAT_STATES]);
    flat_state_idx = 1:FLAT_STATES;
    lin_low = Pol_d_max + c2_local * N_d_safe + (flat_state_idx - 1) * (N_d_safe * num_a2_choices);

    base_vals = low_flat(lin_low);
    Pol_a1_idx = base_vals + c1_local;

    Pol_a2_idx = c2_local + 1;
    Pol_apr_max = Pol_a1_idx + (Pol_a2_idx - 1) * N_a1_dc;
end


end
