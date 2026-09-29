function [V, Policy] = ValueFnIter_VFHorz_RiskyAsset_EpsteinZin(...
    n_d, n_a1, n_a2, n_z, n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ...
    ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ~, ~, vfoptions, ...
    sj, warmglow, ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8)

% =========================================================================
% PHASE 1: PRE-COMPUTATION & AUTONOMOUS PARSING
% =========================================================================
% Establish a bulletproof prototype for instantiating new arrays (matches precision & device)
if ~isempty(a2_grid); proto = a2_grid; elseif ~isempty(a1_grid); proto = a1_grid; else; proto = d_grid; end

if isfield(vfoptions, 'ezc9'); ezc9 = vfoptions.ezc9; else; ezc9 = ones(N_j, 1); end

n_a = [n_a1, n_a2];
N_d_safe = max(1, prod(n_d));
N_a1_safe = max(1, prod(n_a1));
N_a2_safe = max(1, prod(n_a2));
N_z = max(1, prod(n_z));
N_u = max(1, prod(n_u));
N_a = N_a1_safe * N_a2_safe;

pi_u_shape = reshape(pi_u(:), [1, 1, 1, 1, 1, 1, N_u]);

if isfield(vfoptions, 'refine_d')
    l_d1 = vfoptions.refine_d(1); l_d2 = vfoptions.refine_d(2);
else
    l_d1 = 0; l_d2 = 0;
end
d_return_idx = [1:l_d1, (l_d1+l_d2+1):length(n_d)];
d_aprime_idx = (l_d1+1):length(n_d);

temp_ret = getAnonymousFnInputNames(ReturnFn);
first_param_idx = find(isfield(Parameters, temp_ret), 1, 'first');
if isempty(first_param_idx); ReturnFnParamNames = {}; num_prefix_ret = length(temp_ret);
else; ReturnFnParamNames = temp_ret(first_param_idx:end); num_prefix_ret = first_param_idx - 1; end

temp_ap = getAnonymousFnInputNames(aprimeFn);
first_param_idx = find(isfield(Parameters, temp_ap), 1, 'first');
if isempty(first_param_idx); aprimeFnParamNames = {}; num_prefix_ap = length(temp_ap);
else; aprimeFnParamNames = temp_ap(first_param_idx:end); num_prefix_ap = first_param_idx - 1; end

d_grids = cell(1, length(n_d)); offset = 0;
for i = 1:length(n_d); d_grids{i} = d_grid(offset+1 : offset+n_d(i)); offset = offset + n_d(i); end
a1_grids = cell(1, length(n_a1)); offset = 0;
for i = 1:length(n_a1); a1_grids{i} = a1_grid(offset+1 : offset+n_a1(i)); offset = offset + n_a1(i); end
a2_grids = cell(1, max(1, length(n_a2))); offset = 0;
for i = 1:length(n_a2); a2_grids{i} = a2_grid(offset+1 : offset+n_a2(i)); offset = offset + n_a2(i); end

% Strict Orthogonal Dimensions 1-8 prevents VRAM blowouts
D_cells = cell(1, length(n_d));
for i = 1:length(n_d); shape = ones(1, 8); shape(i) = n_d(i); D_cells{i} = reshape(d_grids{i}, shape); end
dim_a1p = length(n_d) + 1; dim_a1s = length(n_d) + 2; dim_a2s = length(n_d) + 3; dim_z = length(n_d) + 4; dim_u = length(n_d) + 5;

Apr_cells = cell(1, length(n_a1));
for i = 1:length(n_a1); shape = ones(1, 8); shape(dim_a1p) = n_a1(i); Apr_cells{i} = reshape(a1_grids{i}, shape); end
A1_cells = cell(1, length(n_a1));
for i = 1:length(n_a1); shape = ones(1, 8); shape(dim_a1s) = n_a1(i); A1_cells{i} = reshape(a1_grids{i}, shape); end
A2_cells = cell(1, max(1, length(n_a2)));
for i = 1:length(n_a2); shape = ones(1, 8); shape(dim_a2s) = n_a2(i); A2_cells{i} = reshape(a2_grids{i}, shape); end
if isempty(n_a2); shape = ones(1, 8); shape(dim_a2s) = 1; A2_cells{1} = zeros(shape, 'like', proto); end

shape = ones(1, 8); shape(dim_u) = N_u;
U_cells = {reshape(u_grid, shape)};

ret_args = cell(1, num_prefix_ret); idx = 1;
for i = 1:length(d_return_idx); ret_args{idx} = D_cells{d_return_idx(i)}; idx = idx + 1; end
if num_prefix_ret >= idx; for i = 1:length(n_a1); ret_args{idx} = Apr_cells{i}; idx = idx + 1; end; end
if num_prefix_ret >= idx; for i = 1:length(n_a1); ret_args{idx} = A1_cells{i}; idx = idx + 1; end; end
if num_prefix_ret >= idx; for i = 1:length(n_a2); ret_args{idx} = A2_cells{i}; idx = idx + 1; end; end
num_z_vars = size(z_gridvals_J, 2);

ap_args = cell(1, num_prefix_ap); idx = 1;
for i = 1:length(d_aprime_idx); ap_args{idx} = D_cells{d_aprime_idx(i)}; idx = idx + 1; end
if num_prefix_ap >= idx + 1; for i = 1:length(n_a2); ap_args{idx} = A2_cells{i}; idx = idx + 1; end; end
ap_args{num_prefix_ap} = U_cells{1};

V_next = zeros(N_a, N_z, 'like', proto);
V_out_flat = zeros(N_a, N_z, N_j, 'like', proto);
Policy_out_flat = zeros(length(n_d) + length(n_a1), N_a, N_z, N_j, 'like', proto);

base_RetParams = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);
Ret_age = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j; Ret_age(ip) = true; end
    if vfoptions.parallel == 2 && isnumeric(base_RetParams{ip}) && ~isa(base_RetParams{ip}, 'gpuArray')
        if ~isscalar(base_RetParams{ip}); base_RetParams{ip} = gpuArray(base_RetParams{ip}); end
    end
end

base_ApParams = CreateCellFromParams(Parameters, aprimeFnParamNames, 1, vfoptions.precision);
Ap_age = false(1, length(aprimeFnParamNames));
for ip = 1:length(aprimeFnParamNames)
    if numel(Parameters.(aprimeFnParamNames{ip})) == N_j; Ap_age(ip) = true; end
    if vfoptions.parallel == 2 && isnumeric(base_ApParams{ip}) && ~isa(base_ApParams{ip}, 'gpuArray')
        if ~isscalar(base_ApParams{ip}); base_ApParams{ip} = gpuArray(base_ApParams{ip}); end
    end
end

% =========================================================================
% PHASE 2: REVERSE TIME LOOP
% =========================================================================
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose == 1; fprintf('RiskyAsset EZ Finite horizon: %i of %i \n', jj, N_j); end

    ReturnFnParamsCell = base_RetParams;
    for ip = find(Ret_age)
        val = Parameters.(ReturnFnParamNames{ip})(jj);
        if vfoptions.parallel == 2; ReturnFnParamsCell{ip} = gpuArray(val); else; ReturnFnParamsCell{ip} = val; end
    end
    aprimeFnParamsCell = base_ApParams;
    for ip = find(Ap_age)
        val = Parameters.(aprimeFnParamNames{ip})(jj);
        if vfoptions.parallel == 2; aprimeFnParamsCell{ip} = gpuArray(val); else; aprimeFnParamsCell{ip} = val; end
    end

    beta_j = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision));
    sj_val = sj(jj);
    pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));

    if isfield(vfoptions, 'EZoneminusbeta') && vfoptions.EZoneminusbeta == 1
        ezc1_j = 1 - beta_j;
    elseif isfield(vfoptions, 'EZoneminusbeta') && vfoptions.EZoneminusbeta == 2
        ezc1_j = 1 - sj_val * beta_j;
    else
        ezc1_j = 1;
    end

    ret_args_run = ret_args;
    idx = length(ret_args_run) - num_z_vars + 1;
    for iz = 1:num_z_vars
        z_shape = ones(1, 8); z_shape(dim_z) = N_z;
        ret_args_run{idx} = reshape(z_gridvals_J(:, iz, min(jj, size(z_gridvals_J,3))), z_shape);
        idx = idx + 1;
    end

    if jj == N_j
        EV_z = zeros(N_a1_safe, N_a2_safe, N_z, 'like', proto);
    else
        valid_V = isfinite(V_next) & (V_next ~= 0);
        V_transformed = V_next;
        ezc4_val = ezc4(min(jj, length(ezc4))); ezc5_val = ezc5(min(jj, length(ezc5)));

        if ezc5_val == 1; V_transformed(valid_V) = ezc4_val * V_next(valid_V); else; V_transformed(valid_V) = max(ezc4_val * V_next(valid_V), 0).^ezc5_val; end
        V_transformed(V_next == 0) = 0;

        V_slice = reshape(V_transformed, [N_a, N_z]);
        V_inf_mask = (V_slice == -Inf);
        V_safe = V_slice;
        V_safe(V_inf_mask) = -1e250;
        EV_z_flat = V_safe * pi_z_j';
        inf_restore = (V_inf_mask * (pi_z_j' > 0)) > 0;
        EV_z_flat(inf_restore) = -Inf;
        EV_z = reshape(EV_z_flat, [N_a1_safe, N_a2_safe, N_z]);
    end

    if warmglow == 1
        wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj, vfoptions.precision);
        WG_eval = vfoptions.WarmGlowBequestsFn(a2_grid, wg_params{:});
        if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a2_grid), 'like', proto); end
        valid_wg = isfinite(WG_eval) & (WG_eval ~= 0);
        WG_transformed = WG_eval;
        ezc4_val = ezc4(min(jj, length(ezc4))); ezc5_val = ezc5(min(jj, length(ezc5)));

        if ezc5_val == 1; WG_transformed(valid_wg) = ezc4_val * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4_val * WG_eval(valid_wg), 0).^ezc5_val; end
        WG_transformed(WG_eval == 0) = 0;
        WG_vec = WG_transformed(:); % Flatten to 1D vector (evaluated at nodes)
    else
        WG_vec = 0;
    end

    % =========================================================================
    % PHASE 3: TENSOR EVALUATION & IMPLICIT EXPANSION
    % =========================================================================
    % 1. Evaluate Return Function (Dynamically bound to active variables)
    ret_shape = ones(1, 8);
    for i = 1:length(ret_args_run)
        sz = size(ret_args_run{i});
        ret_shape(1:length(sz)) = max(ret_shape(1:length(sz)), sz);
    end

    ret_args_exp = cell(size(ret_args_run));
    for i = 1:length(ret_args_run)
        ret_args_exp{i} = ret_args_run{i} + zeros(ret_shape, 'like', proto);
    end

    F_tensor = arrayfun(ReturnFn, ret_args_exp{:}, ReturnFnParamsCell{:});
    F_tensor(isfinite(F_tensor) & F_tensor ~= 0) = F_tensor(isfinite(F_tensor) & F_tensor ~= 0).^ezc2(min(jj, length(ezc2)));
    F_tensor(F_tensor == 0) = -Inf;

    % 2. Evaluate Portfolio Returns (a2_prime)
    % Hoist the portfolio mapping out of the loop if parameters are static!
    if reverse_j == 0 || any(Ap_age)
        ap_shape = ones(1, 8);
        for i = 1:length(ap_args)
            sz = size(ap_args{i});
            ap_shape(1:length(sz)) = max(ap_shape(1:length(sz)), sz);
        end

        ap_args_exp = cell(size(ap_args));
        for i = 1:length(ap_args)
            ap_args_exp{i} = ap_args{i} + zeros(ap_shape, 'like', proto);
        end

        A2_prime = arrayfun(aprimeFn, ap_args_exp{:}, aprimeFnParamsCell{:});
        a2_grid_1d_vec = a2_grids{1};
        a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));

        a2_grid_shape = ones(1, 8); a2_grid_shape(8) = length(a2_grid_1d_vec);
        a2_grid_shape_vals = reshape(a2_grid_1d_vec, a2_grid_shape);
        idx = sum(a2_prime_clipped >= a2_grid_shape_vals, 8);
        idx(idx == 0) = 1;
        idx(idx == length(a2_grid_1d_vec)) = length(a2_grid_1d_vec) - 1;

        a2_left = reshape(a2_grid_1d_vec(idx), size(idx));
        a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
        weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
        weight(a2_right == a2_left) = 0;
        weight(abs(weight) < 1e-12) = 0;
        weight(abs(weight - 1) < 1e-12) = 1;

        % Strictly cast offset matrices to proto to prevent GPU/CPU indexing stalls
        a1_prime_offset_shape = ones(1, 8); a1_prime_offset_shape(dim_a1p) = N_a1_safe;
        a1_prime_offset = cast(reshape(0:N_a1_safe-1, a1_prime_offset_shape), 'like', proto);

        a2_prime_offset = (idx - 1) * N_a1_safe;

        z_offset_shape = ones(1, 8); z_offset_shape(dim_z) = N_z;
        z_offset = cast(reshape(0:N_z-1, z_offset_shape) * (N_a1_safe * N_a2_safe), 'like', proto);

        idx_left  = 1 + a1_prime_offset + a2_prime_offset + z_offset;
        idx_right = idx_left + N_a1_safe;
    end

    EV_flat = EV_z(:);
    term_L = EV_flat(idx_left) .* (1 - weight);
    term_R = EV_flat(idx_right) .* weight;
    term_L(isnan(term_L)) = 0;
    term_R(isnan(term_R)) = 0;

    EV_u = term_L + term_R;
    EV_u_weighted = EV_u .* pi_u_shape;
    EV_u_weighted(EV_u == -Inf & pi_u_shape == 0) = 0;
    EV_compact = sum(EV_u_weighted, dim_u);
    EV_compact(isnan(EV_compact)) = -Inf;

    % Interpolate Warmglow mapping (if active)
    if warmglow == 1
        WG_flat = WG_vec;
        term_L_WG = WG_flat(idx) .* (1 - weight);
        term_R_WG = WG_flat(idx+1) .* weight;
        term_L_WG(isnan(term_L_WG)) = 0;
        term_R_WG(isnan(term_R_WG)) = 0;
        WG_u = term_L_WG + term_R_WG;
        WG_u_weighted = WG_u .* pi_u_shape;
        WG_u_weighted(WG_u == -Inf & pi_u_shape == 0) = 0;
        WG_compact = sum(WG_u_weighted, dim_u);
        WG_compact(isnan(WG_compact)) = -Inf;
    else
        WG_compact = 0;
    end

    ezc6_val = ezc6(min(jj, length(ezc6))); ezc8_val = ezc8(min(jj, length(ezc8)));

    if warmglow == 1
        valid_ez = isfinite(EV_compact) & isfinite(WG_compact);
        EV_transformed = EV_compact;
        EV_transformed(valid_ez) = (sj_val * EV_compact(valid_ez).^ezc8_val + (1 - sj_val) * WG_compact(valid_ez).^ezc8_val).^ezc6_val;
        EV_transformed((EV_compact == 0) & (WG_compact == 0)) = 0;
    else
        valid_ez = isfinite(EV_compact);
        EV_transformed = EV_compact;
        EV_transformed(valid_ez) = (sj_val * EV_compact(valid_ez).^ezc8_val).^ezc6_val;
        EV_transformed(EV_compact == 0) = 0;
    end

    % --- THE TENSOR REDUCTION ---
    ezc9_val = ezc9(min(jj, length(ezc9)));
    ezc3_val = ezc3(min(jj, length(ezc3)));

    EV_scaled = ezc9_val * ezc3_val * EV_transformed;

    Pol_d2_indices = cell(1, length(n_d));
    if l_d2 > 0
        for i = (l_d1 + 1) : (l_d1 + l_d2)
            [EV_scaled, Pol_d2_indices{i}] = max(EV_scaled, [], i);
        end
    end
    % =========================================================================
    % PHASE 4: UNIVERSAL RHS & MAXIMIZATION
    % =========================================================================
    ezc7_val = ezc7(min(jj, length(ezc7)));
    RHS_native = ezc1_j * F_tensor + beta_j * ezc9_val * EV_scaled;

    valid_rhs = isfinite(RHS_native) & (RHS_native ~= 0);
    RHS_native(valid_rhs) = RHS_native(valid_rhs).^ezc7_val;

    n_d_rem = n_d;
    if l_d2 > 0; n_d_rem((l_d1 + 1) : (l_d1 + l_d2)) = 1; end
    N_d_rem = max(1, prod(n_d_rem));

    full_shape = [n_d_rem, N_a1_safe, N_a1_safe, N_a2_safe, N_z];
    if length(full_shape) == 1; full_shape = [full_shape, 1]; end

    RHS_native = RHS_native + zeros(full_shape, 'like', proto);

    FLAT_CHOICES_REM = N_d_rem * N_a1_safe;
    FLAT_STATES  = N_a1_safe * N_a2_safe * N_z;
    RHS_flat = reshape(RHS_native, [FLAT_CHOICES_REM, FLAT_STATES]);

    [V_max_flat, Pol_idx_flat] = max(RHS_flat, [], 1);
    V_j_max = reshape(V_max_flat, [N_a, N_z]);

    choice_sizes = [n_d_rem, N_a1_safe];
    idx_rem = Pol_idx_flat;

    for i = 1:length(n_d)
        sz = choice_sizes(i);
        pol_val = mod(idx_rem - 1, sz) + 1;

        if l_d2 > 0 && i >= (l_d1 + 1) && i <= (l_d1 + l_d2)
            d2_full = Pol_d2_indices{i} + zeros(full_shape, 'like', proto);
            d2_mat = reshape(d2_full, [FLAT_CHOICES_REM, FLAT_STATES]);
            lin_idx = Pol_idx_flat + (0:FLAT_STATES-1) * FLAT_CHOICES_REM;
            Policy_out_flat(i, :, :, jj) = reshape(d2_mat(lin_idx), [1, N_a, N_z]);
        else
            Policy_out_flat(i, :, :, jj) = reshape(pol_val, [1, N_a, N_z]);
        end
        idx_rem = ceil(idx_rem / sz);
    end

    if length(n_a1) > 0
        sz = choice_sizes(length(n_d) + 1);
        pol_val = mod(idx_rem - 1, sz) + 1;
        Policy_out_flat(length(n_d) + 1, :, :, jj) = reshape(pol_val, [1, N_a, N_z]);
    end

    V_out_flat(:, :, jj) = V_j_max;
    V_next = V_j_max;
end

% =========================================================================
% PHASE 5: SHAPE RESTORATION
% =========================================================================
out_n_a = n_a(n_a > 0); if isempty(out_n_a); out_n_a = 1; end
out_n_z = n_z(n_z > 0); if isempty(out_n_z); out_n_z = 1; end

state_shape = [out_n_a, out_n_z, N_j];

V = reshape(V_out_flat, state_shape);
Policy = reshape(Policy_out_flat, [size(Policy_out_flat, 1), state_shape]);


end