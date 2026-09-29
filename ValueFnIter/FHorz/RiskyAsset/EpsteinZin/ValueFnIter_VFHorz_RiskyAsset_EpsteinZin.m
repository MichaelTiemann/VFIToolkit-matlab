function [V, Policy] = ValueFnIter_VFHorz_RiskyAsset_EpsteinZin(...
    n_d, n_a1, n_a2, n_z, n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ...
    ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ~, ~, vfoptions, ...
    sj, warmglow, ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8)

% =========================================================================
% PHASE 1: PRE-COMPUTATION & AUTONOMOUS PARSING
% =========================================================================
N_d_safe = max(1, prod(n_d));
N_a1_safe = max(1, prod(n_a1));
N_a2_safe = max(1, prod(n_a2));
N_z = max(1, prod(n_z));
N_u = max(1, prod(n_u));
N_a = N_a1_safe * N_a2_safe;

if vfoptions.parallel == 2
    pi_z_J = gpuArray(pi_z_J);
    pi_u = gpuArray(pi_u);
end
pi_u_shape = reshape(pi_u(:), [1, 1, 1, 1, 1, N_u]);

% 1.1 Parse refine_d choices
if isfield(vfoptions, 'refine_d')
    l_d1 = vfoptions.refine_d(1); l_d2 = vfoptions.refine_d(2);
else
    l_d1 = 0; l_d2 = 0;
end
d_return_idx = [1:l_d1, (l_d1+l_d2+1):length(n_d)];
d_aprime_idx = (l_d1+1):length(n_d);

% 1.2 Backwards-Parser for ReturnFn (Bulletproof Parameter Extraction)
temp_ret = getAnonymousFnInputNames(ReturnFn);
first_param_idx = find(isfield(Parameters, temp_ret), 1, 'first');
if isempty(first_param_idx)
    ReturnFnParamNames = {}; num_prefix_ret = length(temp_ret);
else
    ReturnFnParamNames = temp_ret(first_param_idx:end); num_prefix_ret = first_param_idx - 1;
end

% 1.3 Backwards-Parser for aprimeFn
temp_ap = getAnonymousFnInputNames(aprimeFn);
first_param_idx = find(isfield(Parameters, temp_ap), 1, 'first');
if isempty(first_param_idx)
    aprimeFnParamNames = {}; num_prefix_ap = length(temp_ap);
else
    aprimeFnParamNames = temp_ap(first_param_idx:end); num_prefix_ap = first_param_idx - 1;
end

% 1.4 Native 1D Grids
d_grids = cell(1, length(n_d)); offset = 0;
for i = 1:length(n_d); d_grids{i} = d_grid(offset+1 : offset+n_d(i)); offset = offset + n_d(i); end
if length(n_d) > 1; [D_mesh{1:length(n_d)}] = ndgrid(d_grids{:}); else; D_mesh{1} = d_grids{1}; end
D_flat = cell(1, length(n_d)); for i = 1:length(n_d); D_flat{i} = D_mesh{i}(:); end

a1_grids = cell(1, length(n_a1)); offset = 0;
for i = 1:length(n_a1); a1_grids{i} = a1_grid(offset+1 : offset+n_a1(i)); offset = offset + n_a1(i); end
if length(n_a1) > 1; [A1_mesh{1:length(n_a1)}] = ndgrid(a1_grids{:}); else; A1_mesh{1} = a1_grids{1}; end
A1_flat = cell(1, length(n_a1)); for i = 1:length(n_a1); A1_flat{i} = A1_mesh{i}(:); end

a2_grids = cell(1, max(1, length(n_a2))); offset = 0;
for i = 1:length(n_a2); a2_grids{i} = a2_grid(offset+1 : offset+n_a2(i)); offset = offset + n_a2(i); end
if length(n_a2) > 1; [A2_mesh{1:length(n_a2)}] = ndgrid(a2_grids{:}); else; A2_mesh{1} = a2_grids{1}; end
A2_flat = cell(1, max(1, length(n_a2))); for i = 1:max(1, length(n_a2)); A2_flat{i} = A2_mesh{i}(:); end

% 1.5 Geometry Construction [N_d_safe, N_a1_prime, N_a1_state, N_a2_state, N_z, N_u]
D_cells = cell(1, length(n_d));
for i = 1:length(n_d); D_cells{i} = cast(reshape(D_flat{i}, [N_d_safe, 1, 1, 1, 1, 1]), 'like', a1_grid); end
Apr_cells = cell(1, length(n_a1));
for i = 1:length(n_a1); Apr_cells{i} = cast(reshape(A1_flat{i}, [1, N_a1_safe, 1, 1, 1, 1]), 'like', a1_grid); end
A1_cells = cell(1, length(n_a1));
for i = 1:length(n_a1); A1_cells{i} = cast(reshape(A1_flat{i}, [1, 1, N_a1_safe, 1, 1, 1]), 'like', a1_grid); end
A2_cells = cell(1, max(1, length(n_a2)));
for i = 1:length(n_a2); A2_cells{i} = cast(reshape(A2_flat{i}, [1, 1, 1, N_a2_safe, 1, 1]), 'like', a1_grid); end
if isempty(n_a2); A2_cells{1} = cast(reshape(0, [1, 1, 1, 1, 1, 1]), 'like', a1_grid); end
U_cells = {cast(reshape(u_grid, [1, 1, 1, 1, 1, N_u]), 'like', a1_grid)};

% 1.6 Argument Mapping
ret_args = cell(1, num_prefix_ret); idx = 1;
for i = 1:length(d_return_idx); ret_args{idx} = D_cells{d_return_idx(i)}; idx = idx + 1; end
if num_prefix_ret > idx - 1; for i = 1:length(n_a1); ret_args{idx} = Apr_cells{i}; idx = idx + 1; end; end
if num_prefix_ret > idx - 1; for i = 1:length(n_a1); ret_args{idx} = A1_cells{i}; idx = idx + 1; end; end
if num_prefix_ret > idx - 1; for i = 1:length(n_a2); ret_args{idx} = A2_cells{i}; idx = idx + 1; end; end
num_z_vars = size(z_gridvals_J, 2); % Z appended dynamically in loop

ap_args = cell(1, num_prefix_ap); idx = 1;
for i = 1:length(d_aprime_idx); ap_args{idx} = D_cells{d_aprime_idx(i)}; idx = idx + 1; end
if num_prefix_ap > length(d_aprime_idx) + 1; for i = 1:length(n_a2); ap_args{idx} = A2_cells{i}; idx = idx + 1; end; end
ap_args{idx} = U_cells{1};

V = zeros(N_a, N_z, N_j, 'like', a1_grid);
Policy = zeros(length(n_d) + length(n_a1), N_a, N_z, N_j, 'like', a1_grid);
V_next = zeros(N_a, N_z, 'like', a1_grid);

% PTX Constant Setup
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

ezc1 = 1 - prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, 1, vfoptions.precision));

% =========================================================================
% PHASE 2: REVERSE TIME LOOP
% =========================================================================
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose == 1; fprintf('RiskyAsset EZ Finite horizon: %i of %i \n', jj, N_j); end

    % 2.1 Dynamic Parameters
    ReturnFnParamsCell = base_RetParams;
    for ip = find(Ret_age)
        val = cast(Parameters.(ReturnFnParamNames{ip})(jj), vfoptions.precision);
        if vfoptions.parallel == 2; ReturnFnParamsCell{ip} = gpuArray(val); else; ReturnFnParamsCell{ip} = val; end
    end
    aprimeFnParamsCell = base_ApParams;
    for ip = find(Ap_age)
        val = cast(Parameters.(aprimeFnParamNames{ip})(jj), vfoptions.precision);
        if vfoptions.parallel == 2; aprimeFnParamsCell{ip} = gpuArray(val); else; aprimeFnParamsCell{ip} = val; end
    end

    beta_j = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision));
    sj_val = sj(jj);
    pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));

    ret_args_run = ret_args;
    idx = length(ret_args_run) - num_z_vars + 1;
    for iz = 1:num_z_vars
        ret_args_run{idx} = cast(reshape(z_gridvals_J(:, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, 1, N_z, 1]), 'like', a1_grid);
        idx = idx + 1;
    end

    % 2.2 Z-Expectation
    if jj == N_j
        EV_z = zeros(N_a1_safe, N_a2_safe, N_z, 'like', a1_grid);
    else
        valid_V = isfinite(V_next) & (V_next ~= 0);
        V_transformed = V_next;
        if ezc5(jj) == 1; V_transformed(valid_V) = ezc4 * V_next(valid_V); else; V_transformed(valid_V) = max(ezc4 * V_next(valid_V), 0).^ezc5(jj); end
        V_transformed(V_next == 0) = 0;

        V_safe = reshape(V_transformed, [N_a, N_z]);
        V_safe(V_safe == -Inf) = -1e250;
        EV_z_flat = V_safe * pi_z_j';
        EV_z_flat((V_safe == -Inf) * (pi_z_j' > 0) > 0) = -Inf;
        EV_z = reshape(EV_z_flat, [N_a1_safe, N_a2_safe, N_z]);
    end

    % Warm Glow
    if warmglow == 1
        wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj, vfoptions.precision);
        WG_eval = vfoptions.WarmGlowBequestsFn(a2_grid, wg_params{:});
        if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a2_grid), 'like', a2_grid); end
        valid_wg = isfinite(WG_eval) & (WG_eval ~= 0);
        WG_transformed = WG_eval;
        if ezc5(jj) == 1; WG_transformed(valid_wg) = ezc4 * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4 * WG_eval(valid_wg), 0).^ezc5(jj); end
        WG_transformed(WG_eval == 0) = 0;
        WG_vec = reshape(WG_transformed, [1, 1, 1, N_a2_safe, 1, 1]);
    else
        WG_vec = 0;
    end

    % =========================================================================
    % PHASE 3: TENSOR EVALUATION & IMPLICIT EXPANSION
    % =========================================================================

    % 1. Evaluate Return Function [N_d, N_a1_prime, N_a1_state, N_a2_state, N_z]
    F_tensor = arrayfun(ReturnFn, ret_args_run{:}, ReturnFnParamsCell{:});
    F_tensor(isfinite(F_tensor) & F_tensor ~= 0) = F_tensor(isfinite(F_tensor) & F_tensor ~= 0).^ezc2(jj);
    F_tensor(F_tensor == 0) = -Inf;

    % 2. Evaluate Portfolio Returns (a2_prime) [N_d, 1, 1, N_a2_state, 1, N_u]
    A2_prime = arrayfun(aprimeFn, ap_args{:}, aprimeFnParamsCell{:});
    a2_grid_1d_vec = a2_grids{1};
    a2_prime_clipped = max(a2_grid_1d_vec(1), min(A2_prime, a2_grid_1d_vec(end)));

    % 3. Find Interpolation Weights for A2_prime on a2_grid
    a2_grid_shape = cast(reshape(a2_grid_1d_vec, [1,1,1,1,1,1,length(a2_grid_1d_vec)]), 'like', A2_prime);
    idx = sum(a2_prime_clipped >= a2_grid_shape, 7);
    idx(idx == 0) = 1;
    idx(idx == length(a2_grid_1d_vec)) = length(a2_grid_1d_vec) - 1;

    a2_left = reshape(a2_grid_1d_vec(idx), size(idx));
    a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
    weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
    weight(a2_right == a2_left) = 0;
    weight(abs(weight) < 1e-12) = 0;
    weight(abs(weight - 1) < 1e-12) = 1;

    % 4. Build Linear Indices for EV_z [N_a1_prime, N_a2_prime, N_z]
    a1_prime_offset = reshape(0:N_a1_safe-1, [1, N_a1_safe, 1, 1, 1, 1]) * 1;
    a2_prime_offset = (idx - 1) * N_a1_safe;
    z_offset = reshape(0:N_z-1, [1, 1, 1, 1, N_z, 1]) * (N_a1_safe * N_a2_safe);

    idx_left  = 1 + a1_prime_offset + a2_prime_offset + z_offset;
    idx_right = idx_left + N_a1_safe;

    % 5. Interpolate EV_z and Evaluate U Expectation
    EV_flat = EV_z(:);
    term_L = EV_flat(idx_left) .* (1 - weight);
    term_R = EV_flat(idx_right) .* weight;
    term_L(isnan(term_L)) = 0;
    term_R(isnan(term_R)) = 0;

    EV_u = term_L + term_R;
    EV_compact = sum(EV_u .* pi_u_shape, 6);

    % 6. Apply Outer Epstein-Zin Transforms
    EV_compact(isnan(EV_compact)) = -Inf;
    if warmglow == 1
        valid_ez = isfinite(EV_compact) & isfinite(WG_vec);
        EV_transformed = EV_compact;
        EV_transformed(valid_ez) = (sj_val * EV_compact(valid_ez).^ezc8(jj) + (1 - sj_val) * WG_vec(valid_ez).^ezc8(jj)).^ezc6(jj);
        EV_transformed((EV_compact == 0) & (WG_vec == 0)) = 0;
    else
        valid_ez = isfinite(EV_compact);
        EV_transformed = EV_compact;
        EV_transformed(valid_ez) = (sj_val * EV_compact(valid_ez).^ezc8(jj)).^ezc6(jj);
        EV_transformed(EV_compact == 0) = 0;
    end

    % =========================================================================
    % PHASE 4: UNIVERSAL RHS & MAXIMIZATION
    % =========================================================================
    % EV_transformed automatically broadcasts across Dim 3 (a1_state)
    RHS_native = ezc1 * F_tensor + beta_j * EV_transformed;

    valid_rhs = isfinite(RHS_native) & (RHS_native ~= 0);
    RHS_native(valid_rhs) = RHS_native(valid_rhs).^ezc7(jj);

    FLAT_CHOICES = N_d_safe * N_a1_safe;
    FLAT_STATES  = N_a1_safe * N_a2_safe * N_z;
    RHS_flat = reshape(RHS_native, [FLAT_CHOICES, FLAT_STATES]);

    [V_max_flat, Pol_idx_flat] = max(RHS_flat, [], 1);
    V_j_max = reshape(V_max_flat, [N_a, N_z]);

    % Unpack Policies: d2 (riskyshare), d3 (savings), a1prime (hprime)
    d_sub = mod(Pol_idx_flat - 1, N_d_safe) + 1;
    Policy(1, :, :, jj) = reshape(mod(d_sub - 1, n_d(1)) + 1, [1, N_a, N_z]); % riskyshare
    Policy(2, :, :, jj) = reshape(ceil(d_sub / n_d(1)), [1, N_a, N_z]); % savings
    Policy(3, :, :, jj) = reshape(ceil(Pol_idx_flat / N_d_safe), [1, N_a, N_z]); % hprime

    V(:, :, jj) = V_j_max;
    V_next = V_j_max;
end


end