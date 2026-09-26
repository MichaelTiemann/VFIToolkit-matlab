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
has_z = prod(n_z) > 0; n_z_work = N_semiz * N_z_exog; n_e_work = max(1, prod(n_e_pass)); N_ze = n_z_work * n_e_work;

V = zeros(N_a, N_ze, N_j, 'gpuArray');
Policy = zeros(N_a, N_ze, N_j, 'gpuArray');
V_next = zeros(N_a, N_ze, 'gpuArray');

base_ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);
is_age_dependent = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j; is_age_dependent(ip) = true; end
    if ~isa(base_ReturnFnParamsCell{ip}, 'gpuArray'); base_ReturnFnParamsCell{ip} = gpuArray(base_ReturnFnParamsCell{ip}); end
end

% Pre-compute z and e values for calls to TensorFn and aprimeFn
ze_chunks = {1:N_ze};
chunk_meta = cell(1, length(ze_chunks));
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
    pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    for ip = find(is_age_dependent)
        ReturnFnParamsCell{ip} = cast(Parameters.(ReturnFnParamNames{ip})(jj), 'like', a_grid);
    end
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision);
    beta_j = prod(DiscountFactorParamsVec);

    % EV Expectation Step
    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(vfoptions.V_Jplus1, [N_a, N_z]);
        V_next = gpuArray(V_next);
    elseif jj == N_j
        V_next = zeros(N_a, N_z, 'like', a_grid);
    else
        V_trans_flat = reshape(V_next, [N_a, N_z]);
        V_inf_mask = (V_trans_flat == -Inf);
        V_safe = V_trans_flat;
        V_safe(V_inf_mask) = -1e250;
        V_expected = V_safe * pi_z_j';
        inf_restore = (V_inf_mask * (pi_z_j' > 0)) > 0;
        V_expected(inf_restore) = -Inf;
        V_next = V_expected;
    end

    EV_local = beta_j .* reshape(V_next, [N_a, N_z]);

    % Precompute static EV offset mapping for 3D Deflation
    if ~is_exp_asset
        EV_reshaped = reshape(EV_local, [N_a1_dc * N_a1_other, n_z_loc, n_e_loc, N_dsemiz]);
        EV_d_sliced = EV_reshaped(:, :, :, dsemiz_idx_tensor(:));
        EV_bounded_pre = beta_j .* permute(EV_d_sliced, [4, 1, 5, 2, 3]);

        % CRITICAL FIX: Use chunk-localized dimensions (n_z_loc, n_e_loc)
        % instead of global N_z to prevent index out-of-bounds.
        d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]);
        z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1_dc * N_a1_other), [1, 1, 1, n_z_loc, 1]);
        e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1_dc * N_a1_other * n_z_loc), [1, 1, 1, 1, n_e_loc]);
        static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
        static_EV_offset_fine = [];
    else
        % (Keep your existing experience asset precomputation block as is)
        error("exp_asset not yet supported")
    end

    for i_ze = 1:length(ze_chunks)
        meta = chunk_meta{i_ze};
        n_z_loc = meta.n_z_loc;
        n_e_loc = meta.n_e_loc;
        curr_ze = ze_chunks{i_ze};
        N_ze_local = length(curr_ze);

        % META TRICK: Build Z and E cells locally per chunk
        if has_semiz || has_z
            num_z_vars = length(n_all_z);
            Z_cells_local = cell(1, num_z_vars);
            if size(z_gridvals_J, 2) ~= num_z_vars
                z_inflated = reshape(z_gridvals_J, [N_z, num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
            else
                for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
            end
        else
            Z_cells_local = {};
        end

        if has_e
            num_e_vars = size(e_work, 2);
            E_cells_local = cell(1, num_e_vars);
            for ie_var = 1:num_e_vars; E_cells_local{ie_var} = reshape(e_work(meta.e_vals, ie_var), [1, 1, 1, 1, n_e_loc]); end
        else
            E_cells_local = {};
        end

        % Define Evaluation Block for DC Slicer using Meta-Trick cells
        LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_DC_TensorBlock(...
            state_idx, loweredge_matrix, maxgap_scalar, N_a1_dc, N_a1_other, max(1, N_a2), N_d_safe, N_z, ...
            Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, A1_grids_1d, ...
            EV_local, static_EV_offset, TensorReturnFn, ReturnFnParamsCell, n_z_loc, n_e_loc);

        % Dispatch Universal DC1 Slicer [ValueFnIter_DC1_Slicer.m](https://github.com/MichaelTiemann/VFIToolkit-matlab/raw/refs/heads/tensor-branch2/ValueFnIter/FHorz/DivideConquer/ValueFnIter_DC1_Slicer.m)
        [v, p_apr, p_d] = ValueFnIter_DC1_Slicer(N_a1_dc, N_a1_dc, 1, N_z, vfoptions, LocalBlockFn, N_d_safe);

        V(:, :, jj) = reshape(v, [N_a, N_z]);
        if N_d > 0
            Policy(:, :, jj) = (reshape(p_apr, [N_a, N_z]) - 1) * N_d + reshape(p_d, [N_a, N_z]);
        else
            Policy(:, :, jj) = reshape(p_apr, [N_a, N_z]);
        end
    end
    V_next = V(:, :, jj);
end

varargout{1} = gather(V);
varargout{2} = gather(Policy);


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
    N_a1_total = N_a1_dc;
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
    F_tensor = reshape(F_tensor, [FLAT_CHOICES, N_states, n_z_loc * n_e_loc]);

    choice_idx_linear = reshape(1:num_choices_total, [1, num_choices_total, 1, 1, 1]);
    a1_offset = (choice_idx_linear - 1) * N_d_safe;
    lin_idx_compact = static_EV_offset + a1_offset;
    EV_bounded = EV_local(lin_idx_compact);
    EV_bounded = reshape(EV_bounded, [FLAT_CHOICES, N_states, n_z_loc * n_e_loc]);
else
    total_gap = maxgap_scalar;
    num_choices_total = total_gap + 1;

    loweredge_matrix = max(1, min(loweredge_matrix, N_a1_dc));
    base_idx_a1 = reshape(loweredge_matrix, [N_d_safe, 1, N_states, N_z, 1]);
    offsets_a1 = reshape(0:total_gap, [1, total_gap + 1, 1, 1, 1]);
    choice_idx_a1_base = max(1, min(base_idx_a1 + offsets_a1, length(A1_grids_1d{1})));

    Apr_cells = { A1_grids_1d{1}(choice_idx_a1_base) };
    choice_idx_linear = choice_idx_a1_base;

    if N_a2 > 1
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_local{:}, E_cells_local{:}, ReturnFnParamsCell{:});
    else
        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_local{:}, E_cells_local{:}, ReturnFnParamsCell{:});
    end

    a1_offset = (choice_idx_linear - 1) * N_d_safe;
    lin_idx_compact = static_EV_offset + a1_offset;
    EV_bounded = EV_local(lin_idx_compact);
end

FLAT_CHOICES = N_d_safe * num_choices_total;
FLAT_STATES = N_states * N_z;

F_tensor = reshape(F_tensor, [FLAT_CHOICES, N_states, N_z]);
EV_bounded = reshape(EV_bounded, [FLAT_CHOICES, 1, N_z]);

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

V_j_max   = reshape(V_sub,       [N_d_safe, N_states, N_z]);
Pol_d_max = reshape(d_idx_local, [N_d_safe, N_states, N_z]);

if isempty(loweredge_matrix)
    Pol_apr_max = reshape(apr_idx_local, [N_d_safe, N_states, N_z]);
else
    a1_apr_offset = mod(apr_idx_local - 1, total_gap + 1) + 1;
    a2_offset_factor = ceil(apr_idx_local / (total_gap + 1));
    loweredge_2d = reshape(loweredge_matrix, [N_d_safe, FLAT_STATES]);
    d_vec = cast((1:N_d_safe)', 'like', apr_idx_local);
    s_vec = cast((0:FLAT_STATES-1) * N_d_safe, 'like', apr_idx_local);

    lin_idx_low = d_vec + s_vec;
    chosen_low = loweredge_2d(lin_idx_low);

    a1_Pol = min(chosen_low + a1_apr_offset - 1, N_a1_dc);
    Pol_apr_max = a1_Pol + (a2_offset_factor - 1) * N_a1_dc;
    Pol_apr_max = reshape(Pol_apr_max, [N_d_safe, N_states, N_z]);
end

Pol_L2idx = [];
Pol_L2flag = [];


end
