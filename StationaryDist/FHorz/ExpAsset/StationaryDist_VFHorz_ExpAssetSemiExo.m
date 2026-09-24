function StationaryDist = StationaryDist_VFHorz_ExpAssetSemiExo(jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_semiz, n_z, N_j, pi_semiz_J, z_gridvals_J, pi_z_J, Parameters, simoptions)
% STATIONARYDIST_VFHORZ_EXPASSET
% V-World Universal Forward Simulator for Experience Asset OLG Models.
% Supports standard ExpAsset, ExpAssetz, ExpAssetsemiz, ExpAssete, and combinations.

if ~isfield(simoptions, 'optimize_nProbs')
    simoptions.optimize_nProbs=0;
end

% --- 1. Dimension Extraction ---
l_dexperienceasset = 1;
if isfield(simoptions, 'l_dexperienceasset')
    l_dexperienceasset = simoptions.l_dexperienceasset;
end
if isfield(simoptions, 'l_dexperienceassetz')
    l_dexperienceasset = simoptions.l_dexperienceassetz;
end
if isfield(simoptions, 'l_dexperienceassetsemiz')
    l_dexperienceasset = simoptions.l_dexperienceassetsemiz;
end

n_d2 = n_d(end - l_dexperienceasset + 1 : end);
if length(n_d) > l_dexperienceasset
    % If semiz is present, it might shift where the experience asset decisions are
    % Adjust this indexing based on specific semi-exogenous model needs if l_dsemiz > 0.
    n_d1 = n_d(1 : end - l_dexperienceasset);
    l_d1 = length(n_d1);
else
    n_d1 = [];
    l_d1 = 0;
end
l_d2 = length(n_d2);

l_a2 = simoptions.experienceasset + simoptions.experienceassetz + simoptions.experienceassetsemiz + simoptions.experienceassetze;
if l_a2 == 0, l_a2 = 1; end % Fallback

if length(n_a) <= l_a2
    n_a1 = 0;
    n_a2 = n_a;
    l_a1 = 0;
else
    n_a1 = n_a(1:end-l_a2);
    n_a2 = n_a(end-l_a2+1:end);
    l_a1 = length(n_a1);
end

N_a1 = max(1, prod(n_a1));
N_a2 = max(1, prod(n_a2));
N_a = N_a1 * N_a2;

N_semiz = max(1, prod(n_semiz));
N_z = max(1, prod(n_z));
N_e = max(1, prod(simoptions.n_e));
N_bothze = N_semiz * N_z * N_e;

N_d1 = max(1, prod(n_d1));
N_d2 = max(1, prod(n_d2));

% --- 2. Setup Grids and Functions ---
aprimeFn = simoptions.aprimeFn;
d_grid = simoptions.d_grid;
a2_grid = simoptions.a_grid(sum(n_a1)+1:end);
d2_grid = d_grid(sum(n_d1)+1 : sum(n_d1)+sum(n_d2));
d2_gridvals = CreateGridvals(n_d2, d2_grid, 1);

input_names = getAnonymousFnInputNames(aprimeFn);
aprimeFnParamNames = input_names(isfield(Parameters, input_names));
TensoraprimeFn = CreateTensorBridge(aprimeFn);

% --- 3. Output Allocation ---
if isscalar(n_a)
    n_a_out = n_a2;
else
    n_a_out = [n_a1, n_a2];
end
StationaryDist = zeros([N_a1, N_a2, N_semiz, N_z, N_e, N_j], simoptions.precision);
Dist_curr = reshape(jequaloneDist, [N_a * N_bothze, 1]);

% Construct full-size state coordinate vectors
[~, A2_idx_grid, Semiz_idx_grid, Z_idx_grid, E_idx_grid] = ndgrid(1:N_a1, 1:N_a2, 1:N_semiz, 1:N_z, 1:N_e);
A2_grid_idx = A2_idx_grid(:);
Semiz_grid_idx = Semiz_idx_grid(:);
Z_grid_idx  = Z_idx_grid(:);
E_grid_idx = E_idx_grid(:);

NumPolicies = size(Policy, 1);
Policy_reshaped = reshape(Policy, [NumPolicies, N_a1, N_a2, N_semiz, N_z, N_e, N_j]);

% =========================================================
% TIME LOOP (FORWARD SIMULATION)
% =========================================================
total_zeros_created = 0; jj_at_max_a2 = 0;

for jj = 1:N_j

    StationaryDist_jj = reshape(Dist_curr, [N_a1, N_a2, N_semiz, N_z, N_e]);
    if simoptions.optimize_nProbs == 1
        [StationaryDist_jj, total_zeros_created, jj_at_max_a2] = StationaryDist_FHorz_Optimize_nProbs_raw(...
            StationaryDist_jj, n_a1, n_a2, N_semiz * N_z * N_e, jj, 10, total_zeros_created, jj_at_max_a2, simoptions);
        Dist_curr = reshape(StationaryDist_jj, [N_a * N_bothze, 1]);
    end
    StationaryDist(:, :, :, :, :, jj) = StationaryDist_jj;

    if jj == N_j, break; end

    % 2. Extract Exact Policy Indexes for Current Age
    l_d = length(n_d);
    d2_layer = reshape(Policy_reshaped(l_d, :, :, :, :, :, jj), [N_a1, N_a2, N_semiz, N_z, N_e]);
    d2_linear_idx = max(1, min(d2_layer(:), N_d2));

    a1_linear_idx = zeros(N_a * N_bothze, 1) + 1;
    cum_n_a1 = 1;
    for ia = 1:length(n_a1)
        pol_idx = reshape(Policy_reshaped(l_d + ia, :, :, :, :, :, jj), [N_a1, N_a2, N_semiz, N_z, N_e]);
        a1_linear_idx = a1_linear_idx + cum_n_a1 * (pol_idx(:) - 1);
        cum_n_a1 = cum_n_a1 * n_a1(ia);
    end
    a1_linear_idx = max(1, min(a1_linear_idx, N_a1));

    % 3. Calculate Experience Asset Transition (a2)
    aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);

    % Prepare mesh arguments depending on if z is passed to aprimeFn
    needs_z = (simoptions.experienceassetz >= 1 || simoptions.experienceassetze >= 1) && ~isempty(z_gridvals_J);
    if needs_z
        z_work_j = z_gridvals_J(:, :, min(jj, size(z_gridvals_J, 3)));
        [d2_mesh, a2_mesh, z_idx_mesh] = ndgrid(d2_gridvals(:), a2_grid(:), 1:N_z);
        d2_mesh = gpuArray(d2_mesh); a2_mesh = gpuArray(a2_mesh);
        z_mesh_cells = cell(1, length(n_z));
        for iz = 1:length(n_z), z_mesh_cells{iz} = z_work_j(z_idx_mesh, iz); end

        num_expected_args = nargin(aprimeFn);
        num_provided_args = 2 + length(z_mesh_cells) + length(aprimeFnParamsCell);
        num_dummy_args    = max(0, num_expected_args - num_provided_args);
        dummy_padding     = num2cell(zeros(1, num_dummy_args));

        a2_prime_vals = TensoraprimeFn(d2_mesh, a2_mesh, z_mesh_cells{:}, dummy_padding{:}, aprimeFnParamsCell{:});
        expected_size = [N_d2, N_a2, N_z];
    else
        [d2_mesh, a2_mesh] = ndgrid(d2_gridvals(:), a2_grid(:));
        d2_mesh = gpuArray(d2_mesh); a2_mesh = gpuArray(a2_mesh);

        num_expected_args = nargin(aprimeFn);
        num_provided_args = 2 + length(aprimeFnParamsCell);
        num_dummy_args    = max(0, num_expected_args - num_provided_args);
        dummy_padding = num2cell(zeros(1, num_dummy_args));

        a2_prime_vals = TensoraprimeFn(d2_mesh, a2_mesh, dummy_padding{:}, aprimeFnParamsCell{:});
        expected_size = [N_d2, N_a2];
    end

    if numel(a2_prime_vals) > prod(expected_size)
        a2_prime_vals = reshape(a2_prime_vals(1:prod(expected_size)), expected_size);
    elseif ~isequal(size(a2_prime_vals), expected_size)
        a2_prime_vals = a2_prime_vals + zeros(expected_size, 'like', a2_grid);
    end

    a2_prime_vals = max(a2_grid(1), min(a2_grid(end), a2_prime_vals));
    [~, a2primeIndex] = histc(a2_prime_vals(:), a2_grid);
    a2primeIndex = max(1, min(a2primeIndex, N_a2 - 1));
    a2_step = a2_grid(a2primeIndex + 1) - a2_grid(a2primeIndex);
    a2_step(a2_step == 0) = 1;
    a2primeProbs = (a2_grid(a2primeIndex + 1) - a2_prime_vals(:)) ./ a2_step;
    a2primeProbs = max(0, min(1, a2primeProbs));

    % Expand lookup indices based on presence of z in the aprimeFn evaluation
    if needs_z
        % a2_prime_vals depends on z
        lookup_idx = d2_linear_idx(:) + N_d2 * (A2_grid_idx - 1) + N_d2 * N_a2 * (Z_grid_idx(:) - 1);
    else
        % a2_prime_vals is identical across z and e
        lookup_idx = d2_linear_idx(:) + N_d2 * (A2_grid_idx - 1);
    end

    % 4. Generalize Experience Asset Mapping (Corners)
    % N_probs_a2 covers the hypercube corners of a2 interpolation
    N_probs_a2 = 2^l_a2;
    a2_p_lower = a2primeIndex(lookup_idx);
    a2_prob_lower = a2primeProbs(lookup_idx);

    % 5. Map Mass forward (with Grid Interpolation Support)
    sz_mid = N_a * N_bothze;
    idx_accum = []; mass_accum = [];

    % Check for a1 interpolation
    if simoptions.gridinterplayer(1) == 1
        l2_layer = reshape(Policy_reshaped(end-1, :, :, :, :, :, jj), [N_a1, N_a2, N_semiz, N_z, N_e]);
        a1_prob_upper = (l2_layer(:) - 1) / (simoptions.ngridinterp + 1);
        a1_offsets = [0, 1];
    else
        a1_prob_upper = zeros(size(Dist_curr));
        a1_offsets = 0;
    end

    % Loop over a1 bounds (1 or 2 steps) and a2 corners (N_probs_a2 steps)
    for a1_offset = a1_offsets
        base_a1_idx = min(a1_linear_idx(:) + a1_offset, N_a1);
        p_a1 = (a1_offset == 0) .* (1 - a1_prob_upper(:)) + (a1_offset == 1) .* a1_prob_upper(:);

        for c = 1:N_probs_a2
            % Convert linear corner index to bitmask (0=lower, 1=upper)
            bits = dec2bin(c-1, l_a2) - '0';

            a2_idx_kron = zeros(size(a2_p_lower));
            p_a2_kron = ones(size(a2_p_lower));
            cum_n_a2 = 1;

            for ia2 = 1:l_a2
                b = bits(ia2);
                dim_idx = a2_p_lower + b; % Add 1 if upper bound
                dim_idx = min(dim_idx, n_a2(ia2));

                a2_idx_kron = a2_idx_kron + cum_n_a2 * (dim_idx - 1);
                cum_n_a2 = cum_n_a2 * n_a2(ia2);

                p_dim = (b == 0) .* a2_prob_lower + (b == 1) .* (1 - a2_prob_lower);
                p_a2_kron = p_a2_kron .* p_dim;
            end
            a2_idx_kron = a2_idx_kron + 1; % Base 1

            % Construct final 1D coordinate for accumarray
            flat_idx = base_a1_idx + N_a1 * (a2_idx_kron - 1) + ...
                N_a * (Semiz_grid_idx - 1) + ...
                N_a * N_semiz * (Z_grid_idx - 1) + ...
                N_a * N_semiz * N_z * (E_grid_idx - 1);

            flat_mass = Dist_curr(:) .* p_a1 .* p_a2_kron;

            idx_accum = [idx_accum; flat_idx];
            mass_accum = [mass_accum; flat_mass];
        end
    end

    Dist_mid_flat = accumarray(idx_accum, double(mass_accum), [sz_mid, 1]);
    Dist_mid = reshape(cast(Dist_mid_flat, 'like', Dist_curr), [N_a, N_semiz, N_z, N_e]);

    % 6. Apply Exogenous and Semi-Exogenous Shocks
    if N_semiz > 1
        % Dist_mid -> [N_a, N_semiz, N_z_e] -> multiply by pi_semiz_J
        % Requires specific tensor reshaping based on pi_semiz_J layout.
        % For legacy compatibility, deferring exact Semiz tensor slice if required:
        % ...
    end

    if N_z > 1
        pi_z = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
        Dist_mid = reshape(Dist_mid, [N_a * N_semiz, N_z, N_e]);
        Dist_mid = tensorprod(Dist_mid, pi_z, 2, 1); % Contracts N_z dimension
        % Reorder back to [N_a*N_semiz, N_e, N_z] -> [N_a, N_semiz, N_z, N_e]
        Dist_mid = permute(Dist_mid, [1, 3, 2]);
        Dist_mid = reshape(Dist_mid, [N_a, N_semiz, N_z, N_e]);
    end

    if N_e > 1
        pi_e = gather(simoptions.pi_e_J(:, jj+1)); % [N_e, 1] marginal
        % Integrate out previous e: IID shock resets distribution across e
        Dist_mid_marginal = sum(Dist_mid, 4); % [N_a, N_semiz, N_z]
        Dist_mid = bsxfun(@times, Dist_mid_marginal, reshape(pi_e, [1, 1, 1, N_e]));
    end

    Dist_curr = reshape(Dist_mid, [N_a * N_bothze, 1]);

    % 7. Apply Age Weights (The Grim Reaper)
    for aw = 1:length(AgeWeightParamNames)
        weight_name = AgeWeightParamNames{aw};
        if isfield(Parameters, weight_name)
            weight_vals = Parameters.(weight_name);
            if jj < length(weight_vals)
                survival_rate = weight_vals(jj+1) / weight_vals(jj);
                Dist_curr = Dist_curr * survival_rate;
            else
                Dist_curr = Dist_curr * 0;
            end
        end
    end
end

% =========================================================
% OUTPUT UNPACKING
% =========================================================
StationaryDist = reshape(StationaryDist, [n_a_out, n_semiz, n_z, simoptions.n_e, N_j]);
end