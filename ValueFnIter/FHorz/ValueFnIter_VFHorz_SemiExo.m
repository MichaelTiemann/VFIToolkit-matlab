function [V, Policy] = ValueFnIter_VFHorz_SemiExo(n_d1, n_d2, n_a, n_semiz, n_z, N_j, ...
    d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, ...
    pi_z_J, pi_semiz_J, ReturnFn, Parameters, ...
    DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Dimensions
n_d = [n_d1, n_d2];
N_d1 = prod(n_d1);
N_d2 = prod(n_d2);
N_d = N_d1 * N_d2;
N_a = prod(n_a);
N_semiz = prod(n_semiz);
N_z = prod(n_z);
n_all_z = [n_semiz, n_z];
N_bothz = prod(n_all_z);
has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
if has_e, error('Vectorized SemiExo does not currently support e shocks.'); end

% --- AGGRESSIVE GPU ALLOCATION ---
if vfoptions.parallel == 2
    pi_semiz_J = gpuArray(pi_semiz_J);
    semiz_gridvals_J = gpuArray(semiz_gridvals_J);
    pi_z_J = gpuArray(pi_z_J);
    z_gridvals_J = gpuArray(z_gridvals_J);
    d1_gridvals = gpuArray(d1_gridvals);
    d2_gridvals = gpuArray(d2_gridvals);

    if isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        vfoptions.V_Jplus1 = gpuArray(vfoptions.V_Jplus1);
    end
end
% ---------------------------------

% 2. Choice Grid (D_cells)
d_gridvals = [repmat(d1_gridvals, N_d2, 1), repelem(d2_gridvals, N_d1, 1)];
num_d = length(n_d);
D_cells = cell(1, num_d);
for i_d = 1:num_d
    D_cells{i_d} = shiftdim(d_gridvals(:, i_d), -3); % Dim 4
end

% 3. Endogenous Grid (A_mat)
num_a = length(n_a);
if num_a > 1
    a_grids_1d = cell(1, num_a); offset = 0;
    for i_a = 1:num_a
        a_grids_1d{i_a} = a_grid((offset + 1):(offset + n_a(i_a)));
        offset = offset + n_a(i_a);
    end
    [A_mesh_raw{1:num_a}] = ndgrid(a_grids_1d{:});
    A_mat = zeros(N_a, num_a, 'like', a_grid);
    for i_a = 1:num_a, A_mat(:, i_a) = A_mesh_raw{i_a}(:); end
else
    A_mat = a_grid(:);
end
a_work = A_mat(:, 1);

% 4. Preallocate
V = zeros(N_a, N_bothz, N_j, 'like', a_grid);
if vfoptions.gridinterplayer == 1
    PolicyKron = zeros(5, N_a, N_bothz, N_j, 'like', a_grid); % Must be 5 rows for GI!
else
    PolicyKron = zeros(3, N_a, N_bothz, N_j, 'like', a_grid);
end
V_next = zeros(N_a, N_bothz, 'like', a_grid);

% 5. Backward Induction
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    
    beta_j = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));
    ReturnFnParamsVec = num2cell(CreateVectorFromParams(Parameters, ReturnFnParamNames, jj));
    
    % Joint Z Grid for this period
    bothz_gridvals_j = [repmat(semiz_gridvals_J(:,:,jj), N_z, 1), repelem(z_gridvals_J(:,:,jj), N_semiz, 1)];
    num_z = length(n_all_z);
    Z_cells = cell(1, num_z);
    for i_z = 1:num_z
        Z_cells{i_z} = shiftdim(bothz_gridvals_j(:, i_z), -1); % Dim 2
    end
    
    % --- EXPECTATIONS ---
    if jj == N_j
        if isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
            temp_V_rs = reshape(vfoptions.V_Jplus1, [N_a * N_semiz, N_z]);
            pi_z_j = pi_z_J(:,:,min(jj, size(pi_z_J, 3)));
            EV_raw_rs = temp_V_rs * (pi_z_j');
            EV_slice = reshape(EV_raw_rs, [N_a, N_semiz, N_z]);
        else
            EV_slice = zeros(N_a, N_semiz, N_z, 'like', a_grid);
            pi_z_j = eye(N_z, 'like', a_grid); % Dummy for dispatchers
        end
    else
        % Integrate out Exogenous Z (independent of choices)
        pi_z_j = pi_z_J(:,:,jj);
        temp_V_rs = reshape(V_next, [N_a * N_semiz, N_z]);
        EV_raw_rs = temp_V_rs * (pi_z_j');
        EV_slice = reshape(EV_raw_rs, [N_a, N_semiz, N_z]);
    end
    
    % Preallocate running max for this period
    V_j_max = -inf(N_a, N_bothz, 'like', a_grid);
    Pol_d1_max = ones(N_a, N_bothz, 'like', a_grid);
    Pol_d2_max = ones(N_a, N_bothz, 'like', a_grid);
    Pol_apr_max = ones(N_a, N_bothz, 'like', a_grid);
    if vfoptions.gridinterplayer == 1
        Pol_tau_max = ones(N_a, N_bothz, 'like', a_grid);
        Pol_L2_max = 2 * ones(N_a, N_bothz, 'like', a_grid);
    end

    pi_semiz_j = pi_semiz_J(:,:,:,min(jj, size(pi_semiz_J, 4)));
    
    % =========================================================
    % TILED MAP-REDUCE: Chunking over d2 to fit GPU L2 Cache
    % =========================================================
    BellmanCombiner = @(F, EV_cont) F + beta_j .* EV_cont;
    for i_d2 = 1:N_d2
        % 1. Choice-Dependent Expectation (Tiny VRAM math)
        temp_EV = reshape(EV_slice, [N_a, N_semiz, N_z]);
        temp_EV_flat = reshape(permute(temp_EV, [1, 3, 2]), [N_a * N_z, N_semiz]);
        
        EV_d2_flat = temp_EV_flat * pi_semiz_j(:,:,i_d2); 
        
        EV_d2 = permute(reshape(EV_d2_flat, [N_a, N_z, N_semiz]), [1, 3, 2]);
        EV_d2_slice = reshape(EV_d2, [N_a, N_semiz * N_z]); % [N_a, N_bothz]
        
        % 2. Choice Grids for THIS d2 slice only (Size N_d1)
        d_gridvals_slice = [d1_gridvals, repmat(d2_gridvals(i_d2), N_d1, 1)];
        D_cells_slice = cell(1, num_d);
        for i_d = 1:num_d
            D_cells_slice{i_d} = shiftdim(d_gridvals_slice(:, i_d), -3); % Dim 4
        end

        eval_kernel = @(d_in, apr_in, A_cells, z_in) ReturnFn(...
            D_cells_slice{:}, apr_in, A_cells{:}, Z_cells{:}, ReturnFnParamsVec{:});

        % 3. Dispatch to Standard Kernels (They evaluate only N_d1 choices!)
        if vfoptions.divideandconquer == 1 && vfoptions.gridinterplayer == 1
            [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_DC1_GI1(...
                eval_kernel, BellmanCombiner, EV_d2_slice, A_mat, bothz_gridvals_j(:,1), d1_gridvals, ...
                N_a, N_bothz, N_d1, pi_z_j, ReturnFnParamsVec, vfoptions);
        elseif vfoptions.gridinterplayer == 1
            [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_GI1_raw(...
                eval_kernel, BellmanCombiner, EV_d2_slice, A_mat, bothz_gridvals_j(:,1), d1_gridvals, ...
                N_a, N_bothz, N_d1, pi_z_j, ReturnFnParamsVec, vfoptions);
        else
            [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_raw(...
                eval_kernel, BellmanCombiner, EV_d2_slice, A_mat, bothz_gridvals_j(:,1), d1_gridvals, ...
                N_a, N_bothz, N_d1, vfoptions);
        end
        
        % 4. Reduce (Keep the maximums)
        V_sub_rs = reshape(V_sub, [N_a, N_bothz]);
        
        % Force seed on first chunk, otherwise strictly greater
        if i_d2 == 1
            update_mask = true(N_a, N_bothz);
        else
            update_mask = V_sub_rs > V_j_max;
        end
        
        V_j_max(update_mask) = V_sub_rs(update_mask);
        
        % Map choices directly from the N_d1 local space
        d1_opt = reshape(mod(Pol_sub(1,:) - 1, N_d1) + 1, [N_a, N_bothz]);
        apr_coarse = reshape(ceil(Pol_sub(1,:) ./ N_d1), [N_a, N_bothz]);
        
        Pol_d1_max(update_mask) = d1_opt(update_mask);
        Pol_d2_max(update_mask) = i_d2;
        Pol_apr_max(update_mask) = apr_coarse(update_mask);
        
        if vfoptions.gridinterplayer == 1
            tau_opt = reshape(Pol_sub(2,:), [N_a, N_bothz]);
            L2_opt = reshape(Pol_sub(3,:), [N_a, N_bothz]);
            Pol_tau_max(update_mask) = tau_opt(update_mask);
            Pol_L2_max(update_mask) = L2_opt(update_mask);
        end
    end % End Map-Reduce Loop
    
    % Assign to global containers
    V(:,:,jj) = V_j_max;
    PolicyKron(1,:,:,jj) = Pol_d1_max;
    PolicyKron(2,:,:,jj) = Pol_d2_max;
    
    if vfoptions.gridinterplayer == 1
        % Apply toolkit safety clamp to the WINNING choices
        G_segments = vfoptions.ngridinterp + 1; 
        at_top = (Pol_apr_max == N_a);
        Pol_apr_max(at_top) = N_a - 1;
        Pol_tau_max(at_top) = G_segments + 1; 
        
        PolicyKron(3,:,:,jj) = Pol_apr_max;
        PolicyKron(4,:,:,jj) = Pol_tau_max;
        PolicyKron(5,:,:,jj) = Pol_L2_max;
    else
        PolicyKron(3,:,:,jj) = Pol_apr_max;
    end
    
    V_next = V(:,:,jj);
end

Policy = PolicyKron;


end