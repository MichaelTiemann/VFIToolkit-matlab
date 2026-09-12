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
    PolicyKron = zeros(4, N_a, N_bothz, N_j, 'like', a_grid);
else
    PolicyKron = zeros(3, N_a, N_bothz, N_j, 'like', a_grid);
end
V_next = zeros(N_a, N_bothz, 'like', a_grid);

% 5. Backward Induction
for reverse_j = 1:N_j-1
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
    % Integrate out Exogenous Z (independent of choices)
    pi_z_j = pi_z_J(:,:,jj);
    temp_V_rs = reshape(V_next, [N_a * N_semiz, N_z]);
    EV_raw_rs = temp_V_rs * (pi_z_j');
    EV_slice = reshape(EV_raw_rs, [N_a, N_semiz, N_z]);
    
    % Pass SemiExo transition matrix to dispatchers for choice-dependent integration
    vfoptions.pi_semiz_j_active = pi_semiz_J(:,:,:,jj);
    
    % Define Kernel & Combiner
    eval_kernel = @(d_in, apr_in, A_cells, z_in) ReturnFn(D_cells{:}, apr_in, A_cells{:}, Z_cells{:}, ReturnFnParamsVec{:});
    BellmanCombiner = @(F, EV_cont) F + beta_j .* EV_cont;
    
    % --- DISPATCH TO UNIVERSAL KERNELS ---
    if vfoptions.divideandconquer == 1 && vfoptions.gridinterplayer == 1
        [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_DC1_GI1(...
            eval_kernel, BellmanCombiner, EV_slice, A_mat, bothz_gridvals_j(:,1), d_gridvals(:,1), ...
            N_a, N_bothz, N_d, pi_z_j, ReturnFnParamsVec, vfoptions);
    elseif vfoptions.gridinterplayer == 1
        [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_GI1_raw(...
            eval_kernel, BellmanCombiner, EV_slice, A_mat, bothz_gridvals_j(:,1), d_gridvals(:,1), ...
            N_a, N_bothz, N_d, pi_z_j, ReturnFnParamsVec, vfoptions);
    else
        [V_sub, Pol_sub] = ValueFnIter_FHorz_vectorized_raw(...
            eval_kernel, BellmanCombiner, EV_slice, A_mat, bothz_gridvals_j(:,1), d_gridvals(:,1), ...
            N_a, N_bothz, N_d, vfoptions);
    end
    
    V(:,:,jj) = reshape(V_sub, [N_a, N_bothz]);
    
    % Map flat Policy into SemiExo multi-row format [d1, d2, aprime_coarse, tau]
    d_opt = mod(Pol_sub(1,:) - 1, N_d) + 1;
    PolicyKron(1,:,:,jj) = reshape(mod(d_opt - 1, N_d1) + 1, [N_a, N_bothz]); % d1
    PolicyKron(2,:,:,jj) = reshape(ceil(d_opt / N_d1), [N_a, N_bothz]);       % d2
    
    if vfoptions.gridinterplayer == 1
        apr_tau_opt = ceil(Pol_sub(1,:) ./ N_d);
        coarse_a_opt = mod(apr_tau_opt - 1, N_a) + 1;
        tau_opt = ceil(apr_tau_opt ./ N_a);
        
        PolicyKron(3,:,:,jj) = reshape(coarse_a_opt, [N_a, N_bothz]);
        PolicyKron(4,:,:,jj) = reshape(tau_opt, [N_a, N_bothz]);
    else
        PolicyKron(3,:,:,jj) = reshape(ceil(Pol_sub(1,:) ./ N_d), [N_a, N_bothz]);
    end
    
    V_next = V(:,:,jj);
end

Policy = PolicyKron;


end