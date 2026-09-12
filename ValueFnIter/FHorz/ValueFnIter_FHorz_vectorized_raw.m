function [V_current, Policy_Row] = ValueFnIter_FHorz_vectorized_raw(...
    eval_kernel, BellmanCombiner, EV, A_mat, z_work, d_work, ...
    N_a, N_z, N_d, vfoptions)

a_work = A_mat(:, 1); % Extract primary asset grid for broadcasting and 'like' typing

% Check if e exists and whether this call processes multiple e simultaneously.
if isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0 && vfoptions.lowmemory == 0
    has_e = true;
    N_e = prod(vfoptions.n_e);
else
    has_e = false;
    N_e = 1;
end

% Align 5D grid: Dim 1: a, Dim 2: z, Dim 3: e, Dim 4: d, Dim 5: aprime
A_in   = num2cell(A_mat, 1);       % Packages all endogenous states natively spanning Dim 1
Z_in   = shiftdim(z_work(:), -1);  % Dim 2
D_in   = shiftdim(d_work(:), -3);  % Dim 4
Apr_in = shiftdim(a_work(:), -4);  % Dim 5

% Lines 22-25 in ValueFnIter_FHorz_vectorized_raw.m:
F = eval_kernel(D_in, Apr_in, A_in, Z_in);

% Zero-overhead guard: Only expands if an author writes a non-broadcasting ReturnFn
expected_sz = [N_a, N_z, N_e, N_d, N_a];
if ~isequal(size(F), expected_sz)
    F = F + zeros(expected_sz, 'like', a_work);
end

% -------------------------------------------------------------------------
% NEW: Choice-Dependent Tensor Contraction for Semi-Exogenous States
% -------------------------------------------------------------------------
if isfield(vfoptions, 'pi_semiz_j_active')
    pi_semiz = vfoptions.pi_semiz_j_active;
    N_semiz  = prod(vfoptions.n_semiz);
    N_z_exog = N_z / N_semiz;
    N_d2     = size(pi_semiz, 3);
    N_d1     = N_d / N_d2;
    
    if has_e
        EV_reshaped = reshape(EV, [N_a, N_semiz, N_z_exog, N_e]);
        EV_perm = permute(EV_reshaped, [1, 3, 4, 2]); 
        EV_flat = reshape(EV_perm, [N_a * N_z_exog * N_e, N_semiz]);
    else
        EV_reshaped = reshape(EV, [N_a, N_semiz, N_z_exog]);
        EV_perm = permute(EV_reshaped, [1, 3, 2]);
        EV_flat = reshape(EV_perm, [N_a * N_z_exog, N_semiz]);
    end
    
    EV_new = zeros(size(EV_flat, 1), N_semiz, N_d2, 'like', a_work);
    for d2_idx = 1:N_d2
        EV_new(:,:,d2_idx) = EV_flat * pi_semiz(:,:,d2_idx);
    end
    
    if has_e
        EV_new = reshape(EV_new, [N_a, N_z_exog, N_e, N_semiz, N_d2]);
        EV_new = permute(EV_new, [1, 4, 2, 3, 5]); 
        EV_new = reshape(EV_new, [N_a, N_z, N_e, N_d2]);
        
        EV_expected = repelem(EV_new, 1, 1, 1, N_d1); 
        EV_expected = reshape(EV_expected, [N_a, N_z, N_e, N_d, 1]);
        V_cont = permute(EV_expected, [5, 2, 3, 4, 1]); % -> [1, N_z, N_e, N_d, N_a]
    else
        EV_new = reshape(EV_new, [N_a, N_z_exog, N_semiz, N_d2]);
        EV_new = permute(EV_new, [1, 3, 2, 4]); 
        EV_new = reshape(EV_new, [N_a, N_z, N_d2]);
        
        EV_expected = repelem(EV_new, 1, 1, N_d1); 
        EV_expected = reshape(EV_expected, [N_a, N_z, 1, N_d, 1]);
        V_cont = permute(EV_expected, [3, 2, 5, 4, 1]); % -> [1, N_z, 1, N_d, N_a]
    end
else
    % Standard invariant expectation mapping
    if has_e
        V_cont = permute(EV, [4, 2, 3, 5, 1]);
    else
        V_cont = permute(EV, [3, 2, 4, 5, 1]);
    end
end

% Fold states: (N_a * N_z * N_e)
% Fold choices: (N_d * N_a)
n_states  = N_a * N_z * N_e;
n_choices = N_d * N_a;
RHS_m     = reshape(BellmanCombiner(F, V_cont), [n_states, n_choices]);
[sub_V, sub_Pol] = max(RHS_m, [], 2);

% Unpack choices
d_opt        = mod(sub_Pol - 1, N_d) + 1;
coarse_a_opt = ceil(sub_Pol ./ N_d);

row1_kron = (coarse_a_opt - 1) .* N_d + d_opt;

if has_e
    V_current  = reshape(sub_V, [N_a, N_z, N_e]);
    Policy_Row = reshape(row1_kron, [N_a, N_z, N_e]);
else
    V_current  = reshape(sub_V, [N_a, N_z]);
    Policy_Row = reshape(row1_kron, [N_a, N_z]);
end


end
