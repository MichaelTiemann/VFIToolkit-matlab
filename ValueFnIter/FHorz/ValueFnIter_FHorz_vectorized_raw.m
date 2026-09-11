function [V_current, Policy_Row] = ValueFnIter_FHorz_vectorized_raw(...
    eval_kernel, BellmanCombiner, EV, a_work, z_work, d_work, ...
    N_a, N_z, N_d, vfoptions)

% Check if e exists in the model
has_e = isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0;

% If lowmemory >= 1, the caller loops over e sequentially, so this call only sees 1 slice.
if has_e && vfoptions.lowmemory == 0
    N_e = prod(vfoptions.n_e);
else
    N_e = 1;
end

% Align 5D grid: Dim 1: a, Dim 2: z, Dim 3: e, Dim 4: d, Dim 5: aprime
A_in   = a_work(:);                % Dim 1
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

% EV enters as (N_a x N_z x N_e) or (N_a x N_z). Map to (1, N_z, N_e, 1, N_a)
if has_e
    V_cont = permute(EV, [4, 2, 3, 5, 1]);
else
    V_cont = permute(EV, [3, 2, 4, 5, 1]);
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
