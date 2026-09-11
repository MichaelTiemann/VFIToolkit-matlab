function [V_current, Policy_3Row] = ValueFnIter_FHorz_vectorized_GI1_raw(...
    eval_kernel, ReturnFnParamsVec, EV, a_work, z_work, d_work, ...
    N_a, N_z, N_d, pi_z_j, beta_j, vfoptions)

G = vfoptions.ngridinterp;
tau_vec = linspace(0, (G - 1) / G, G);
if vfoptions.parallel == 2
    tau_vec = gpuArray(tau_vec);
end

% Dense sub-grid for interpolation: (N_a x G)
a_diff = [diff(a_work); 0];
Apr_dense = a_work + a_diff * tau_vec;

% Dense continuation values: (N_a x G x N_z)
EV_pad = [EV; EV(end, :)];
EV_dense_3d = (1 - tau_vec) .* reshape(EV, [N_a, 1, N_z]) + ...
              tau_vec .* reshape(EV_pad(2:end, :), [N_a, 1, N_z]);

% Check if e exists in the model
has_e = isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0;

% If lowmemory >= 1, the caller loops over e sequentially, so this call only sees 1 slice.
% If lowmemory == 0, this call processes all N_e simultaneously.
if has_e && vfoptions.lowmemory == 0
    N_e = prod(vfoptions.n_e);
    state_dims = [N_a, N_z, N_e];
else
    N_e = 1;
    state_dims = [N_a, N_z];
end

% Align 6D grid:
% Dim 1: a, Dim 2: z, Dim 3: e, Dim 4: d, Dim 5: aprime_coarse, Dim 6: tau
% Broadcast shapes (Dim 1: a, Dim 2: z, Dim 4: d, Dim 5: aprime)
A_in   = a_anchors;                % Natively spans Dim 1
Z_in   = shiftdim(z_work(:), -1);  % Pushed to Dim 2
D_in   = shiftdim(d_work(:), -3);  % Pushed to Dim 4
Apr_in = shiftdim(a_work(:), -4);  % Pushed to Dim 5

F = eval_kernel(D_in, Apr_in, A_in, Z_in);
guard = zeros([N_a, N_z, N_e, N_d, N_a, G], 'like', a_work);
F = F + guard;

% Continuation values mapped to (1, N_z, 1, 1, N_a, G)
V_cont = permute(EV_dense_3d, [4, 3, 5, 6, 1, 2]);

RHS = F + beta_j .* V_cont;

% Fold states: (N_a * N_z * N_e)
% Fold choices: (N_d * N_a * G)
n_states  = N_a * N_z * N_e;
n_choices = N_d * N_a * G;
RHS_m     = reshape(RHS, [n_states, n_choices]);
[sub_V, sub_Pol] = max(RHS_m, [], 2);

% Unpack choices
d_opt        = mod(sub_Pol - 1, N_d) + 1;
apr_tau_opt  = ceil(sub_Pol ./ N_d);
coarse_a_opt = mod(apr_tau_opt - 1, N_a) + 1;
tau_opt      = ceil(apr_tau_opt ./ N_a);

% Boundary clamp at upper bound
at_upper = (coarse_a_opt == N_a);
tau_opt(at_upper) = 1;

row1_kron = (coarse_a_opt - 1) .* N_d + d_opt;

if has_e
    V_current   = reshape(sub_V, [N_a, N_z, N_e]);
    Policy_row1 = reshape(row1_kron, [N_a, N_z, N_e]);
    Policy_row2 = reshape(tau_opt, [N_a, N_z, N_e]);

    Policy_3Row = zeros([3, N_a, N_z, N_e], 'like', a_work);
    Policy_3Row(1, :, :, :) = Policy_row1;
    Policy_3Row(2, :, :, :) = Policy_row2;
    Policy_3Row(3, :, :, :) = ones(N_a, N_z, N_e, 'like', a_work);
else
    V_current   = reshape(sub_V, [N_a, N_z]);
    Policy_row1 = reshape(row1_kron, [N_a, N_z]);
    Policy_row2 = reshape(tau_opt, [N_a, N_z]);

    Policy_3Row = zeros([3, N_a, N_z], 'like', a_work);
    Policy_3Row(1, :, :) = Policy_row1;
    Policy_3Row(2, :, :) = Policy_row2;
    Policy_3Row(3, :, :) = ones(N_a, N_z, 'like', a_work);
end

end