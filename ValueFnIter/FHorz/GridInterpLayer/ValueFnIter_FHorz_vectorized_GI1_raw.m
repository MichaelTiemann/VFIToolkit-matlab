function [V_current, Policy_3Row] = ValueFnIter_FHorz_vectorized_GI1_raw(...
    eval_kernel, BellmanCombiner, EV, A_mat, z_work, d_work, ...
    N_a, N_z, N_d, pi_z_j, ReturnFnParamsVec, vfoptions)

G = vfoptions.ngridinterp;
a_work = A_mat(:, 1); % Extract primary asset grid
tau_vec = linspace(0, (G - 1) / G, G);
if vfoptions.parallel == 2
    tau_vec = gpuArray(tau_vec);
end

% Dense sub-grid for interpolation: (N_a x G)
a_diff = [diff(a_work); 0];
Apr_dense = a_work + a_diff * tau_vec;

% Dense continuation values: (N_a x G x N_bothz)
EV_pad = [EV; EV(end, :)];
EV_dense_3d = (1 - tau_vec) .* reshape(EV, [N_a, 1, N_z]) + ...
              tau_vec .* reshape(EV_pad(2:end, :), [N_a, 1, N_z]);

% -------------------------------------------------------------------------
% NEW: Choice-Dependent Tensor Contraction for Semi-Exogenous States
% -------------------------------------------------------------------------
if isfield(vfoptions, 'pi_semiz_j_active')
    pi_semiz = vfoptions.pi_semiz_j_active; % [N_semiz_prime, N_semiz, N_d2]
    N_semiz  = prod(vfoptions.n_semiz);
    N_z_exog = N_z / N_semiz;
    N_d2     = size(pi_semiz, 3);
    N_d1     = N_d / N_d2;
    
    % Reshape EV to expose semiz' for matrix multiplication
    EV_reshaped = reshape(EV_dense_3d, [N_a * G, N_semiz, N_z_exog]);
    EV_flat     = reshape(permute(EV_reshaped, [1, 3, 2]), [N_a * G * N_z_exog, N_semiz]);
    
    % Integrate out semiz' for each d2 choice
    EV_new = zeros(N_a * G * N_z_exog, N_semiz, N_d2, 'like', a_work);
    for d2_idx = 1:N_d2
        EV_new(:,:,d2_idx) = EV_flat * pi_semiz(:,:,d2_idx);
    end
    
    % Reconstruct dimensions: [N_a, G, N_semiz, N_z_exog, N_d2]
    EV_new = reshape(EV_new, [N_a, G, N_z_exog, N_semiz, N_d2]);
    EV_new = permute(EV_new, [1, 2, 4, 3, 5]); 
    EV_new = reshape(EV_new, [N_a, G, N_z, N_d2]);
    
    % Expand N_d2 across N_d1 to match the full choice grid (d1 varies fastest)
    EV_expected = repelem(EV_new, 1, 1, 1, N_d1); 
    
    % Map to (1, N_z, 1, N_d, N_a, G) to match F for BellmanCombiner
    EV_expected = reshape(EV_expected, [N_a, G, N_z, 1, N_d, 1]);
    V_cont = permute(EV_expected, [6, 3, 4, 5, 1, 2]);
else
    % Standard invariant expectation mapping
    V_cont = permute(EV_dense_3d, [4, 3, 5, 6, 1, 2]);
end

% Check if e exists and whether this call processes multiple e simultaneously.
% If lowmemory >= 1, the caller loops over e sequentially, so this invocation sees exactly 1 slice (has_e = false).
if isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0 && vfoptions.lowmemory == 0
    has_e = true;
    N_e = prod(vfoptions.n_e);
else
    has_e = false;
    N_e = 1;
end

% Align 6D grid:
% Dim 1: a, Dim 2: z, Dim 3: e, Dim 4: d, Dim 5: aprime_coarse, Dim 6: tau
A_in   = num2cell(A_mat, 1); % Packages all endogenous states natively spanning Dim 1
Z_in   = shiftdim(z_work(:), -1);  % Pushed to Dim 2
D_in   = shiftdim(d_work(:), -3);  % Pushed to Dim 4
Apr_in = shiftdim(Apr_dense, -4);  % Apr_dense is (N_a x G). Pushed to Dims 5 & 6!

F = eval_kernel(D_in, Apr_in, A_in, Z_in);
guard = zeros([N_a, N_z, N_e, N_d, N_a, G], 'like', a_work);
F = F + guard;

z_coords = 1:N_z; 
if isfield(vfoptions, 'pi_semiz_j_active')
    % Choice-dependent EV lookup: index into (a', z, e, d)
    d_coords = shiftdim(1:N_d, -3); % [1, 1, 1, N_d]
    if has_e
        e_coords = shiftdim(1:N_e, -2);
        ev_lin_idx_d = coarse_cand_idx + (z_coords - 1) .* N_a + (e_coords - 1) .* (N_a * N_z) + (d_coords - 1) .* (N_a * N_z * N_e);
        V_cont_bin = reshape(EV_expected(ev_lin_idx_d(:)), [1, N_z, N_e, N_d, n_cand_bin]);
    else
        ev_lin_idx_d = coarse_cand_idx + (z_coords - 1) .* N_a + (d_coords - 1) .* (N_a * N_z);
        V_cont_bin = reshape(EV_expected(ev_lin_idx_d(:)), [1, N_z, 1, N_d, n_cand_bin]);
    end
else
    % Standard invariant EV lookup: index into (a', z, e)
    if has_e
        e_coords = shiftdim(1:N_e, -2);
        ev_lin_idx = coarse_cand_idx + (z_coords - 1) .* N_a + (e_coords - 1) .* (N_a * N_z);
    else
        ev_lin_idx = coarse_cand_idx + (z_coords - 1) .* N_a;
    end
    V_cont_bin = reshape(EV(ev_lin_idx(:)), cand_shape);
end

RHS = BellmanCombiner(F, V_cont);

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