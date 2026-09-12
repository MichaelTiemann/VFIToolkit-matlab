function [V_current, Policy_Row] = ValueFnIter_FHorz_vectorized_DC1(...
    eval_kernel, BellmanCombiner, EV, A_mat, z_work, d_work, ...
    N_a, N_z, N_d, pi_z_j, vfoptions)

% Check if e exists and whether this call processes multiple e simultaneously.
% If lowmemory >= 1, the caller loops over e sequentially, so this invocation sees exactly 1 slice (has_e = false).
if isfield(vfoptions, 'n_e') && ~isempty(vfoptions.n_e) && prod(vfoptions.n_e) > 0 && vfoptions.lowmemory == 0
    has_e = true;
    N_e = prod(vfoptions.n_e);
    state_dims = [N_a, N_z, N_e];
else
    has_e = false;
    N_e = 1;
    state_dims = [N_a, N_z];
end

a_work = A_mat(:, 1);
V_current  = zeros(state_dims, 'like', a_work);
Policy_Row = zeros(state_dims, 'like', a_work);

%% =========================================================================
% PASS 1: Coarse Anchors across N_a choices
% =========================================================================
level1ii  = round(linspace(1, N_a, vfoptions.level1n(1)));
n_anchors = length(level1ii);

% Broadcast shapes (Dim 1: a, Dim 2: z, Dim 4: d, Dim 5: aprime)
% For the raw/full-grid evaluators:
A_1 = num2cell(A_mat, 1);
Z_1   = shiftdim(z_work(:), -1);  % Pushed to Dim 2
D_1   = shiftdim(d_work(:), -3);  % Pushed to Dim 4
Apr_1 = shiftdim(a_work(:), -4);  % Pushed to Dim 5

F_1 = eval_kernel(D_1, Apr_1, A_1, Z_1);

% Zero-overhead shape guard
expected_sz1 = [n_anchors, N_z, N_e, N_d, N_a];
if ~isequal(size(F_1), expected_sz1)
    F_1 = F_1 + zeros(expected_sz1, 'like', a_work);
end

% Ensure EV has explicit 2D shape [N_a, N_z]
EV = reshape(EV, [N_a, N_z]);

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

RHS_1 = BellmanCombiner(F_1, V_cont);

n_states_1  = n_anchors * N_z * N_e;
n_choices_1 = N_d * N_a;
RHS_m1 = reshape(RHS_1, [n_states_1, n_choices_1]);
[sub_V1, sub_Pol1] = max(RHS_m1, [], 2);

coarse_apr_opt1 = ceil(sub_Pol1 ./ N_d);

if has_e
    opt_coarse_anchors = reshape(coarse_apr_opt1, [n_anchors, N_z, N_e]);
    V_current(level1ii, :, :)  = reshape(sub_V1, [n_anchors, N_z, N_e]);
    Policy_Row(level1ii, :, :) = reshape(sub_Pol1, [n_anchors, N_z, N_e]);
else
    opt_coarse_anchors = reshape(coarse_apr_opt1, [n_anchors, N_z]);
    V_current(level1ii, :)  = reshape(sub_V1, [n_anchors, N_z]);
    Policy_Row(level1ii, :) = reshape(sub_Pol1, [n_anchors, N_z]);
end

%% =========================================================================
% PASS 2: Monotonic Bins on Coarse Grid
% =========================================================================
D_2 = shiftdim(d_work(:), -3);    % Dim 4
Z_2 = shiftdim(z_work(:), -1);    % Dim 2

for bin = 1:(n_anchors - 1)
    idx_start = level1ii(bin) + 1;
    idx_end   = level1ii(bin + 1) - 1;
    if idx_start > idx_end
        continue;
    end

    bin_a_idx = idx_start:idx_end;
    n_bin_a   = length(bin_a_idx);

    if has_e
        lb_bin = opt_coarse_anchors(bin, :, :);
        ub_bin = opt_coarse_anchors(bin + 1, :, :);
    else
        lb_bin = opt_coarse_anchors(bin, :);
        ub_bin = opt_coarse_anchors(bin + 1, :);
    end

    lb_bin_pad = max(1, lb_bin - 1);
    ub_bin_pad = min(N_a, ub_bin + 1);
    
    maxgap_bin = max(ub_bin_pad(:) - lb_bin_pad(:));
    n_cand_bin = maxgap_bin + 1;
    
    % k_offsets along Dim 5: size [1, 1, 1, 1, n_cand_bin]
    k_offsets = shiftdim((0:maxgap_bin)', -4); 
    
    % coarse_cand_idx: size [1, N_z, N_e, 1, n_cand_bin]
    coarse_cand_idx = min(lb_bin_pad + k_offsets, ub_bin_pad);

    % Define the explicit 5D candidate shape: [1, N_z, N_e, 1, n_cand_bin]
    cand_shape = [1, N_z, N_e, 1, n_cand_bin];

    % 1. Force candidate asset choices to stay strictly along Dim 5
    Apr_val_bin = reshape(a_work(coarse_cand_idx(:)), cand_shape);
    A_bin = num2cell(A_mat(bin_a_idx, :), 1); % Packages all endogenous states for this bin
    F_bin = eval_kernel(D_2, Apr_val_bin, A_bin, Z_2);

    % Zero-overhead shape guard
    expected_sz_bin = [n_bin_a, N_z, N_e, N_d, n_cand_bin];
    if ~isequal(size(F_bin), expected_sz_bin)
        F_bin = F_bin + zeros(expected_sz_bin, 'like', a_work);
    end

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

    RHS_bin = BellmanCombiner(F_bin, V_cont_bin);

    % Fold choices: (N_d * n_cand_bin)
    n_states_bin  = n_bin_a * N_z * N_e;
    n_choices_bin = N_d * n_cand_bin;
    RHS_m_bin     = reshape(RHS_bin, [n_states_bin, n_choices_bin]);
    [sub_V_bin, sub_Pol_bin] = max(RHS_m_bin, [], 2);

    % Unpack optimal decision d and fine asset index
    d_chosen          = mod(sub_Pol_bin - 1, N_d) + 1;
    coarse_offset_opt = ceil(sub_Pol_bin ./ N_d);

    coarse_cand_expanded = repmat(coarse_cand_idx, [n_bin_a, 1, 1, 1, 1]);
    coarse_cand_2d       = reshape(coarse_cand_expanded, [n_states_bin, n_cand_bin]);
    state_lin            = (1:n_states_bin)';
    chosen_lin           = sub2ind([n_states_bin, n_cand_bin], state_lin, coarse_offset_opt);
    coarse_apr_bin       = coarse_cand_2d(chosen_lin);

    % Form 1-based Kron policy index
    Pol_sub = (coarse_apr_bin - 1) .* N_d + d_chosen;
    V_sub   = sub_V_bin;

    % Store directly into period containers
    if has_e
        V_current(bin_a_idx, :, :)  = reshape(V_sub,   [n_bin_a, N_z, N_e]);
        Policy_Row(bin_a_idx, :, :) = reshape(Pol_sub, [n_bin_a, N_z, N_e]);
    else
        V_current(bin_a_idx, :)     = reshape(V_sub,   [n_bin_a, N_z]);
        Policy_Row(bin_a_idx, :)    = reshape(Pol_sub, [n_bin_a, N_z]);
    end
end

end
