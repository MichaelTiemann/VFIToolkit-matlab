function [V_current, Policy_Indices] = ValueFnIter_FHorz_vectorized_DC(eval_func, V_next, a_work, z_work, d_work, n_a, n_z, n_d, pi_z_j, beta_j)

% Precompute expected continuation value: EV_next is (n_a x n_z)
EV_next = V_next * (pi_z_j');

V_current = -Inf(n_a, n_z, 'like', a_work);
Policy_Indices = zeros(n_a, n_z, 'like', a_work);

% Monotonicity applies along the endogenous asset state a
% Lower and upper search bounds for aprime index (1 to n_a)
lower_bound = ones(n_a, n_z, 'like', a_work);
upper_bound = n_a * ones(n_a, n_z, 'like', a_work);

% Multi-level dyadic grid evaluation (Breadth-first D&C)
step = 2^floor(log2(n_a));

while step >= 1
    eval_a_idx = 1:step:n_a;
    n_sub_a = length(eval_a_idx);

    % Unsolved states at current step
    mask_to_solve = isinf(V_current(eval_a_idx, :));

    if any(mask_to_solve(:))
        sub_a = a_work(eval_a_idx);

        % Build bounded state-choice tensor
        [A_m, Z_m, D_m, Apr_m] = ndgrid(sub_a, z_work, d_work, a_work);
        [~, Z_idx_m, ~, Apr_idx_m] = ndgrid(1:n_sub_a, 1:n_z, 1:n_d, 1:n_a);

        A_f = A_m(:);
        Z_f = Z_m(:);
        D_f = D_m(:);
        Apr_f = Apr_m(:);
        Apr_idx_f = Apr_idx_m(:);

        F_f = eval_func(D_f, Apr_f, A_f, Z_f);

        % Vectorized continuation value lookup
        z_idx_state = repelem((1:n_z)', n_sub_a, 1);
        z_idx_f = repmat(z_idx_state, n_d * n_a, 1);
        lin_idx = sub2ind([n_a, n_z], Apr_idx_f, z_idx_f);
        V_cont_f = EV_next(lin_idx);

        RHS_f = F_f + beta_j .* V_cont_f;

        % Apply bounds: Prune choices outside [lower_bound, upper_bound]
        cur_lb = lower_bound(eval_a_idx, :);
        cur_ub = upper_bound(eval_a_idx, :);

        lb_f = repmat(cur_lb(:), n_d * n_a, 1);
        ub_f = repmat(cur_ub(:), n_d * n_a, 1);

        invalid_choice = (Apr_idx_f < lb_f) | (Apr_idx_f > ub_f);
        RHS_f(invalid_choice) = -Inf;

        RHS_m = reshape(RHS_f, [n_sub_a * n_z, n_d * n_a]);
        [sub_V, sub_Pol] = max(RHS_m, [], 2);

        % Store solved states
        sub_V_mat = reshape(sub_V, [n_sub_a, n_z]);
        sub_Pol_mat = reshape(sub_Pol, [n_sub_a, n_z]);

        V_current(eval_a_idx, :) = sub_V_mat;
        Policy_Indices(eval_a_idx, :) = sub_Pol_mat;

        % Extract the optimal aprime index from Kron choice index (1:n_d * n_a)
        % Choice ordering in Kron: d varies fastest, aprime varies slowest
        opt_aprime_idx = ceil(sub_Pol_mat ./ n_d);

        % Update lower and upper bounds for adjacent uncalculated states
        for i = 1:n_sub_a
            idx = eval_a_idx(i);
            opt_apr = opt_aprime_idx(i, :);

            % Downstream propagation: states a' > idx must have astar >= opt_apr
            if idx + 1 <= n_a
                lower_bound(idx+1:end, :) = max(lower_bound(idx+1:end, :), opt_apr);
            end
            % Upstream propagation: states a' < idx must have astar <= opt_apr
            if idx - 1 >= 1
                upper_bound(1:idx-1, :) = min(upper_bound(1:idx-1, :), opt_apr);
            end
        end
    end

    step = floor(step / 2);
end

V_current = V_current(:);
Policy_Indices = Policy_Indices(:);

end
