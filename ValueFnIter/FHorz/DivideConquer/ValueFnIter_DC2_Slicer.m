function [V_max, Pol_apr, Pol_d1] = ValueFnIter_DC2_Slicer(N_a1, N_a2prime, N_other_states, N_ze, vfoptions, EvalBlockFn, N_d)
% Universal 2-Asset CPU Divide-and-Conquer (n-Monotonicity) Slicer (Vectorized over N_d)

if nargin < 7; N_d = 1; end

level1ii = round(linspace(1, N_a1, vfoptions.level1n(1)));
num_anchors = length(level1ii);

% 1. Construct full anchor state array
state_chunk_mat_L1 = level1ii(:) + (0:N_other_states-1) * N_a1;

% 2. Evaluate all anchors across all other states (Get all 6 outputs at once)
[V_anch_flat, Pol_apr_anch_flat, ~, ~, ~, p_a1_per_a2_anch] = EvalBlockFn(state_chunk_mat_L1(:)', [], 0);

% 3. Reshape conditional bounds to separate the anchor dimension
% p_a1_per_a2 comes out as [N_d, N_a2prime, N_states_passed, N_ze]
maxindex1 = reshape(p_a1_per_a2_anch, [N_d, N_a2prime, num_anchors, N_other_states, N_ze]);

% Preallocate global outputs (N_d aware)
V_d       = -inf(N_d, N_a1, N_other_states, N_ze, 'gpuArray');
Pol_apr_d = ones(N_d, N_a1, N_other_states, N_ze, 'gpuArray');

% Reshape and assign the Anchor Evaluation
V_d(:, level1ii, :, :)       = reshape(V_anch_flat, [N_d, num_anchors, N_other_states, N_ze]);
Pol_apr_d(:, level1ii, :, :) = reshape(Pol_apr_anch_flat, [N_d, num_anchors, N_other_states, N_ze]);

[V_anch, Pol_apr_anch, ~, ~, ~] = EvalBlockFn(level1ii, [], 0);

V_d(:, level1ii, :, :)       = V_anch;
Pol_apr_d(:, level1ii, :, :) = Pol_apr_anch;

% maxgap is calculated PER D
diff_anch = Pol_apr_anch(:, 2:end, :, :) - Pol_apr_anch(:, 1:end-1, :, :);
maxgap = squeeze(max(max(diff_anch, [], 4), [], 3));
if N_d == 1 && size(maxgap, 1) > 1; maxgap = maxgap'; end

% CRITICAL FIX: Gather maxgap to the CPU to prevent pipeline flushes in the loop below
maxgap = gather(maxgap);

for ii = 1:(num_anchors - 1)
    segment_states = (level1ii(ii) + 1) : (level1ii(ii+1) - 1);
    if isempty(segment_states); continue; end

    state_chunk_mat_seg = segment_states(:) + (0:N_other_states-1) * N_a1;

    % Extract the specific loweredge conditional bounds for this anchor
    mg_eval = maxgap(ii);
    loweredge = min(maxindex1(:, :, ii, :, :), N_a1 - mg_eval);

    % Pass the loweredge matrix and mg_eval to the Block Evaluator
    [V_seg, Pol_apr_seg, Pol_d_seg, ~, ~] = EvalBlockFn(state_chunk_mat_seg(:)', loweredge(:), mg_eval);

    % --- DEAD ZONE SURVIVAL PATCH ---
    V_anch_left = V_d(:, level1ii(ii), :, :);
    V_anch_right = V_d(:, level1ii(ii+1), :, :);
    if any(V_anch_left(:) == -Inf) || any(V_anch_right(:) == -Inf)
        mg_eval = N_a1 - 1;
        loweredge(:) = 1;
    end
    % --------------------------------

    if mg_eval > 0
        % Cap the evaluation window so it doesn't physically exceed the grid
        mg_eval = min(mg_eval, N_a1 - 1);
        loweredge = min(loweredge, N_a1 - mg_eval);

        [V_seg, Pol_apr_seg, ~, ~, ~] = EvalBlockFn(segment_states, loweredge, mg_eval);
    else
        [V_seg, Pol_apr_seg, ~, ~, ~] = EvalBlockFn(segment_states, loweredge, 0);
    end

    V_d(:, segment_states, :, :)       = V_seg;
    Pol_apr_d(:, segment_states, :, :) = Pol_apr_seg;
end

% Collapse N_d Dimension Safely
[V_max, best_d] = max(V_d, [], 1);
V_max = reshape(V_max, [N_a1, max(1, N_other_states), max(1, N_ze)]);

if N_d > 1
    d_stride = N_d;
    a1_stride = d_stride * N_a1;
    a2_stride = a1_stride * max(1, N_other_states);

    [A1_grid, A2_grid, ZE_grid] = ndgrid(1:N_a1, 1:max(1, N_other_states), 1:max(1, N_ze));
    lin_idx = best_d(:) + (A1_grid(:) - 1) * d_stride + (A2_grid(:) - 1) * a1_stride + (ZE_grid(:) - 1) * a2_stride;

    Pol_apr = reshape(Pol_apr_d(lin_idx), [N_a1, max(1, N_other_states), max(1, N_ze)]);
    Pol_d1  = reshape(best_d, [N_a1, max(1, N_other_states), max(1, N_ze)]);
else
    Pol_apr = reshape(Pol_apr_d, [N_a1, max(1, N_other_states), max(1, N_ze)]);
    Pol_d1  = reshape(best_d, [N_a1, max(1, N_other_states), max(1, N_ze)]);
end

if N_a2 == 1
    V_max = reshape(V_max, [N_a1, N_ze]);
    Pol_apr = reshape(Pol_apr, [N_a1, N_ze]);
    Pol_d1 = reshape(Pol_d1, [N_a1, N_ze]);
end


end