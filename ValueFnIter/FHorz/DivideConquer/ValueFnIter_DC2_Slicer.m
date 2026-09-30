function [V_max, Pol_apr, Pol_d1] = ValueFnIter_DC2_Slicer(N_a1, N_a2prime, N_other_states, N_ze, vfoptions, EvalBlockFn, N_d)
% Universal 2-Asset CPU Divide-and-Conquer (n-Monotonicity) Slicer (Vectorized over N_d)

if nargin < 7; N_d = 1; end

level1ii = round(linspace(1, N_a1, vfoptions.level1n(1)));
num_anchors = length(level1ii);

% 1. Construct full anchor state array
state_chunk_mat_L1 = level1ii(:) + (0:N_other_states-1) * N_a1;

% 2. Evaluate all anchors across all other states (Get all 6 outputs at once)
[V_anch_flat, Pol_apr_anch_flat, ~, p_a1_per_a2_anch] = EvalBlockFn(state_chunk_mat_L1(:)', [], 0);

% 3. Reshape conditional bounds to separate the anchor dimension
% p_a1_per_a2 comes out as [N_d, N_a2prime, N_states_passed, N_ze]
maxindex1 = reshape(p_a1_per_a2_anch, [N_d, N_a2prime, num_anchors, N_other_states, N_ze]);

% Preallocate global outputs (N_d aware)
V_d       = -inf(N_d, N_a1, N_other_states, N_ze, 'gpuArray');
Pol_apr_d = ones(N_d, N_a1, N_other_states, N_ze, 'gpuArray');

% Reshape and assign the Anchor Evaluation
V_d(:, level1ii, :, :)       = reshape(V_anch_flat, [N_d, num_anchors, N_other_states, N_ze]);
Pol_apr_d(:, level1ii, :, :) = reshape(Pol_apr_anch_flat, [N_d, num_anchors, N_other_states, N_ze]);

% --- CRITICAL FIX 1: Pre-calculate Dead Zone flags for all anchors and pull to CPU
% to absolutely prevent GPU pipeline flushes inside the segment loop.
anch_inf_3d = reshape(V_anch_flat, [N_d, num_anchors, N_other_states * max(1, N_ze)]);
anch_is_inf = gather(any(anch_inf_3d == -Inf, [1, 3]));
anch_is_inf = anch_is_inf(:)'; % Ensure it evaluates as a clean row vector

% Calculate maxgap STRICTLY on the a1prime conditional bounds (maxindex1) to avoid the 2D Global Index Trap
diff_anch = maxindex1(:, :, 2:end, :, :) - maxindex1(:, :, 1:end-1, :, :);

% Collapse all dimensions except the anchor intervals (Dim 3 of maxindex1)
maxgap = squeeze(max(diff_anch, [], [1, 2, 4, 5]));
maxgap = maxgap(:)'; % Ensure it is a row vector

% CRITICAL FIX: Gather maxgap to the CPU to prevent pipeline flushes in the loop below
maxgap = gather(maxgap);

unique_mg = [];
l2_indices_grp = {};
l2_low_grp = {};

for ii = 1:(num_anchors - 1)
    segment_states = (level1ii(ii) + 1) : (level1ii(ii+1) - 1);
    if isempty(segment_states); continue; end

    % Extract the specific loweredge conditional bounds for this anchor
    mg_eval = maxgap(ii);
    loweredge = min(maxindex1(:, :, ii, :, :), N_a1 - mg_eval);

    % --- DEAD ZONE SURVIVAL PATCH (Zero-Sync Version) ---
    if anch_is_inf(ii) || anch_is_inf(ii+1)
        mg_eval = N_a1 - 1;
        loweredge(:) = 1;
    end
    % ----------------------------------------------------

    if mg_eval > 0
        % Cap the evaluation window so it doesn't physically exceed the grid
        mg_eval = min(mg_eval, N_a1 - 1);
        loweredge = min(loweredge, N_a1 - mg_eval);
    end

    % Replicate the bounds across the intermediate segment states
    loweredge_rep = repmat(loweredge, [1, 1, length(segment_states), 1, 1]);

    % Group segments by their exact mg_eval requirement
    grp_idx = find(unique_mg == mg_eval, 1);
    if isempty(grp_idx)
        unique_mg(end+1) = mg_eval;
        l2_indices_grp{end+1} = segment_states(:);
        l2_low_grp{end+1} = loweredge_rep;
    else
        l2_indices_grp{grp_idx} = [l2_indices_grp{grp_idx}; segment_states(:)];
        l2_low_grp{grp_idx} = cat(3, l2_low_grp{grp_idx}, loweredge_rep);
    end
end

% Evaluate each group optimally without padding (Chunked for VRAM safety)
CHUNK_SIZE = 50;
for g = 1:length(unique_mg)
    mg_e = unique_mg(g);
    sub_idx_all = l2_indices_grp{g};
    sub_low_all = l2_low_grp{g};

    for c_start = 1:CHUNK_SIZE:length(sub_idx_all)
        c_end = min(length(sub_idx_all), c_start + CHUNK_SIZE - 1);
        sub_idx = sub_idx_all(c_start:c_end);
        sub_low = sub_low_all(:, :, c_start:c_end, :, :);

        state_chunk_mat_seg = sub_idx(:) + (0:N_other_states-1) * N_a1;

        % Evaluate the chunk
        [V_seg_flat, Pol_apr_seg_flat, ~] = EvalBlockFn(state_chunk_mat_seg(:)', sub_low(:), mg_e);

        % Reshape and assign directly back to the global tensor
        V_d(:, sub_idx, :, :)       = reshape(V_seg_flat, [N_d, length(sub_idx), N_other_states, N_ze]);
        Pol_apr_d(:, sub_idx, :, :) = reshape(Pol_apr_seg_flat, [N_d, length(sub_idx), N_other_states, N_ze]);
    end
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

if N_other_states == 1
    V_max = reshape(V_max, [N_a1, N_ze]);
    Pol_apr = reshape(Pol_apr, [N_a1, N_ze]);
    Pol_d1 = reshape(Pol_d1, [N_a1, N_ze]);
end


end