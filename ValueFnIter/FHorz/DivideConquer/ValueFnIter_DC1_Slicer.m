function [V_max, Pol_apr, Pol_d1] = ValueFnIter_DC1_Slicer(N_a1, N_choice, N_a2, N_ze, vfoptions, EvalBlockFn, N_d)
% Universal CPU Divide-and-Conquer (n-Monotonicity) Slicer (Vectorized over N_d)

if nargin < 7; N_d = 1; end

level1ii = round(linspace(1, N_a1, vfoptions.level1n));
num_anchors = length(level1ii);

% Preallocate global outputs (N_d aware)
V_d       = -inf(N_d, N_a1, N_a2, N_ze, 'gpuArray');
Pol_apr_d = ones(N_d, N_a1, N_a2, N_ze, 'gpuArray');

% 1. Construct full anchor state array across all N_a2 states
state_chunk_mat_L1 = level1ii(:) + (0:max(1, N_a2)-1) * N_a1;

[V_anch_flat, Pol_apr_anch_flat, ~] = EvalBlockFn(state_chunk_mat_L1(:)', [], 0);

V_anch = reshape(V_anch_flat, [N_d, num_anchors, max(1, N_a2), N_ze]);
Pol_apr_anch = reshape(Pol_apr_anch_flat, [N_d, num_anchors, max(1, N_a2), N_ze]);

V_d(:, level1ii, :, :)       = V_anch;
Pol_apr_d(:, level1ii, :, :) = Pol_apr_anch;

anch_inf_3d = reshape(V_anch, [N_d, num_anchors, max(1, N_a2) * max(1, N_ze)]);
anch_is_inf = gather(any(anch_inf_3d == -Inf, [1, 3]));
anch_is_inf = anch_is_inf(:)'; % Ensure it evaluates as a clean row vector

% maxgap is calculated PER D
diff_anch = Pol_apr_anch(:, 2:end, :, :) - Pol_apr_anch(:, 1:end-1, :, :);
maxgap = squeeze(max(max(diff_anch, [], 4), [], 3));
if N_d == 1 && size(maxgap, 1) > 1; maxgap = maxgap'; end

% Gather maxgap to the CPU to prevent pipeline flushes in the loop below
maxgap = gather(maxgap);

unique_mg = [];
l2_indices_grp = {};
l2_low_grp = {};

for ii = 1:(num_anchors - 1)
    segment_states = (level1ii(ii) + 1) : (level1ii(ii+1) - 1);
    if isempty(segment_states); continue; end

    num_seg = length(segment_states);
    anchor_slice = reshape(Pol_apr_anch(:, ii, :, :), [N_d, 1, 1, max(1, N_a2), N_ze]);
    loweredge = repmat(anchor_slice, [1, 1, num_seg, 1, 1]);
    loweredge = reshape(loweredge, [N_d, 1, num_seg * max(1, N_a2), N_ze]);

    mg_eval = max(maxgap(:, ii));

    % --- DEAD ZONE SURVIVAL PATCH ---
    if anch_is_inf(ii) || anch_is_inf(ii+1)
        mg_eval = N_choice - 1;
        loweredge(:) = 1;
    end

    if mg_eval > 0
        mg_eval = min(mg_eval, N_choice - 1);
        loweredge = min(loweredge, N_choice - mg_eval);
    end

    grp_idx = find(unique_mg == mg_eval, 1);
    if isempty(grp_idx)
        unique_mg(end+1) = mg_eval;
        l2_indices_grp{end+1} = segment_states(:);
        l2_low_grp{end+1} = loweredge;
    else
        l2_indices_grp{grp_idx} = [l2_indices_grp{grp_idx}; segment_states(:)];
        l2_low_grp{grp_idx} = cat(3, l2_low_grp{grp_idx}, loweredge);
    end
end

if vfoptions.parallel == 2
    gpu_device_info = gpuDevice();
    safe_elements = max(1e7, floor((gpu_device_info.AvailableMemory / 8) / 8));
else
    safe_elements = 50000000;
end

for g = 1:length(unique_mg)
    mg_e = unique_mg(g);
    flat_choices_L2 = max(1, N_d) * (mg_e + 1);
    CHUNK_SIZE = max(1, floor(safe_elements / (flat_choices_L2 * max(1, N_a2) * max(1, N_ze))));

    sub_idx_all = l2_indices_grp{g};
    sub_low_all = l2_low_grp{g};

    for c_start = 1:CHUNK_SIZE:length(sub_idx_all)
        c_end = min(length(sub_idx_all), c_start + CHUNK_SIZE - 1);
        sub_idx = sub_idx_all(c_start:c_end);
        sub_low = sub_low_all(:, :, c_start:c_end, :, :);

        state_chunk_mat_seg = sub_idx(:) + (0:max(1, N_a2)-1) * N_a1;
        [V_seg_flat, Pol_apr_seg_flat, ~] = EvalBlockFn(state_chunk_mat_seg(:)', sub_low(:), mg_e);

        V_d(:, sub_idx, :, :)       = reshape(V_seg_flat, [N_d, length(sub_idx), max(1, N_a2), N_ze]);
        Pol_apr_d(:, sub_idx, :, :) = reshape(Pol_apr_seg_flat, [N_d, length(sub_idx), max(1, N_a2), N_ze]);
    end
end

% Collapse N_d Dimension Safely
[V_max, best_d] = max(V_d, [], 1);
V_max = reshape(V_max, [N_a1, max(1, N_a2), max(1, N_ze)]);

if N_d > 1
    d_stride = N_d;
    a1_stride = d_stride * N_a1;
    a2_stride = a1_stride * max(1, N_a2);

    [A1_grid, A2_grid, ZE_grid] = ndgrid(1:N_a1, 1:max(1, N_a2), 1:max(1, N_ze));
    lin_idx = best_d(:) + (A1_grid(:) - 1) * d_stride + (A2_grid(:) - 1) * a1_stride + (ZE_grid(:) - 1) * a2_stride;

    Pol_apr = reshape(Pol_apr_d(lin_idx), [N_a1, max(1, N_a2), max(1, N_ze)]);
    Pol_d1  = reshape(best_d, [N_a1, max(1, N_a2), max(1, N_ze)]);
else
    % Bypass linear indexing if N_d == 1 to save memory and time
    Pol_apr = reshape(Pol_apr_d, [N_a1, max(1, N_a2), max(1, N_ze)]);
    Pol_d1  = reshape(best_d, [N_a1, max(1, N_a2), max(1, N_ze)]);
end

if N_a2 == 1
    V_max = reshape(V_max, [N_a1, N_ze]);
    Pol_apr = reshape(Pol_apr, [N_a1, N_ze]);
    Pol_d1 = reshape(Pol_d1, [N_a1, N_ze]);
end


end