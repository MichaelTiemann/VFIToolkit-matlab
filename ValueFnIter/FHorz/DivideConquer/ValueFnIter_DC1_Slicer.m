function [V_max, Pol_apr, Pol_d1, Pol_L2idx, Pol_L2flag] = ValueFnIter_DC1_Slicer(N_a1, N_choice, N_a2, N_ze, vfoptions, EvalBlockFn, N_d)
% Universal CPU Divide-and-Conquer (n-Monotonicity) Slicer (Vectorized over N_d)

if nargin < 7; N_d = 1; end
gridinterplayer = vfoptions.gridinterplayer(1) == 1;

level1ii = round(linspace(1, N_a1, vfoptions.level1n));
num_anchors = length(level1ii);

% Preallocate global outputs (N_d aware)
V_d       = -inf(N_d, N_a1, N_a2, N_ze, 'gpuArray');
Pol_apr_d = ones(N_d, N_a1, N_a2, N_ze, 'gpuArray');

if gridinterplayer
    Pol_L2idx_d  = ones(N_d, N_a1, N_a2, N_ze, 'gpuArray');
    Pol_L2flag_d = 2 * ones(N_d, N_a1, N_a2, N_ze, 'gpuArray');
else
    Pol_L2idx_d = [];
    Pol_L2flag_d = [];
end

[V_anch, Pol_apr_anch, ~, L2idx_anch, L2flag_anch] = EvalBlockFn(level1ii, [], 0);

V_d(:, level1ii, :, :)       = V_anch;
Pol_apr_d(:, level1ii, :, :) = Pol_apr_anch;

if gridinterplayer
    Pol_L2idx_d(:, level1ii, :, :)  = L2idx_anch;
    Pol_L2flag_d(:, level1ii, :, :) = L2flag_anch;
end

% maxgap is calculated PER D
diff_anch = Pol_apr_anch(:, 2:end, :, :) - Pol_apr_anch(:, 1:end-1, :, :);
maxgap = squeeze(max(max(diff_anch, [], 4), [], 3));
if N_d == 1 && size(maxgap, 1) > 1; maxgap = maxgap'; end

% CRITICAL FIX: Gather maxgap to the CPU to prevent pipeline flushes in the loop below
maxgap = gather(maxgap);

for ii = 1:(num_anchors - 1)
    segment_states = (level1ii(ii) + 1) : (level1ii(ii+1) - 1);
    if isempty(segment_states); continue; end

    num_seg = length(segment_states);

    % CRITICAL FIX: ZERO-COST REP-MAT BOUNDS
    % Replicates bounds safely across segment states to prevent TensorBlock scrambling
    anchor_slice = reshape(Pol_apr_anch(:, ii, :, :), [N_d, 1, 1, max(1, N_a2), N_ze]);
    loweredge = repmat(anchor_slice, [1, 1, num_seg, 1, 1]);
    loweredge = reshape(loweredge, [N_d, 1, num_seg * max(1, N_a2), N_ze]);

    % CRITICAL FIX: Zero-Sync CPU Bounds Assignment.
    mg_seg = maxgap(:, ii);
    mg_eval = max(mg_seg);

    if mg_eval > 0
        % Cap the evaluation window so it doesn't physically exceed the grid
        mg_eval = min(mg_eval, N_choice - 1);
        loweredge = min(loweredge, N_choice - mg_eval);

        [V_seg, Pol_apr_seg, ~, L2idx_seg, L2flag_seg] = EvalBlockFn(segment_states, loweredge, mg_eval);
    else
        [V_seg, Pol_apr_seg, ~, L2idx_seg, L2flag_seg] = EvalBlockFn(segment_states, loweredge, 0);
    end

    V_d(:, segment_states, :, :)       = V_seg;
    Pol_apr_d(:, segment_states, :, :) = Pol_apr_seg;

    if gridinterplayer
        Pol_L2idx_d(:, segment_states, :, :)  = L2idx_seg;
        Pol_L2flag_d(:, segment_states, :, :) = L2flag_seg;
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

    if gridinterplayer
        Pol_L2idx  = reshape(Pol_L2idx_d(lin_idx), [N_a1, max(1, N_a2), max(1, N_ze)]);
        Pol_L2flag = reshape(Pol_L2flag_d(lin_idx), [N_a1, max(1, N_a2), max(1, N_ze)]);
    else
        Pol_L2idx = [];
        Pol_L2flag = [];
    end
else
    % Bypass linear indexing if N_d == 1 to save memory and time
    Pol_apr = reshape(Pol_apr_d, [N_a1, max(1, N_a2), max(1, N_ze)]);
    Pol_d1  = reshape(best_d, [N_a1, max(1, N_a2), max(1, N_ze)]);

    if gridinterplayer
        Pol_L2idx  = reshape(Pol_L2idx_d, [N_a1, max(1, N_a2), max(1, N_ze)]);
        Pol_L2flag = reshape(Pol_L2flag_d, [N_a1, max(1, N_a2), max(1, N_ze)]);
    else
        Pol_L2idx = [];
        Pol_L2flag = [];
    end
end

if N_a2 == 1
    V_max = reshape(V_max, [N_a1, N_ze]);
    Pol_apr = reshape(Pol_apr, [N_a1, N_ze]);
    Pol_d1 = reshape(Pol_d1, [N_a1, N_ze]);
    if gridinterplayer
        Pol_L2idx = reshape(Pol_L2idx, [N_a1, N_ze]);
        Pol_L2flag = reshape(Pol_L2flag, [N_a1, N_ze]);
    end
end


end