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
    Pol_L2idx_d = []; Pol_L2flag_d = [];
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

for ii = 1:(num_anchors - 1)
    segment_states = (level1ii(ii) + 1) : (level1ii(ii+1) - 1);
    if isempty(segment_states); continue; end

    mg_seg = maxgap(:, ii);
    mg_max = max(mg_seg);

    if mg_max > 0
        loweredge = min(Pol_apr_anch(:, ii, :, :), N_choice - mg_max);
        [V_seg, Pol_apr_seg, ~, L2idx_seg, L2flag_seg] = EvalBlockFn(segment_states, loweredge, mg_max);
    else
        loweredge = Pol_apr_anch(:, ii, :, :);
        [V_seg, Pol_apr_seg, ~, L2idx_seg, L2flag_seg] = EvalBlockFn(segment_states, loweredge, 0);
    end

    V_d(:, segment_states, :, :)       = V_seg;
    Pol_apr_d(:, segment_states, :, :) = Pol_apr_seg;
    if gridinterplayer
        Pol_L2idx_d(:, segment_states, :, :)  = L2idx_seg;
        Pol_L2flag_d(:, segment_states, :, :) = L2flag_seg;
    end
end

% Collapse N_d Dimension
[V_max, best_d] = max(V_d, [], 1);
V_max = squeeze(V_max);
best_d = squeeze(best_d);

[A1_grid, A2_grid, ZE_grid] = ndgrid(1:N_a1, 1:max(1, N_a2), 1:max(1, N_ze));
lin_idx = sub2ind([N_d, N_a1, max(1, N_a2), max(1, N_ze)], best_d(:), A1_grid(:), A2_grid(:), ZE_grid(:));

Pol_apr = reshape(Pol_apr_d(lin_idx), [N_a1, max(1, N_a2), max(1, N_ze)]);
Pol_d1  = reshape(best_d, [N_a1, max(1, N_a2), max(1, N_ze)]);
if gridinterplayer
    Pol_L2idx  = reshape(Pol_L2idx_d(lin_idx), [N_a1, max(1, N_a2), max(1, N_ze)]);
    Pol_L2flag = reshape(Pol_L2flag_d(lin_idx), [N_a1, max(1, N_a2), max(1, N_ze)]);
else
    Pol_L2idx = []; Pol_L2flag = [];
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
