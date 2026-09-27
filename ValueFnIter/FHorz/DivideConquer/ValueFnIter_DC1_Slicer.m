function [V_max, Pol_apr, Pol_d1] = ValueFnIter_DC1_Slicer(N_a1, N_choice, N_a2, N_ze, vfoptions, EvalBlockFn, N_d)
% Universal CPU Divide-and-Conquer (n-Monotonicity) Slicer (Vectorized over N_d)

if nargin < 7; N_d = 1; end

level1ii = round(linspace(1, N_a1, vfoptions.level1n));
num_anchors = length(level1ii);

% Preallocate global outputs (N_d aware)
V_d       = -inf(N_d, N_a1, N_a2, N_ze, 'gpuArray');
Pol_apr_d = ones(N_d, N_a1, N_a2, N_ze, 'gpuArray');

[V_anch, Pol_apr_anch, ~] = EvalBlockFn(level1ii, [], 0);

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

for ii = 1:(num_anchors - 1)
    segment_states = (level1ii(ii) + 1) : (level1ii(ii+1) - 1);
    if isempty(segment_states); continue; end

    num_seg = length(segment_states);

    % Replicates bounds safely across segment states to prevent TensorBlock scrambling
    anchor_slice = reshape(Pol_apr_anch(:, ii, :, :), [N_d, 1, 1, max(1, N_a2), N_ze]);
    loweredge = repmat(anchor_slice, [1, 1, num_seg, 1, 1]);
    loweredge = reshape(loweredge, [N_d, 1, num_seg * max(1, N_a2), N_ze]);

    mg_seg = maxgap(:, ii);
    mg_eval = max(mg_seg);

    % --- DEAD ZONE SURVIVAL PATCH (Zero-Sync Version) ---
    if anch_is_inf(ii) || anch_is_inf(ii+1)
        mg_eval = N_choice - 1;
        loweredge(:) = 1;
    end
    % ----------------------------------------------------

    if mg_eval > 0
        % Cap the evaluation window so it doesn't physically exceed the grid
        mg_eval = min(mg_eval, N_choice - 1);
        loweredge = min(loweredge, N_choice - mg_eval);
    end

    [V_seg, Pol_apr_seg, Pol_d_seg] = EvalBlockFn(segment_states, loweredge(:), mg_eval);

    V_d(:, segment_states, :, :)       = V_seg;
    Pol_apr_d(:, segment_states, :, :) = Pol_apr_seg;
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