function [V,Policy]=ValueFnIter_FHorz_raw(n_d,n_a,n_z,N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(1,N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

use_vec_broadcast = isfield(vfoptions,'vectorized') && (vfoptions.vectorized==1) && (N_d > 0);
if use_vec_broadcast
    ReturnFn_Direct = unwrap_passthrough_fn_fhorz(ReturnFn);
    a_gridvals = CreateGridvals(n_a, a_grid, 1);
end

%%
if vfoptions.lowmemory>0
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if use_vec_broadcast
        [F_ref, d_idx] = eval_fhorz_return_refine1_broadcast(ReturnFn_Direct, n_d, n_a, n_z, d_gridvals, a_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp, ap_idx] = max(F_ref, [], 1);
        ap_mat = reshape(ap_idx, [N_a, N_z]);
        lin_idx = ap_mat + N_a * ((0:N_a-1)' + N_a * (0:N_z-1));
        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d_idx(lin_idx) + N_d * (ap_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,N_j) = v_j;
        Policy(1,:,:,N_j) = pol_j;

    elseif vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1); % Refine=1
        [F_ref, d_idx]=max(ReturnMatrix,[],1);
        F_ref=reshape(F_ref,[N_a,N_a,N_z]);
        d_idx=reshape(d_idx,[N_a,N_a,N_z]);
        %Calc the max and it's index
        [Vtemp,ap_idx]=max(F_ref,[],1);
        ap_mat = reshape(ap_idx, [N_a, N_z]);
        lin_idx = ap_mat + N_a * ((0:N_a-1)' + N_a * (0:N_z-1));
        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d_idx(lin_idx) + N_d * (ap_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,N_j)=v_j;
        Policy(1,:,:,N_j)=pol_j;

    elseif vfoptions.lowmemory==1

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,0);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(1,:,z_c,N_j)=maxindex;
        end

    end
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]);    % First, switch V_Jplus1 into Kron form

    EVinf=(EV==-Inf);
    EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
    EV=EV*pi_z_J(:,:,N_j)';
    EV(EVinf*(pi_z_J(:,:,N_j)'>0)>0)=-Inf; % exact -Inf restoration
    EV=reshape(EV,[N_a,1,N_z]);

    if use_vec_broadcast
        [F_ref, d_idx] = eval_fhorz_return_refine1_broadcast(ReturnFn_Direct, n_d, n_a, n_z, d_gridvals, a_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        entireRHS = F_ref + DiscountFactorParamsVec * EV; % autofill a for EV
        [Vtemp, ap_idx] = max(entireRHS, [], 1);
        ap_mat = reshape(ap_idx, [N_a, N_z]);
        lin_idx = ap_mat + N_a * ((0:N_a-1)' + N_a * (0:N_z-1));
        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d_idx(lin_idx) + N_d * (ap_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,N_j) = v_j;
        Policy(1,:,:,N_j) = pol_j;

    elseif vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1); % Refine=1
        [F_ref, d_idx]=max(ReturnMatrix,[],1);
        F_ref=reshape(F_ref,[N_a,N_a,N_z]);
        d_idx=reshape(d_idx,[N_a,N_a,N_z]);

        entireRHS=F_ref+DiscountFactorParamsVec*EV; % autofill a for EV

        %Calc the max and it's index
        [Vtemp,ap_idx]=max(entireRHS,[],1);
        ap_mat = reshape(ap_idx, [N_a, N_z]);
        lin_idx = ap_mat + N_a * ((0:N_a-1)' + N_a * (0:N_z-1));

        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d_idx(lin_idx) + N_d * (ap_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,N_j)=v_j;
        Policy(1,:,:,N_j)=pol_j;

    elseif vfoptions.lowmemory==1
        entireEV=repelem(EV,N_d,1,1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,0);

            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*entireEV_z; % autofill a for EV

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(1,:,z_c,N_j)=maxindex;
        end

    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=V(:,:,jj+1);

    EVinf=(EV==-Inf);
    EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
    EV=EV*pi_z_J(:,:,jj)';
    EV(EVinf*(pi_z_J(:,:,jj)'>0)>0)=-Inf; % exact -Inf restoration
    EV=reshape(EV,[N_a,1,N_z]);

    if use_vec_broadcast
        [F_ref, d_idx] = eval_fhorz_return_refine1_broadcast(ReturnFn_Direct, n_d, n_a, n_z, d_gridvals, a_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec);
        entireRHS = F_ref + DiscountFactorParamsVec * EV; % autofill a for EV
        [Vtemp, ap_idx] = max(entireRHS, [], 1);
        ap_mat = reshape(ap_idx, [N_a, N_z]);
        lin_idx = ap_mat + N_a * ((0:N_a-1)' + N_a * (0:N_z-1));
        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d_idx(lin_idx) + N_d * (ap_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,jj) = v_j;
        Policy(1,:,:,jj) = pol_j;

    elseif vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1); % Refine=1
        [F_ref, d_idx]=max(ReturnMatrix,[],1);
        F_ref=reshape(F_ref,[N_a,N_a,N_z]);
        d_idx=reshape(d_idx,[N_a,N_a,N_z]);

        entireRHS=F_ref+DiscountFactorParamsVec*EV; % autofill a for EV

        % Calc the max and it's index
        [Vtemp,ap_idx]=max(entireRHS,[],1);
        ap_mat = reshape(ap_idx, [N_a, N_z]);
        lin_idx = ap_mat + N_a * ((0:N_a-1)' + N_a * (0:N_z-1));

        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d_idx(lin_idx) + N_d * (ap_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,jj)=v_j;
        Policy(1,:,:,jj)=pol_j;

    elseif vfoptions.lowmemory==1
        entireEV=repelem(EV,N_d,1,1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_z, d_gridvals, a_grid, z_val, ReturnFnParamsVec,0);
            % (d,aprime,a)

            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*entireEV_z; % autofill a for EV

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(1,:,z_c,jj)=maxindex;
        end
    end
end

end

%% ---- LOCAL HELPERS FOR TILED GPU BROADCAST + FUSED REFINE=1 ----
function [F_max, F_idx] = eval_fhorz_return_refine1_broadcast(ReturnFn, n_d, n_a, n_z, d_gridvals, a_gridvals, z_gridvals, ReturnFnParamsVec)
% Evaluates dual-compatible Case1 ReturnFn via gpuArray implicit expansion in ~2 GB tiles
% along a, reducing max(..., [], 1) over N_d inside each tile to minimize VRAM traffic.
N_d = prod(n_d);
N_a = prod(n_a);
N_z = prod(n_z);

l_d = length(n_d);
l_a = length(n_a);
l_z = length(n_z);

pcell = num2cell(ReturnFnParamsVec)';

d_args = cell(1, l_d);
for k = 1:l_d, d_args{k} = d_gridvals(:, k); end

ap_args = cell(1, l_a);
for k = 1:l_a, ap_args{k} = shiftdim(a_gridvals(:, k), -1); end

z_args = cell(1, l_z);
for k = 1:l_z, z_args{k} = shiftdim(z_gridvals(:, k), -3); end

F_max = zeros(N_a, N_a, N_z, 'gpuArray');
F_idx = zeros(N_a, N_a, N_z, 'gpuArray');

pts_per_a = max(1, N_d * N_a * N_z);
blk_sz    = max(1, min(N_a, floor(268435456 / pts_per_a)));

a_args = cell(1, l_a);
for s = 1:blk_sz:N_a
    idx = s:min(N_a, s + blk_sz - 1);
    nb  = numel(idx);
    for k = 1:l_a
        a_args{k} = shiftdim(a_gridvals(idx, k), -2);
    end
    Fb = ReturnFn(d_args{:}, ap_args{:}, a_args{:}, z_args{:}, pcell{:});
    target_sz = [N_d, N_a, nb, N_z];
    cur_sz    = [size(Fb,1), size(Fb,2), size(Fb,3), size(Fb,4)];
    if ~isequal(cur_sz, target_sz)
        rep_fac = ones(1, 4);
        for dim = 1:4
            if cur_sz(dim) == 1 && target_sz(dim) > 1
                rep_fac(dim) = target_sz(dim);
            end
        end
        [m_b, i_b] = max(Fb, [], 1);
        if any(rep_fac(2:4) > 1)
            m_b = repmat(m_b, [1, rep_fac(2), rep_fac(3), rep_fac(4)]);
            i_b = repmat(i_b, [1, rep_fac(2), rep_fac(3), rep_fac(4)]);
        end
    else
        [m_b, i_b] = max(Fb, [], 1);
    end
    F_max(:, idx, :) = reshape(m_b, [N_a, nb, N_z]);
    F_idx(:, idx, :) = reshape(i_b, [N_a, nb, N_z]);
end
end

function fn_out = unwrap_passthrough_fn_fhorz(fn_in)
% Unwraps @(x1,x2,...) target_fn(x1,x2,...) to @target_fn when arguments pass 1:1 in order.
fn_out = fn_in;
if ischar(fn_in) || isstring(fn_in)
    fn_out = str2func(char(fn_in));
    return;
end
if ~isa(fn_in, 'function_handle'), return; end
s = func2str(fn_in);
if isempty(s) || s(1) ~= '@', return; end
tok = regexp(s, '^@\s*\(([^)]*)\)\s*([A-Za-z]\w*)\s*\(([^)]*)\)\s*$', 'tokens', 'once');
if numel(tok) == 3
    lhs = regexprep(tok{1}, '\s+', '');
    rhs = regexprep(tok{3}, '\s+', '');
    if strcmp(lhs, rhs) && ~isempty(lhs)
        fn_out = str2func(tok{2});
    end
end


end
