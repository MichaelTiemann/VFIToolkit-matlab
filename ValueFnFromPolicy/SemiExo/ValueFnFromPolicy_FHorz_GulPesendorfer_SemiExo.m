function varargout=ValueFnFromPolicy_FHorz_GulPesendorfer_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,pi_z_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions)
% Gul-Pesendorfer variant of ValueFnFromPolicy_FHorz_SemiExo: values the given Policy under
%   V_j = u(policy_j) + v(policy_j) - MostTempting_j + beta*E[V_{j+1} at policy_j]
% (no continuation term at j=N_j), where u is the ReturnFn, v is the temptation fn
% (vfoptions.temptationFn, same input signature convention as the ReturnFn, own parameters),
% and MostTempting_j(a,bothz) is the max of v over the FULL joint (d1,d2,aprime) choice set.
% The max of v is collected per d2 (special_n_d/d12_gridvals slicing, exactly as the solver
% raws) and then maxed over d2; the semiz transition pi_semiz for the continuation depends on
% the policy's d_semiz choice (last l_dsemiz components of d), exactly as in the core
% ValueFnFromPolicy_FHorz_SemiExo.
%
% Under vfoptions.gridinterplayer==1 the aprime choice set is the FINE grid, so per d2 the
% max of v is found by the same two-stage scheme as in the GP GI solver raws: around v's OWN
% coarse argmax (otherwise the chosen fine point could be more tempting than the coarse max
% of v, making the self-control cost negative).
%
% Two endogenous states are supported: under gridinterplayer the interpolation applies to
% a1prime only (a2prime stays on the coarse grid; its block offset is folded into the linear
% aprime index for the continuation lookup), and the per-d2 fine max of v is over the fine
% a1prime window jointly with the FULL a2prime grid (GI2A, as in the GP SemiExo GI2A raws).
%
% This is dispatched from ValueFnFromPolicy_FHorz_GulPesendorfer AFTER the parent
% ValueFnFromPolicy_FHorz has run ExogShockSetup_FHorz, so z_gridvals_J/pi_z_J and
% vfoptions.e_gridvals_J/pi_e_J arrive pre-processed (vfoptions.n_e exists, 0-equivalent when
% there are no e variables). Only the semiz setup runs here. Handles gridinterplayer itself.

%% Setup
% Need semiz gridvals and pi_semiz: SemiExogShockSetup_FHorz populates vfoptions.semiz_gridvals_J and vfoptions.pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end

n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);

% l_dsemiz: number of d variables that affect the semi-exogenous state (last l_dsemiz of d)
if isfield(vfoptions,'l_dsemiz')
    l_dsemiz=vfoptions.l_dsemiz;
else
    l_dsemiz=1;
end
n_dsemiz=n_d(end-l_dsemiz+1:end);
N_dsemiz=prod(n_dsemiz);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_GulPesendorfer_SemiExo: SemiExo requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);
l_aprime=l_a;

TemptationFn=vfoptions.temptationFn;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
TemptationFnParamNames=ReturnFnParamNamesFn(TemptationFn,n_d,n_a,n_z,N_j,vfoptions,Parameters); % the temptation fn has the same leading model args as the return fn (semiz counted via vfoptions.n_semiz), so the same convention applies

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

% Per-d2 temptation machinery (mirrors the solver raws): d1 are the leading, d2=dsemiz the trailing d variables
n_d1=n_d(1:end-l_dsemiz); % possibly empty (nod1)
N_d1=prod(n_d1); % prod of empty is 1, which is what the per-d2 creator calls need for nod1
d_gridvals=CreateGridvals(n_d,d_grid,1); % [N_d, l_d]: d1 varies fastest, so rows are [repmat(d1_gridvals,N_dsemiz,1),repelem(d2_gridvals,N_d1,1)]
special_n_d=[n_d1,ones(1,l_dsemiz)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_dsemiz,l_d]),[1,3,2]); % [N_d1,l_d,N_dsemiz]: version to use when looping over d2

% Treat (semiz, z) as joint shock for PolicyInd2Val and ReturnFn/TemptationFn evaluation
if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*max(N_z,1);

%% Joint shock gridvals for ReturnFn/TemptationFn
if N_z==0
    joint_gridvals_J=semiz_gridvals_J; % [N_semiz, length(n_semiz), N_j]
else
    joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
    for jj=1:N_j
        joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
    end
end

% Preallocate the per-d2 most-tempting collector
if N_e==0
    MostTempting_ford2=zeros(N_a,N_shocks,N_dsemiz,'gpuArray');
else
    MostTempting_ford2=zeros(N_a,N_shocks,N_e,N_dsemiz,'gpuArray');
end

%%
if vfoptions.gridinterplayer==1

    % Grid interpolation parameters
    n2short=vfoptions.ngridinterp; % evenly spaced points between each pair of a_grid points
    if isscalar(n_a)
        aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
    else % GI2A: the interpolation layer applies to a1prime only, so the fine grid is a1-only (a2prime stays on the coarse grid)
        n_a1=n_a(1);
        a1_grid=a_grid(1:n_a1);
        a2_grid=a_grid(n_a1+1:end);
        a1prime_grid=interp1(1:1:n_a1,a1_grid,linspace(1,n_a1,n_a1+(n_a1-1)*n2short))';
    end

    %% PolicyValues (PolicyInd2Val handles GI internally; returns interpolated aprime values)
    if N_e==0
        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_semiz and vfoptions.n_e
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_shocks, l_d+l_aprime, N_j]
    else
        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_semiz and vfoptions.n_e
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_shocks*N_e, l_d+l_aprime, N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
    end
    l_daprime=size(PolicyValues,1);

    %% Extract per-state indices from Policy: d_semiz_idx, aprime lower index and L2 weight
    % Strip trailing L2flag channel if present (Policy may carry it; we only need l_d+l_aprime+1 channels)
    if size(Policy,1) > (l_d+l_aprime+1)
        tempsize=size(Policy);
        Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
        Policy=reshape(Policy(1:l_d+l_aprime+1,:), [(l_d+l_aprime+1), tempsize(2:end)]);
    end

    % Reshape Policy to Kron form
    if N_e==0
        Policy_k=reshape(Policy,[l_d+l_aprime+1, N_a, N_shocks, N_j]);
    else
        Policy_k=reshape(Policy,[l_d+l_aprime+1, N_a, N_shocks, N_e, N_j]);
    end

    % d_semiz_idx: last l_dsemiz components of d
    if N_e==0
        d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
    else
        d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    end
    cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
    for ii=1:l_dsemiz
        comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
        d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
    end

    % aprime: position l_d+1 is the a1 lower grid index; position l_d+l_aprime+1 is L2_idx
    % ValueFnIter converts the midpoint to the lower grid index before returning Policy (the adjust
    % block at the end of the GI raws), so this row is the lower index and not the midpoint.
    a1_lowerind=shiftdim(Policy_k(l_d+1,:,:,:,:),1); % [N_a, N_shocks, N_j] or [N_a, N_shocks, N_e, N_j]
    L2_idx=shiftdim(Policy_k(l_d+l_aprime+1,:,:,:,:),1);

    % Build a1 fine index: a1_fine = (n2short+1)*(a1_lowerind-1) + L2_idx
    a1_fine_idx=(n2short+1)*(a1_lowerind-1)+L2_idx;

    % Convert a1_fine_idx to fractional position in original a1 grid: frac = 1 + (a1_fine_idx-1)/(n2short+1)
    a1_frac=1+(a1_fine_idx-1)/(n2short+1);
    a1_lower=floor(a1_frac);
    a1_weight=a1_frac-a1_lower; % weight on upper point
    % a1_frac is a1_lowerind+(L2_idx-1)/(n2short+1), so floor gives a1_lowerind while L2_idx<=n2short+1,
    % and a1_lowerind+1 with zero weight at L2_idx==n2short+2 (which is the upper grid point). Both correct.
    % Clamp upper to n_a(1)
    a1_upper=min(a1_lower+1, n_a(1));
    % When at the top exactly (a1_frac == n_a(1)), weight should be 0 and upper=lower
    a1_upper(a1_lower>=n_a(1))=n_a(1);
    a1_lower(a1_lower<1)=1;

    if isscalar(n_a)
        aprime_lower_idx=a1_lower;
        aprime_upper_idx=a1_upper;
    else % GI2A: fold the a2prime block offset (Policy channel l_d+2) into the linear aprime index; the a1
        % interpolation happens WITHIN the chosen a2prime block, and a1_upper is clamped to n_a(1) above so
        % upper=lower+1 never steps outside the block
        a2prime_idx=shiftdim(Policy_k(l_d+2,:,:,:,:),1);
        aprime_lower_idx=a1_lower+n_a(1)*(a2prime_idx-1);
        aprime_upper_idx=a1_upper+n_a(1)*(a2prime_idx-1);
    end

    %% Backward iteration
    if N_e==0
        V=zeros(N_a, N_shocks, N_j, 'gpuArray');
    else
        V=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
    end

    [~, SZ_grid_noz]=ndgrid(1:N_a, 1:N_semiz);
    if N_z>0
        [~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);
    end

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        % Evaluate ReturnFn and TemptationFn at policy
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
        if N_e==0
            F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
            TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
        else
            F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
            TofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
        end

        % Most-tempting term: per d2, two-stage max of v over the FINE grid, around v's own coarse argmax; then max over d2
        TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
        if N_e==0
            for d2_c=1:N_dsemiz
                d12c_gridvals=d12_gridvals(:,:,d2_c);
                if isscalar(n_a)
                    TemptationMatrix_d2=CreateReturnFnMatrix_Disc(TemptationFn, special_n_d, n_a, n_shocks, d12c_gridvals, a_grid, joint_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
                    [~,maxindexT]=max(TemptationMatrix_d2,[],2);
                    midpointT=max(min(maxindexT,n_a-1),2);
                    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1(TemptationFn, special_n_d, n_shocks, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, joint_gridvals_J(:,:,jj), TemptationFnParamsVec,2);
                else % GI2A: fine a1prime window jointly with the full a2prime grid (as in the GP SemiExo GI2A solver raws)
                    TemptationMatrix_d2=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_shocks, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, joint_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0);
                    [~,maxindexT]=max(TemptationMatrix_d2,[],2); % dim 2 is a1prime, so this is directly the a1prime component of the argmax (per d1,a2prime)
                    midpointT=max(min(maxindexT,n_a1-1),2);
                    a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A(TemptationFn, special_n_d, n_shocks, d12c_gridvals, a1prime_grid(a1primeindexesT), a2_grid, a1_grid, a2_grid, joint_gridvals_J(:,:,jj), TemptationFnParamsVec,2,0);
                end
                MostTempting_ford2(:,:,d2_c)=shiftdim(max(TemptationMatrix_Tii,[],1),1); % fine (d1,aprime) max for this d2
            end
            MostTempting=max(MostTempting_ford2,[],3); % [N_a, N_shocks]
        else
            for d2_c=1:N_dsemiz
                d12c_gridvals=d12_gridvals(:,:,d2_c);
                if isscalar(n_a)
                    TemptationMatrix_d2=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, n_shocks, vfoptions.n_e, d12c_gridvals, a_grid, joint_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
                    [~,maxindexT]=max(TemptationMatrix_d2,[],2);
                    midpointT=max(min(maxindexT,n_a-1),2);
                    aprimeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC1_e(TemptationFn, special_n_d, n_shocks, vfoptions.n_e, d12c_gridvals, aprime_grid(aprimeindexesT), a_grid, joint_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,2);
                else % GI2A: fine a1prime window jointly with the full a2prime grid (as in the GP SemiExo GI2A solver raws)
                    TemptationMatrix_d2=CreateReturnFnMatrix_Disc_DC2A_e(TemptationFn, special_n_d, n_shocks, vfoptions.n_e, d12c_gridvals, a1_grid, a2_grid, a1_grid, a2_grid, joint_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,1,0);
                    [~,maxindexT]=max(TemptationMatrix_d2,[],2); % dim 2 is a1prime, so this is directly the a1prime component of the argmax (per d1,a2prime)
                    midpointT=max(min(maxindexT,n_a1-1),2);
                    a1primeindexesT=(midpointT+(midpointT-1)*n2short)+(-n2short-1:1:1+n2short);
                    TemptationMatrix_Tii=CreateReturnFnMatrix_Disc_DC2A_e(TemptationFn, special_n_d, n_shocks, vfoptions.n_e, d12c_gridvals, a1prime_grid(a1primeindexesT), a2_grid, a1_grid, a2_grid, joint_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,2,0);
                end
                MostTempting_ford2(:,:,:,d2_c)=reshape(max(TemptationMatrix_Tii,[],1),[N_a,N_shocks,N_e]); % fine (d1,aprime) max for this d2
            end
            MostTempting=max(MostTempting_ford2,[],4); % [N_a, N_shocks, N_e]
        end

        if jj==N_j
            if N_e==0
                V(:,:,jj)=F_jj+TofPolicy_jj-MostTempting;
            else
                V(:,:,:,jj)=F_jj+TofPolicy_jj-MostTempting;
            end
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

            % Integrate next-period V over e' (if present)
            if N_e==0
                V_next=V(:,:,jj+1);
            else
                V_next=V(:,:,:,jj+1);
                V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
                V_next(isnan(V_next))=0; % 0*(-Inf)=NaN when pi_e puts zero weight on an infeasible e'
                V_next=reshape(V_next, [N_a, N_shocks]);
            end

            % Integrate over z' and per d_semiz over semiz' (same as core SemiExo)
            if N_z==0
                V_next_r=V_next;
                EV_after_z=V_next_r;
                EVnext_byd2=zeros(N_a, N_semiz, N_dsemiz, 'gpuArray');
                for d2_c=1:N_dsemiz
                    pi_d2c=pi_semiz_J(:,:,d2_c,jj)'; % transpose: pi_semiz_J is [N_semiz_from, N_semiz_to]; we want [N_semiz_to, N_semiz_from]
                    EVd2c=sum(EV_after_z .* shiftdim(pi_d2c, -1), 2);
                    EVd2c(isnan(EVd2c))=0;
                    EVnext_byd2(:,:,d2_c)=reshape(EVd2c, [N_a, N_semiz]);
                end
            else
                V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
                EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3);
                EV_after_z(isnan(EV_after_z))=0;
                EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]);
                EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
                for d2_c=1:N_dsemiz
                    pi_d2c=pi_semiz_J(:,:,d2_c,jj)'; % transpose: pi_semiz_J is [N_semiz_from, N_semiz_to]; we want [N_semiz_to, N_semiz_from]
                    pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]); % [1, N_semiz_to, 1, N_semiz_from]
                    EVd2c=sum(EV_after_z .* pi_reshape, 2);
                    EVd2c(isnan(EVd2c))=0;
                    EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
                end
            end

            % Per-state INTERPOLATED lookup on aprime
            if N_e==0
                aprime_lo_jj=aprime_lower_idx(:,:,jj);
                aprime_up_jj=aprime_upper_idx(:,:,jj);
                w_jj=a1_weight(:,:,jj);
                d2_jj=d_semiz_idx(:,:,jj);
                if N_z==0
                    aprime_lo_r=reshape(aprime_lo_jj, [N_a, N_semiz]);
                    aprime_up_r=reshape(aprime_up_jj, [N_a, N_semiz]);
                    w_r=reshape(w_jj, [N_a, N_semiz]);
                    d2_r=reshape(d2_jj, [N_a, N_semiz]);
                    base_off=N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_r(:)-1);
                    lo_idx=aprime_lo_r(:)+base_off;
                    up_idx=aprime_up_r(:)+base_off;
                    EVnext_atpolicy=reshape((1-w_r(:)).*EVnext_byd2(lo_idx)+w_r(:).*EVnext_byd2(up_idx), [N_a, N_semiz]);
                    EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % interpolation weights are probabilities: 0*(-Inf) gives NaN, replace with zeros
                    V(:,:,jj)=F_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
                else
                    aprime_lo_r=reshape(aprime_lo_jj, [N_a, N_semiz, N_z]);
                    aprime_up_r=reshape(aprime_up_jj, [N_a, N_semiz, N_z]);
                    w_r=reshape(w_jj, [N_a, N_semiz, N_z]);
                    d2_r=reshape(d2_jj, [N_a, N_semiz, N_z]);
                    base_off=N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r(:)-1);
                    lo_idx=aprime_lo_r(:)+base_off;
                    up_idx=aprime_up_r(:)+base_off;
                    EVnext_atpolicy=reshape((1-w_r(:)).*EVnext_byd2(lo_idx)+w_r(:).*EVnext_byd2(up_idx), [N_a, N_semiz, N_z]);
                    EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % interpolation weights are probabilities: 0*(-Inf) gives NaN, replace with zeros
                    V(:,:,jj)=F_jj+TofPolicy_jj-MostTempting+beta*reshape(EVnext_atpolicy, [N_a, N_shocks]);
                end
            else
                if N_z==0
                    EVnext_atpolicy=zeros(N_a, N_semiz, N_e, 'gpuArray');
                    for e_c=1:N_e
                        aprime_lo_e=reshape(aprime_lower_idx(:,:,e_c,jj), [N_a, N_semiz]);
                        aprime_up_e=reshape(aprime_upper_idx(:,:,e_c,jj), [N_a, N_semiz]);
                        w_e=reshape(a1_weight(:,:,e_c,jj), [N_a, N_semiz]);
                        d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz]);
                        base_off=N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1);
                        lo_idx=aprime_lo_e(:)+base_off;
                        up_idx=aprime_up_e(:)+base_off;
                        EVnext_atpolicy(:,:,e_c)=reshape((1-w_e(:)).*EVnext_byd2(lo_idx)+w_e(:).*EVnext_byd2(up_idx), [N_a, N_semiz]);
                    end
                    EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % interpolation weights are probabilities: 0*(-Inf) gives NaN, replace with zeros
                    V(:,:,:,jj)=F_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
                else
                    EVnext_atpolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
                    for e_c=1:N_e
                        aprime_lo_e=reshape(aprime_lower_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                        aprime_up_e=reshape(aprime_upper_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                        w_e=reshape(a1_weight(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                        d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                        base_off=N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1);
                        lo_idx=aprime_lo_e(:)+base_off;
                        up_idx=aprime_up_e(:)+base_off;
                        EVnext_atpolicy(:,:,:,e_c)=reshape((1-w_e(:)).*EVnext_byd2(lo_idx)+w_e(:).*EVnext_byd2(up_idx), [N_a, N_semiz, N_z]);
                    end
                    EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % interpolation weights are probabilities: 0*(-Inf) gives NaN, replace with zeros
                    V(:,:,:,jj)=F_jj+TofPolicy_jj-MostTempting+beta*reshape(EVnext_atpolicy, [N_a, N_shocks, N_e]);
                end
            end
        end
    end

else % no grid interpolation layer

    %% PolicyValues for ReturnFn/TemptationFn evaluation
    if N_e==0
        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_semiz and vfoptions.n_e
        % PolicyValues shape: [l_d+l_aprime, N_a, N_shocks, N_j]
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_shocks, l_d+l_aprime, N_j]
    else
        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_semiz and vfoptions.n_e
        % PolicyValues shape: [l_d+l_aprime, N_a, N_shocks*N_e, N_j]
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_shocks*N_e, l_d+l_aprime, N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
    end
    l_daprime=size(PolicyValues,1);

    %% Extract per-state indices: aprime_idx, d_semiz_idx
    % Reshape Policy to Kron form
    if N_e==0
        Policy_k=reshape(Policy,[l_d+l_aprime, N_a, N_shocks, N_j]);
    else
        Policy_k=reshape(Policy,[l_d+l_aprime, N_a, N_shocks, N_e, N_j]);
    end

    % d_semiz components are positions (l_d-l_dsemiz+1) through l_d
    if N_e==0
        d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
    else
        d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    end
    cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
    for ii=1:l_dsemiz
        comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1); % drop leading singleton
        d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
    end

    % aprime components are positions (l_d+1) through (l_d+l_aprime)
    if N_e==0
        aprime_idx=ones(N_a,N_shocks,N_j,'gpuArray');
    else
        aprime_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    end
    cumprods_a=[1, cumprod(n_a(1:end-1))];
    for ii=1:l_aprime
        comp=shiftdim(Policy_k(l_d+ii, :, :, :, :),1);
        aprime_idx=aprime_idx+cumprods_a(ii)*(comp-1);
    end

    %% Backward iteration
    if N_e==0
        V=zeros(N_a, N_shocks, N_j, 'gpuArray');
    else
        V=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
    end

    [~, SZ_grid_noz]=ndgrid(1:N_a, 1:N_semiz); % for N_z==0 lookup
    if N_z>0
        [~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z); % for N_z>0 lookup
    end

    for reverse_j=0:N_j-1
        jj=N_j-reverse_j;

        % Evaluate ReturnFn and TemptationFn at policy
        FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
        TemptationFnParamsCell=CreateCellFromParams(Parameters,TemptationFnParamNames,jj);
        if N_e==0
            F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
            TofPolicy_jj=EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
            % shape: [N_a, N_shocks]
        else
            F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
            TofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(TemptationFn, TemptationFnParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
        end

        % Most-tempting term: per d2, max of v over the full (d1,aprime) choice set; then max over d2
        TemptationFnParamsVec=CreateVectorFromParams(Parameters,TemptationFnParamNames,jj);
        if N_e==0
            for d2_c=1:N_dsemiz
                d12c_gridvals=d12_gridvals(:,:,d2_c);
                TemptationMatrix_d2=CreateReturnFnMatrix_Disc(TemptationFn, special_n_d, n_a, n_shocks, d12c_gridvals, a_grid, joint_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
                MostTempting_ford2(:,:,d2_c)=shiftdim(max(TemptationMatrix_d2,[],1),1); % full (d1,aprime) for this d2
            end
            MostTempting=max(MostTempting_ford2,[],3); % [N_a, N_shocks]
        else
            for d2_c=1:N_dsemiz
                d12c_gridvals=d12_gridvals(:,:,d2_c);
                TemptationMatrix_d2=CreateReturnFnMatrix_Disc_e(TemptationFn, special_n_d, n_a, n_shocks, vfoptions.n_e, d12c_gridvals, a_grid, joint_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
                MostTempting_ford2(:,:,:,d2_c)=reshape(max(TemptationMatrix_d2,[],1),[N_a,N_shocks,N_e]); % full (d1,aprime) for this d2
            end
            MostTempting=max(MostTempting_ford2,[],4); % [N_a, N_shocks, N_e]
        end

        if jj==N_j
            if N_e==0
                V(:,:,jj)=F_jj+TofPolicy_jj-MostTempting;
            else
                V(:,:,:,jj)=F_jj+TofPolicy_jj-MostTempting;
            end
        else
            beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

            % Integrate next-period V over e' (if present), then over z' (if present), then over semiz' (per d_semiz)
            if N_e==0
                V_next=V(:,:,jj+1); % [N_a, N_shocks]
            else
                V_next=V(:,:,:,jj+1); % [N_a, N_shocks, N_e]
                % Integrate over e' using iid pi_e_J(:,jj+1) (the distribution of the e realized in period jj+1)
                V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_shocks, 1]
                V_next(isnan(V_next))=0; % 0*(-Inf)=NaN when pi_e puts zero weight on an infeasible e'
                V_next=reshape(V_next, [N_a, N_shocks]);
            end

            % Reshape V_next as [N_a, N_semiz, N_z (or 1)]
            if N_z==0
                V_next_r=V_next; % [N_a, N_semiz]
                % Step 1: integrate over z' is trivial (no z)
                EV_after_z=V_next_r; % [N_a, N_semiz_to]
                % Step 2: for each d_semiz, integrate over semiz'
                EVnext_byd2=zeros(N_a, N_semiz, N_dsemiz, 'gpuArray');
                for d2_c=1:N_dsemiz
                    pi_d2c=pi_semiz_J(:,:,d2_c,jj)'; % transpose: pi_semiz_J is [N_semiz_from, N_semiz_to]; we want [N_semiz_to, N_semiz_from] so the broadcast contracts semiz_to with V's semiz_to
                    EVd2c=sum(EV_after_z .* shiftdim(pi_d2c, -1), 2); % [N_a, 1, N_semiz_from]
                    EVd2c(isnan(EVd2c))=0;
                    EVnext_byd2(:,:,d2_c)=reshape(EVd2c, [N_a, N_semiz]);
                end
            else
                V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
                % Step 1: integrate over z' (does not depend on d_semiz)
                % EV_after_z[anext, semiz_to, z_from] = sum_{z_to} pi_z_J(z_from, z_to, jj) * V_next[anext, semiz_to, z_to]
                EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3); % [N_a, N_semiz_to, 1, N_z_from]
                EV_after_z(isnan(EV_after_z))=0;
                EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]); % [anext, semiz_to, z_from]
                % Step 2: for each d_semiz, integrate over semiz'
                EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
                for d2_c=1:N_dsemiz
                    pi_d2c=pi_semiz_J(:,:,d2_c,jj)'; % transpose: pi_semiz_J is [N_semiz_from, N_semiz_to]; we want [N_semiz_to, N_semiz_from]
                    pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]); % [1, N_semiz_to, 1, N_semiz_from]
                    EVd2c=sum(EV_after_z .* pi_reshape, 2); % [N_a, 1, N_z, N_semiz_from]
                    EVd2c(isnan(EVd2c))=0;
                    EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
                end
            end

            % Step 3: per-state lookup using aprime_idx and d_semiz_idx
            if N_e==0
                % aprime_idx, d_semiz_idx shape at jj: [N_a, N_shocks]
                aprime_jj=aprime_idx(:,:,jj);
                d2_jj=d_semiz_idx(:,:,jj);
                if N_z==0
                    % EVnext_byd2: [N_a, N_semiz, N_dsemiz]; index = aprime + N_a*(sz-1) + N_a*N_semiz*(d2-1)
                    aprime_jj_r=reshape(aprime_jj, [N_a, N_semiz]);
                    d2_jj_r=reshape(d2_jj, [N_a, N_semiz]);
                    linear_idx=aprime_jj_r(:)+N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_jj_r(:)-1);
                    EVnext_atpolicy=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz]);
                    V(:,:,jj)=F_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
                else
                    % EVnext_byd2: [N_a, N_semiz, N_z, N_dsemiz]; index = aprime + N_a*(sz-1) + N_a*N_semiz*(z-1) + N_a*N_semiz*N_z*(d2-1)
                    aprime_jj_r=reshape(aprime_jj, [N_a, N_semiz, N_z]);
                    d2_jj_r=reshape(d2_jj, [N_a, N_semiz, N_z]);
                    linear_idx=aprime_jj_r(:)+N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_jj_r(:)-1);
                    EVnext_atpolicy=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz, N_z]);
                    V(:,:,jj)=F_jj+TofPolicy_jj-MostTempting+beta*reshape(EVnext_atpolicy, [N_a, N_shocks]);
                end
            else
                % With e, V update is per-e
                if N_z==0
                    EVnext_atpolicy=zeros(N_a, N_semiz, N_e, 'gpuArray');
                    for e_c=1:N_e
                        aprime_e=reshape(aprime_idx(:,:,e_c,jj), [N_a, N_semiz]);
                        d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz]);
                        linear_idx=aprime_e(:)+N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1);
                        EVnext_atpolicy(:,:,e_c)=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz]);
                    end
                    V(:,:,:,jj)=F_jj+TofPolicy_jj-MostTempting+beta*EVnext_atpolicy;
                else
                    EVnext_atpolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
                    for e_c=1:N_e
                        aprime_e=reshape(aprime_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                        d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                        linear_idx=aprime_e(:)+N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1);
                        EVnext_atpolicy(:,:,:,e_c)=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz, N_z]);
                    end
                    V(:,:,:,jj)=F_jj+TofPolicy_jj-MostTempting+beta*reshape(EVnext_atpolicy, [N_a, N_shocks, N_e]);
                end
            end
        end
    end

end

%% Reshape V out of Kron form
if N_z==0 && N_e==0
    V=reshape(V, [n_a, n_semiz, N_j]);
elseif N_z==0 && N_e>0
    V=reshape(V, [n_a, n_semiz, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V=reshape(V, [n_a, n_semiz, n_z, N_j]);
else
    V=reshape(V, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
end



varargout={V};

end
