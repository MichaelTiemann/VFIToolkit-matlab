function [V,Policy]=ValueFnIter_FHorz_ExpAsset_raw(n_d1,n_d2,n_a1,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); % indexes the optimal choice for d and a1prime rest of dimensions a,z

use_vec_broadcast = isfield(vfoptions,'vectorized') && (vfoptions.vectorized==1);
if use_vec_broadcast
    ReturnFn_Direct = unwrap_passthrough_fn_local(ReturnFn);
end

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==1
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if use_vec_broadcast
        [F_ref, d1_idx] = eval_expasset_return_refine1_broadcast(ReturnFn_Direct, n_d1, n_d2, n_a1, n_a2, n_z, d_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp, d2a1p_idx] = max(F_ref, [], 1);
        d2a1p_mat = reshape(d2a1p_idx, [N_a, N_z]);
        lin_idx = d2a1p_mat + (N_d2*N_a1) * ((0:N_a-1)' + N_a * (0:N_z-1));
        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d1_idx(lin_idx) + N_d1 * (d2a1p_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,N_j) = v_j;
        Policy(:,:,N_j) = pol_j;
    elseif vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,1); % Level=0, Refine=1
        [F_ref, d1_idx]=max(ReturnMatrix,[],1);
        F_ref=reshape(F_ref,[N_d2*N_a1,N_a,N_z]);
        d1_idx=reshape(d1_idx,[N_d2*N_a1,N_a,N_z]);
        %Calc the max and its index
        [Vtemp, d2a1p_idx]=max(F_ref,[],1);
        d2a1p_mat = reshape(d2a1p_idx, [N_a, N_z]);
        lin_idx = d2a1p_mat + (N_d2*N_a1) * ((0:N_a-1)' + N_a * (0:N_z-1));
        v_j = reshape(Vtemp,[N_a,N_z]);
        pol_j = d1_idx(lin_idx) + N_d1*(d2a1p_mat-1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,N_j)=v_j;
        Policy(:,:,N_j)=pol_j;
    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1, n_d2, n_a1, n_a1,n_a2, special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            %Calc the max and its index
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_z]); % First, switch V_Jplus1 into Kron form

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2]
        aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z); % [N_d2*N_a1,N_a2,N_z]

        Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u,zprime)
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2], per-dim factored (lower-grid
        % index and prob of lower, one row per a2 dim). Fold to the four corners keeping the
        % a1prime offset, then nested 2-corner interp with skipinterp at each level and
        % per-contribution NaN cleanup for 0*(-Inf).
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[N_d2,N_a2]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:),[N_d2,N_a2]),N_a1,1,N_z);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:),[N_d2,N_a2]),N_a1,1,N_z);
        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2);
        aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1);
        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1,N_a2,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1,N_a2,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1,N_a2,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1,N_a2,N_z]);
        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    % Already applied the probabilities from interpolating onto grid

    EVinf=(EV==-Inf);
    EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
    EV=reshape(EV,[N_d2*N_a1*N_a2,N_z])*pi_z_J(:,:,N_j)';
    EV(reshape(EVinf,[N_d2*N_a1*N_a2,N_z])*(pi_z_J(:,:,N_j)'>0)>0)=-Inf; % exact -Inf restoration
    EV=squeeze(reshape(EV,[N_d2*N_a1,N_a2,N_z]));
    % EV is over (d2,a1prime,a2,z)

    if use_vec_broadcast
        DiscountedEV_ref = DiscountFactorParamsVec * repelem(EV, 1, N_a1); % [N_d2*N_a1, N_a1*N_a2, N_z]
        [F_ref, d1_idx] = eval_expasset_return_refine1_broadcast(ReturnFn_Direct, n_d1, n_d2, n_a1, n_a2, n_z, d_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        entireRHS = F_ref + DiscountedEV_ref;
        [Vtemp, d2a1p_idx] = max(entireRHS, [], 1);
        d2a1p_mat = reshape(d2a1p_idx, [N_a, N_z]);
        lin_idx = d2a1p_mat + (N_d2*N_a1) * ((0:N_a-1)' + N_a * (0:N_z-1));
        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d1_idx(lin_idx) + N_d1 * (d2a1p_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,N_j) = v_j;
        Policy(:,:,N_j) = pol_j;
    elseif vfoptions.lowmemory==0
        DiscountedEV_ref = DiscountFactorParamsVec * repelem(EV, 1, N_a1); % Refine=1: 1/N_d1 size!
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2,n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,1); % Level=0, Refine=1
        [F_ref, d1_idx]=max(ReturnMatrix,[],1);
        F_ref=reshape(F_ref,[N_d2*N_a1,N_a,N_z]);
        d1_idx=reshape(d1_idx,[N_d2*N_a1,N_a,N_z]);

        entireRHS=F_ref+DiscountedEV_ref;

        %Calc the max and its index
        [Vtemp,d2a1p_idx]=max(entireRHS,[],1);
        d2a1p_mat = reshape(d2a1p_idx, [N_a, N_z]);
        lin_idx = d2a1p_mat + (N_d2*N_a1) * ((0:N_a-1)' + N_a * (0:N_z-1));

        v_j = reshape(Vtemp,[N_a,N_z]);
        pol_j = d1_idx(lin_idx) + N_d1*(d2a1p_mat-1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,N_j)=v_j;
        Policy(:,:,N_j)=pol_j;

    elseif vfoptions.lowmemory==1
        DiscountedEV=DiscountFactorParamsVec*repelem(EV,N_d1,N_a1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

            entireRHS_z=ReturnMatrix_z+DiscountedEV_z;

            %Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            Policy(:,z_c,N_j)=maxindex;
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

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1,1); % [N_d2*N_a1,N_a2]
        aprimeProbs=repmat(a2primeProbs,N_a1,1,N_z); % [N_d2*N_a1,N_a2,N_z]

        Vlower=reshape(V(aprimeIndex(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
        Vupper=reshape(V(aprimeplus1Index(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,zprime)
        EV(aprimeProbs==0)=Vupper(aprimeProbs==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
        EV(aprimeProbs==1)=Vlower(aprimeProbs==1);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2], per-dim factored (lower-grid
        % index and prob of lower, one row per a2 dim). Fold to the four corners keeping the
        % a1prime offset, then nested 2-corner interp with skipinterp at each level and
        % per-contribution NaN cleanup for 0*(-Inf).
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[N_d2,N_a2]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:),[N_d2,N_a2]),N_a1,1,N_z);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:),[N_d2,N_a2]),N_a1,1,N_z);
        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2);
        aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1);
        V_ll=reshape(V(aprime_ll(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
        V_hl=reshape(V(aprime_hl(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
        V_lh=reshape(V(aprime_lh(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
        V_hh=reshape(V(aprime_hh(:),:,jj+1),[N_d2*N_a1,N_a2,N_z]);
        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end
    % Already applied the probabilities from interpolating onto grid

    % EV is over (d2,a1prime,a2,zprime)
    EVinf=(EV==-Inf);
    EV(EVinf)=-1e250; % stop -Inf*0 -> NaN inside the product
    EV=reshape(EV,[N_d2*N_a1*N_a2,N_z])*pi_z_J(:,:,jj)';
    EV(reshape(EVinf,[N_d2*N_a1*N_a2,N_z])*(pi_z_J(:,:,jj)'>0)>0)=-Inf; % exact -Inf restoration
    EV=squeeze(reshape(EV,[N_d2*N_a1,N_a2,N_z]));
    % EV is over (d2,a1prime,a2,z)

    if use_vec_broadcast
        DiscountedEV_ref = DiscountFactorParamsVec * repelem(EV, 1, N_a1); % [N_d2*N_a1, N_a1*N_a2, N_z]
        [F_ref, d1_idx] = eval_expasset_return_refine1_broadcast(ReturnFn_Direct, n_d1, n_d2, n_a1, n_a2, n_z, d_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), ReturnFnParamsVec);
        entireRHS = F_ref + DiscountedEV_ref;
        [Vtemp, d2a1p_idx] = max(entireRHS, [], 1);
        d2a1p_mat = reshape(d2a1p_idx, [N_a, N_z]);
        lin_idx = d2a1p_mat + (N_d2*N_a1) * ((0:N_a-1)' + N_a * (0:N_z-1));
        v_j = reshape(Vtemp, [N_a, N_z]);
        pol_j = d1_idx(lin_idx) + N_d1 * (d2a1p_mat - 1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,jj) = v_j;
        Policy(:,:,jj) = pol_j;

    elseif vfoptions.lowmemory==0
        DiscountedEV_ref = DiscountFactorParamsVec * repelem(EV, 1, N_a1); % Refine=1: 1/N_d1 size!
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2,n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals,z_gridvals_J(:,:,jj), ReturnFnParamsVec,0,1); % Level=0, Refine=1
        [F_ref, d1_idx]=max(ReturnMatrix,[],1);
        F_ref=reshape(F_ref,[N_d2*N_a1,N_a,N_z]);
        d1_idx=reshape(d1_idx,[N_d2*N_a1,N_a,N_z]);

        entireRHS=F_ref+DiscountedEV_ref;

        %Calc the max and its index
        [Vtemp,d2a1p_idx]=max(entireRHS,[],1);
        d2a1p_mat = reshape(d2a1p_idx, [N_a, N_z]);
        lin_idx = d2a1p_mat + (N_d2*N_a1) * ((0:N_a-1)' + N_a * (0:N_z-1));

        v_j = reshape(Vtemp,[N_a,N_z]);
        pol_j = d1_idx(lin_idx) + N_d1*(d2a1p_mat-1);
        pol_j(~isfinite(v_j)) = 1;
        V(:,:,jj)=v_j;
        Policy(:,:,jj)=pol_j;

    elseif vfoptions.lowmemory==1
        DiscountedEV=DiscountFactorParamsVec*repelem(EV,N_d1,N_a1);
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,z_c);

            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, special_n_z, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

            entireRHS_z=ReturnMatrix_z+DiscountedEV_z;

            %Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
    end
end

%%
Policy=shiftdim(Policy,-1);

end

%% ---- LOCAL HELPERS FOR TILED GPU BROADCAST + FUSED REFINE=1 ----
function [F_max, F_idx] = eval_expasset_return_refine1_broadcast(ReturnFn, n_d1, n_d2, n_a1, n_a2, n_z, d_gridvals, a1_gridvals, a2_gridvals, z_gridvals, ReturnFnParamsVec)
% Evaluates dual-compatible ReturnFn via gpuArray implicit expansion in ~2 GB tiles
% along a1, reducing max(..., [], 1) over N_d1 inside each tile to minimize VRAM traffic.
N_d1 = prod(n_d1);
N_d2 = prod(n_d2);
N_d  = N_d1 * N_d2;
N_a1 = prod(n_a1);
N_a2 = prod(n_a2);
N_z  = prod(n_z);

l_d  = length([n_d1, n_d2]);
l_a1 = length(n_a1);
l_a2 = length(n_a2);
l_z  = length(n_z);

pcell = num2cell(ReturnFnParamsVec)';

d_args = cell(1, l_d);
for k = 1:l_d, d_args{k} = d_gridvals(:, k); end

ap_args = cell(1, l_a1);
for k = 1:l_a1, ap_args{k} = shiftdim(a1_gridvals(:, k), -1); end

a2_args = cell(1, l_a2);
for k = 1:l_a2, a2_args{k} = shiftdim(a2_gridvals(:, k), -3); end

z_args = cell(1, l_z);
for k = 1:l_z, z_args{k} = shiftdim(z_gridvals(:, k), -4); end

F_max = zeros(N_d2 * N_a1, N_a1 * N_a2, N_z, 'gpuArray');
F_idx = zeros(N_d2 * N_a1, N_a1 * N_a2, N_z, 'gpuArray');

% Target ~268M doubles (~2.1 GB) per tile so peak temporary VRAM stays comfortably < 8 GB
pts_per_a1 = max(1, N_d * N_a1 * N_a2 * N_z);
blk_sz     = max(1, min(N_a1, floor(268435456 / pts_per_a1)));
a2_offsets = N_a1 * (0:N_a2-1);

a1_args = cell(1, l_a1);
for s = 1:blk_sz:N_a1
    idx = s:min(N_a1, s + blk_sz - 1);
    nb  = numel(idx);
    for k = 1:l_a1
        a1_args{k} = shiftdim(a1_gridvals(idx, k), -2);
    end
    Fb = ReturnFn(d_args{:}, ap_args{:}, a1_args{:}, a2_args{:}, z_args{:}, pcell{:});
    target_sz = [N_d, N_a1, nb, N_a2, N_z];
    cur_sz    = [size(Fb,1), size(Fb,2), size(Fb,3), size(Fb,4), size(Fb,5)];
    if ~isequal(cur_sz, target_sz)
        rep_fac = ones(1, 5);
        for dim = 1:5
            if cur_sz(dim) == 1 && target_sz(dim) > 1
                rep_fac(dim) = target_sz(dim);
            end
        end
        Fb = reshape(Fb, [N_d1, N_d2 * cur_sz(2), cur_sz(3), cur_sz(4), cur_sz(5)]);
        [m_b, i_b] = max(Fb, [], 1);
        if any(rep_fac(2:5) > 1)
            m_b = repmat(m_b, [1, rep_fac(2), rep_fac(3), rep_fac(4), rep_fac(5)]);
            i_b = repmat(i_b, [1, rep_fac(2), rep_fac(3), rep_fac(4), rep_fac(5)]);
        end
    else
        Fb = reshape(Fb, [N_d1, N_d2 * N_a1, nb, N_a2, N_z]);
        [m_b, i_b] = max(Fb, [], 1);
    end
    lin_cols = idx(:) + a2_offsets;
    F_max(:, lin_cols(:), :) = reshape(m_b, [N_d2 * N_a1, nb * N_a2, N_z]);
    F_idx(:, lin_cols(:), :) = reshape(i_b, [N_d2 * N_a1, nb * N_a2, N_z]);
end
end

function fn_out = unwrap_passthrough_fn_local(fn_in)
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
