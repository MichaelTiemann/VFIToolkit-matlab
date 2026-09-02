function [Vtilde,Policy3,Valt,Policy3alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetsemizN_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% experienceassetsemiz, no d1, no ordinary z: d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% semiz is semi-exog state (drives the asset); no ordinary z, so bothz = semiz
% aprimeFn = aprimeFn(d2, a2, semiz, ...)
% Naive quasi-hyperbolic dual.  Two independent maximisations over the same argmax axis:
%   Valt/Policy3alt maximise  F + beta*EV        (the exponential value; drives the recursion)
%   Vtilde/Policy3  maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0 is received as a trailing input.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_bothz=N_semiz; % no ordinary z

Valt=zeros(N_a,N_bothz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_j,'gpuArray');
% Policy storage with separate entries for d2, d3, a1prime (no d1)
Policy3alt=zeros(3,N_a,N_bothz,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_bothz,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=semiz_gridvals_J; % bothz = semiz

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz));
end

% Preallocate
V_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_semiz, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy3alt(1,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy3alt(2,:,:,N_j)=ceil(d_ind/N_d2); % d3
        Policy3alt(3,:,:,N_j)=ceil(maxindex/N_d23); % a1prime
    elseif vfoptions.lowmemory==1
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy3alt(1,:,z_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy3alt(2,:,z_c,N_j)=ceil(d_ind/N_d2);
            Policy3alt(3,:,z_c,N_j)=ceil(maxindex/N_d23);
        end
    end
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy3(:,:,:,N_j)=Policy3alt(:,:,:,N_j);
else
    % aprime depends on (d2, a1, a2, current_semiz); independent of d3 -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_semiz]  (N_semiz==N_bothz here)

    aprimeIndex_full=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1, N_a2, N_bothz]
    aprimeplus1Index_full=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_full=repmat(a2primeProbs,N_a1,1,1);

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            entireRHS_alt=ReturnMatrix_d3+beta*repelem(entireEV,1,N_a1,1);
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*repelem(entireEV,1,N_a1,1);
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

                EV_z=V_Jplus1.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                aprime_slice=aprimeIndex_full(:,:,z_c); % bothz==semiz, so z_c indexes semiz directly
                aprimeplus1_slice=aprimeplus1Index_full(:,:,z_c);
                aprimeProbs_slice=aprimeProbs_full(:,:,z_c);

                EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                entireRHS_alt=ReturnMatrix_d3z+beta*repelem(entireEV_z,1,N_a1);
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=Vtemp_alt;
                Policy_ford3_alt(:,z_c,d3_c)=maxindex_alt;
                entireRHS_tilde=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,1,N_a1);
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=Vtemp_tilde;
                Policy_ford3_tilde(:,z_c,d3_c)=maxindex_tilde;
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,N_j)=V_jj;
    Policy3alt(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy3alt(1,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3alt(3,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy3(1,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime

end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex_full=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index_full=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_full=repmat(a2primeProbs,N_a1,1,1);

    EVpre=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_semiz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            entireRHS_alt=ReturnMatrix_d3+beta*repelem(entireEV,1,N_a1,1);
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*repelem(entireEV,1,N_a1,1);
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                aprime_slice=aprimeIndex_full(:,:,z_c);
                aprimeplus1_slice=aprimeplus1Index_full(:,:,z_c);
                aprimeProbs_slice=aprimeProbs_full(:,:,z_c);

                EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                entireRHS_alt=ReturnMatrix_d3z+beta*repelem(entireEV_z,1,N_a1);
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=Vtemp_alt;
                Policy_ford3_alt(:,z_c,d3_c)=maxindex_alt;
                entireRHS_tilde=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,1,N_a1);
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=Vtemp_tilde;
                Policy_ford3_tilde(:,z_c,d3_c)=maxindex_tilde;
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,jj)=V_jj;
    Policy3alt(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy3alt(1,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1;
    Policy3alt(3,:,:,jj)=ceil(d2a1prime_ind/N_d2);

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy3(1,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1;
    Policy3(3,:,:,jj)=ceil(d2a1prime_ind/N_d2);

end


end
