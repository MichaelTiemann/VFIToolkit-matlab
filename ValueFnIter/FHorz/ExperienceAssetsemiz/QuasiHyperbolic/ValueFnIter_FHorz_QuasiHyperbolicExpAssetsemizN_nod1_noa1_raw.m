function [Vtilde,Policy2,Valt,Policy2alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetsemizN_nod1_noa1_raw(n_d2,n_d3,n_a2,n_z,n_semiz,N_j, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% noa1 version of ValueFnIter_FHorz_ExpAssetsemiz_nod1_raw (nod1, z, noe).
% a = a2 (the experience asset is the only endogenous state)
% semiz is semi-exog state (drives the asset), z is an ordinary Markov; bothz=[semiz,z] with semiz the fast index
% aprimeFn = aprimeFn(d2, a2, semiz, ...)
% Policy2 stores (d2, d3) -- no a1prime channel since noa1.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);
N_semiz=prod(n_semiz);
N_bothz=prod(n_bothz);

Valt=zeros(N_a,N_bothz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_j,'gpuArray');
Policy2alt=zeros(2,N_a,N_bothz,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_bothz,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_semizblock=[n_semiz,ones(1,length(n_z))]; % semiz vectorised, z scalar (lowmemory=1 split over z)
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

V_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');

% Offsets for linear indexing
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);
semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, n_bothz, d23_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy2alt(1,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy2alt(2,:,:,N_j)=ceil(d_ind/N_d2); % d3
    elseif vfoptions.lowmemory==1 % loop over z (markov), vectorize over semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, special_n_semizblock, d23_gridvals, a2_grid, z_valblock, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,semizblock,N_j)=shiftdim(Vtemp,1);
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2alt(1,:,semizblock,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2alt(2,:,semizblock,N_j)=ceil(d_ind/N_d2);
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, special_n_bothz, d23_gridvals, a2_grid, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2alt(1,:,z_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2alt(2,:,z_c,N_j)=ceil(d_ind/N_d2);
        end
    end
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy2(:,:,:,N_j)=Policy2alt(:,:,:,N_j);
else
    % aprime depends on (d2, a2, current_semiz); independent of d3 and of z -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_semiz]

    % noa1: aprimeIndex/Plus1Index reduce to a2primeIndex(+1) directly (no N_a1*... combination)
    aprimeIndex=a2primeIndex;        % [N_d2, N_a2, N_semiz]
    aprimeplus1Index=a2primeIndex+1;
    aprimeProbs_d2a2semiz=a2primeProbs;
    aprimeIndex_full=repmat(aprimeIndex,1,1,N_z); % [N_d2, N_a2, N_bothz] (semiz fast, so tile over z)
    aprimeplus1Index_full=repmat(aprimeplus1Index,1,1,N_z);
    aprimeProbs_full=repmat(aprimeProbs_d2a2semiz,1,1,N_z);

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, n_bothz, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

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

            entireRHS_alt=ReturnMatrix_d3+beta*entireEV;
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*entireEV;
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end
    elseif vfoptions.lowmemory==1 % loop over z (markov), vectorize over semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_semizblock, d23_gridvals_val, a2_grid, z_valblock, ReturnFnParamsVec);

                EV=V_Jplus1.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                EV1=EV_2D(aprimeIndex+semizblock_offset);       % aprimeIndex is [N_d2,N_a2,N_semiz]
                EV2=EV_2D(aprimeplus1Index+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_d2a2semiz;
                aprimeProbs_z(skipinterp)=0;
                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                entireRHS_alt=ReturnMatrix_d3z+beta*entireEV_z;
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,semizblock,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3z+beta0beta*entireEV_z;
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,semizblock,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,semizblock,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_bothz, d23_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);

                EV_z=V_Jplus1.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                semiz_part=rem(z_c-1,N_semiz)+1; % aprime depends on semiz (fast index of bothz)
                aprime_slice=aprimeIndex(:,:,semiz_part);
                aprimeplus1_slice=aprimeplus1Index(:,:,semiz_part);
                aprimeProbs_slice=aprimeProbs_d2a2semiz(:,:,semiz_part);

                EV1=reshape(EV_z(aprime_slice),[N_d2,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                entireRHS_alt=ReturnMatrix_d3z+beta*entireEV_z;
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=Vtemp_alt;
                Policy_ford3_alt(:,z_c,d3_c)=maxindex_alt;
                entireRHS_tilde=ReturnMatrix_d3z+beta0beta*entireEV_z;
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=Vtemp_tilde;
                Policy_ford3_tilde(:,z_c,d3_c)=maxindex_tilde;
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,N_j)=V_jj;
    Policy2alt(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy2alt(1,:,:,N_j)=d2_ind; % d2 (no a1prime split)

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy2(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy2(1,:,:,N_j)=d2_ind; % d2 (no a1prime split)

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
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_semiz]

    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;
    aprimeProbs_d2a2semiz=a2primeProbs;
    aprimeIndex_full=repmat(aprimeIndex,1,1,N_z);
    aprimeplus1Index_full=repmat(aprimeplus1Index,1,1,N_z);
    aprimeProbs_full=repmat(aprimeProbs_d2a2semiz,1,1,N_z);

    EVpre=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, n_bothz, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec);

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

            entireRHS_alt=ReturnMatrix_d3+beta*entireEV;
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*entireEV;
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end
    elseif vfoptions.lowmemory==1 % loop over z (markov), vectorize over semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_semizblock, d23_gridvals_val, a2_grid, z_valblock, ReturnFnParamsVec);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

                EV1=EV_2D(aprimeIndex+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_d2a2semiz;
                aprimeProbs_z(skipinterp)=0;
                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                entireRHS_alt=ReturnMatrix_d3z+beta*entireEV_z;
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,semizblock,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3z+beta0beta*entireEV_z;
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,semizblock,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,semizblock,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_bothz, d23_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                semiz_part=rem(z_c-1,N_semiz)+1;
                aprime_slice=aprimeIndex(:,:,semiz_part);
                aprimeplus1_slice=aprimeplus1Index(:,:,semiz_part);
                aprimeProbs_slice=aprimeProbs_d2a2semiz(:,:,semiz_part);

                EV1=reshape(EV_z(aprime_slice),[N_d2,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                entireRHS_alt=ReturnMatrix_d3z+beta*entireEV_z;
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,z_c,d3_c)=Vtemp_alt;
                Policy_ford3_alt(:,z_c,d3_c)=maxindex_alt;
                entireRHS_tilde=ReturnMatrix_d3z+beta0beta*entireEV_z;
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,z_c,d3_c)=Vtemp_tilde;
                Policy_ford3_tilde(:,z_c,d3_c)=maxindex_tilde;
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,jj)=V_jj;
    Policy2alt(2,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy2alt(1,:,:,jj)=d2_ind; % d2

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy2(2,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy2(1,:,:,jj)=d2_ind; % d2


end


end
