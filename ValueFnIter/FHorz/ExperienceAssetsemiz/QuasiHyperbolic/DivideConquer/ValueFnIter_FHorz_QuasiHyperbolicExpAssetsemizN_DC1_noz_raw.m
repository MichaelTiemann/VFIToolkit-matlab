function [Vtilde,Policy3,Valt,Policy3alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetsemizN_DC1_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic discounting + ExperienceAssetsemiz, Divide-and-Conquer (DC1 over a1prime).
% d2 determines the experience asset a2, d3 determines the semi-exogenous state, a1 is the standard endogenous state.
% Naive: two fully independent maximisations,
%   Valt/Policy3alt maximise  F + beta*EV        (the exponential value; drives the backward recursion)
%   Vtilde/Policy3  maximise  F + beta0*beta*EV  (the QH-perceived value that is reported)
% Each maximisation is a full divide-and-conquer pass (its own maxindex1/maxgap/narrow band), and
% each has its own per-d3 store, so the max over d3 is also done twice.
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state
% aprimeFn = aprimeFn(d2, a2, semiz, ...)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest

n_bothz=n_semiz;

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
d2ind=repelem((1:1:N_d2)',N_d1,1); % [N_d12,1]
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_bothz=N_semiz;

Valt=zeros(N_a,N_bothz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_j,'gpuArray');
Policy3alt=zeros(4,N_a,N_bothz,N_j,'gpuArray');
Policy3=zeros(4,N_a,N_bothz,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=semiz_gridvals_J;

n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz));
else
    bothzind=shiftdim((0:1:N_bothz-1),-1); % already includes -1
end

% Preallocate
V_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,n_d3],n_a1,vfoptions.level1n,n_a2,n_bothz, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
        dind=rem(maxindex2-1,N_d)+1;
        d12_ind=rem(dind-1,N_d12)+1;
        Policy3alt(1,curraindex,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy3alt(2,curraindex,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy3alt(3,curraindex,:,N_j)=ceil(dind/N_d12); % d3
        Policy3alt(4,curraindex,:,N_j)=ceil(maxindex2/N_d); % a1prime

        maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,n_d3],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                allind=dind+N_d*a2ind+N_d*N_a2*bothzind;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy3alt(1,curraindex,:,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy3alt(2,curraindex,:,N_j)=ceil(d12_ind/N_d1);
                Policy3alt(3,curraindex,:,N_j)=ceil(dind/N_d12);
                Policy3alt(4,curraindex,:,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
            else
                loweredge=maxindex1(:,1,ii,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,n_d3],1,level1iidiff(ii),n_a2,n_bothz, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                Valt(curraindex,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d)+1);
                a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                allind=dind+N_d*a2ind+N_d*N_a2*bothzind;
                d12_ind=rem(dind-1,N_d12)+1;
                Policy3alt(1,curraindex,:,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy3alt(2,curraindex,:,N_j)=ceil(d12_ind/N_d1);
                Policy3alt(3,curraindex,:,N_j)=ceil(dind/N_d12);
                Policy3alt(4,curraindex,:,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,n_d3],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);
            [~,maxindex1]=max(ReturnMatrix_ii_z,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_z,[N_d*N_a1,vfoptions.level1n*N_a2]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            Valt(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
            dind=rem(maxindex2-1,N_d)+1;
            d12_ind=rem(dind-1,N_d12)+1;
            Policy3alt(1,curraindex,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3alt(2,curraindex,z_c,N_j)=ceil(d12_ind/N_d1);
            Policy3alt(3,curraindex,z_c,N_j)=ceil(dind/N_d12);
            Policy3alt(4,curraindex,z_c,N_j)=ceil(maxindex2/N_d);

            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,n_d3],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Valt(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d*a2ind;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy3alt(1,curraindex,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
                    Policy3alt(2,curraindex,z_c,N_j)=ceil(d12_ind/N_d1);
                    Policy3alt(3,curraindex,z_c,N_j)=ceil(dind/N_d12);
                    Policy3alt(4,curraindex,z_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,n_d3],1,level1iidiff(ii),n_a2,special_n_bothz, d_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii_z,[],1);
                    Valt(curraindex,z_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d*a2ind;
                    d12_ind=rem(dind-1,N_d12)+1;
                    Policy3alt(1,curraindex,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
                    Policy3alt(2,curraindex,z_c,N_j)=ceil(d12_ind/N_d1);
                    Policy3alt(3,curraindex,z_c,N_j)=ceil(dind/N_d12);
                    Policy3alt(4,curraindex,z_c,N_j)=ceil(maxindex/N_d+loweredge(allind)-1);
                end
            end
        end
    end
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy3(:,:,:,N_j)=Policy3alt(:,:,:,N_j);
else
    % aprime depends on (d2, a1, a2, current semiz); independent of d3 and z
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2semiz=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=aprimeIndex; % [N_d2*N_a1, N_a2, N_bothz]
    aprimeplus1Index_full=aprimeplus1Index;
    aprimeProbs_full=aprimeProbs_d2a1a2semiz;

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

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

            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);   % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);   % QH-perceived

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            % Valt (beta): the exponential value

            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii_d3,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(maxindex2alt,1);

            maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_d3_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3_dc+DiscountedEV_alt(d2aprimez),[N_d12*(maxgap_V(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allindalt=dindalt+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind;
                    Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii_d3_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3_dc+DiscountedEV_alt(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allindalt=dindalt+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind;
                    Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                end
            end

            % Vtilde (beta0*beta): the QH-perceived value

            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3_dc+DiscountedEV_tilde(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind;
                    Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii_d3_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3_dc+DiscountedEV_tilde(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind;
                    Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                EV_z=V_Jplus1.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                semiz_part=rem(z_c-1,N_semiz)+1;
                aprime_slice=aprimeIndex(:,:,semiz_part);
                aprimeplus1_slice=aprimeplus1Index(:,:,semiz_part);
                aprimeProbs_slice=aprimeProbs_d2a1a2semiz(:,:,semiz_part);

                EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                DiscountedEV_z_alt=beta*reshape(entireEV_z,[N_d2,N_a1,1,N_a2]);   % exponential
                DiscountedEV_z_tilde=beta0beta*reshape(entireEV_z,[N_d2,N_a1,1,N_a2]);   % QH-perceived

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);

                % Valt (beta): the exponential value

                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(DiscountedEV_z_alt,N_d1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii_z,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(maxindex2alt,1);

                maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap_V(ii));
                        a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                        ReturnMatrix_ii_z_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii_z_dc+DiscountedEV_z_alt(d2aprime),[N_d12*(maxgap_V(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindexalt]=max(entireRHS_ii_z,[],1);
                        V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allindalt=dindalt+N_d1*N_d2*a2ind;
                        Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii_z_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii_z_dc+DiscountedEV_z_alt(d2aprime),[N_d12,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindexalt]=max(entireRHS_ii_z,[],1);
                        V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allindalt=dindalt+N_d1*N_d2*a2ind;
                        Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                    end
                end

                % Vtilde (beta0*beta): the QH-perceived value

                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(DiscountedEV_z_tilde,N_d1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_z_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii_z_dc+DiscountedEV_z_tilde(d2aprime),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind;
                        Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii_z_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii_z_dc+DiscountedEV_z_tilde(d2aprime),[N_d12,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind;
                        Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    end

    % Max over d3 (alt: beta -- the exponential pass)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,N_j)=V_jj;
    Policy3alt(3,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3alt(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
    Policy3alt(2,:,:,N_j)=ceil(d12_ind/N_d1);
    Policy3alt(4,:,:,N_j)=ceil(d12a1prime_ind/N_d12);

    % Max over d3 (tilde: beta0*beta -- the QH-perceived pass)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy3(3,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
    Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1);
    Policy3(4,:,:,N_j)=ceil(d12a1prime_ind/N_d12);
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
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2semiz=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=aprimeIndex;
    aprimeplus1Index_full=aprimeplus1Index;
    aprimeProbs_full=aprimeProbs_d2a1a2semiz;

    EVpre=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

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

            DiscountedEV_alt=beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);   % exponential
            DiscountedEV_tilde=beta0beta*reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);   % QH-perceived

            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            % Valt (beta): the exponential value

            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_alt,N_d1,1,1,1,1);

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii_d3,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(maxindex2alt,1);

            maxgap_V=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap_V(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap_V(ii));
                    a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                    ReturnMatrix_ii_d3_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3_dc+DiscountedEV_alt(d2aprimez),[N_d12*(maxgap_V(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allindalt=dindalt+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind;
                    Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii_d3_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3_dc+DiscountedEV_alt(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindexalt]=max(entireRHS_ii,[],1);
                    V_ford3_alt(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allindalt=dindalt+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind;
                    Policy_ford3_alt(curraindex,:,d3_c)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                end
            end

            % Vtilde (beta0*beta): the QH-perceived value

            entireRHS_ii_d3=ReturnMatrix_ii_d3+repelem(DiscountedEV_tilde,N_d1,1,1,1,1);

            [~,maxindex1]=max(entireRHS_ii_d3,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_d3,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2,N_bothz]),[],1);

            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
            Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(maxindex2,1);

            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii_d3_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3_dc+DiscountedEV_tilde(d2aprimez),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind;
                    Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii_d3_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,n_bothz, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimez=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2)+N_d2*N_a1*N_a2*shiftdim((0:1:N_bothz-1),-3);
                    entireRHS_ii=reshape(ReturnMatrix_ii_d3_dc+DiscountedEV_tilde(d2aprimez),[N_d12,level1iidiff(ii)*N_a2,N_bothz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford3_tilde(curraindex,:,d3_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1*N_d2)+1);
                    a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                    allind=dind+N_d1*N_d2*a2ind+N_d1*N_d2*N_a2*bothzind;
                    Policy_ford3_tilde(curraindex,:,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                end
            end
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                semiz_part=rem(z_c-1,N_semiz)+1;
                aprime_slice=aprimeIndex(:,:,semiz_part);
                aprimeplus1_slice=aprimeplus1Index(:,:,semiz_part);
                aprimeProbs_slice=aprimeProbs_d2a1a2semiz(:,:,semiz_part);

                EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                DiscountedEV_z_alt=beta*reshape(entireEV_z,[N_d2,N_a1,1,N_a2]);   % exponential
                DiscountedEV_z_tilde=beta0beta*reshape(entireEV_z,[N_d2,N_a1,1,N_a2]);   % QH-perceived

                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, ReturnFnParamsVec,1,0);

                % Valt (beta): the exponential value

                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(DiscountedEV_z_alt,N_d1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2alt]=max(reshape(entireRHS_ii_z,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(maxindex2alt,1);

                maxgap_V=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap_V(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap_V(ii));
                        a1primeindexes=loweredge+(0:1:maxgap_V(ii));
                        ReturnMatrix_ii_z_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap_V(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii_z_dc+DiscountedEV_z_alt(d2aprime),[N_d12*(maxgap_V(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindexalt]=max(entireRHS_ii_z,[],1);
                        V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allindalt=dindalt+N_d1*N_d2*a2ind;
                        Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii_z_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii_z_dc+DiscountedEV_z_alt(d2aprime),[N_d12,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindexalt]=max(entireRHS_ii_z,[],1);
                        V_ford3_alt(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dindalt=(rem(maxindexalt-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allindalt=dindalt+N_d1*N_d2*a2ind;
                        Policy_ford3_alt(curraindex,z_c,d3_c)=shiftdim(maxindexalt+N_d1*N_d2*(loweredge(allindalt)-1),1);
                    end
                end

                % Vtilde (beta0*beta): the QH-perceived value

                entireRHS_ii_z=ReturnMatrix_ii_z+repelem(DiscountedEV_z_tilde,N_d1,1,1,1);

                [~,maxindex1]=max(entireRHS_ii_z,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_z,[N_d1*N_d2*N_a1,vfoptions.level1n*N_a2]),[],1);

                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(maxindex2,1);

                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii_z_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime=d2ind+N_d2*(a1primeindexes-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii_z_dc+DiscountedEV_z_tilde(d2aprime),[N_d12*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind;
                        Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii_z_dc=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],1,level1iidiff(ii),n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, ReturnFnParamsVec,3,0);
                        d2aprime=d2ind+N_d2*(loweredge-1)+N_d2*N_a1*shiftdim((0:1:N_a2-1),-2);
                        entireRHS_ii_z=reshape(ReturnMatrix_ii_z_dc+DiscountedEV_z_tilde(d2aprime),[N_d12,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_z,[],1);
                        V_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1*N_d2)+1);
                        a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii));
                        allind=dind+N_d1*N_d2*a2ind;
                        Policy_ford3_tilde(curraindex,z_c,d3_c)=shiftdim(maxindex+N_d1*N_d2*(loweredge(allind)-1),1);
                    end
                end
            end
        end
    end

    % Max over d3 (alt: beta -- the exponential pass)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,jj)=V_jj;
    Policy3alt(3,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3alt(1,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy3alt(2,:,:,jj)=ceil(d12_ind/N_d1);
    Policy3alt(4,:,:,jj)=ceil(d12a1prime_ind/N_d12);

    % Max over d3 (tilde: beta0*beta -- the QH-perceived pass)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy3(3,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy3(2,:,:,jj)=ceil(d12_ind/N_d1);
    Policy3(4,:,:,jj)=ceil(d12a1prime_ind/N_d12);
end


end
