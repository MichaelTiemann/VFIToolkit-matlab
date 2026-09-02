function [V,Policy]=ValueFnIter_FHorz_AmbAverse_RiskyAsset_DC1_nod1_e_raw(n_ambiguity, n_d2,n_d3,n_a1,n_a2,n_z,n_e,n_u,N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, e_gridvals_J, u_grid, ambiguity_pi_z_J, ambiguity_pi_e_J, ambiguity_pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Ambiguity aversion: multiple priors over ambiguity_pi_u (the risky return distribution is ambiguous, not known risk), pi_z and pi_e;
% the continuation is the worst case at each expectation stage, with the aprime lottery conditional on the prior.
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% No d1, with e.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_z,N_e,N_j,'gpuArray'); % (1)=d2, (2)=d3, (3)=a1prime
% We will refine away d2 out of EV before combining with ReturnFn

%%
u_grid=gpuArray(u_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
a1_gridvals=a1_grid;
d3_gridvals=CreateGridvals(n_d3,d3_grid,1);

if vfoptions.lowmemory==0
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1);
    eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
elseif vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
    zBind=shiftdim(gpuArray(0:1:N_z-1),-1);
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

% Setup for DC
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Precompute
a2Bind=gpuArray(0:1:N_a2-1);
d3ind=(1:1:N_d3)';

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d3*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        pol_d3_a1=shiftdim(maxindex2,1); % (npts,N_z,N_e)
        Policy(2,curraindex,:,:,N_j)=rem(pol_d3_a1-1,N_d3)+1; % d3
        Policy(3,curraindex,:,:,N_j)=ceil(pol_d3_a1/N_d3);    % a1prime

        % Divide-and-conquer layer 2
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_z*eBind;
                pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                Policy(2,curraindex,:,:,N_j)=rem(pol_d3_a1-1,N_d3)+1;
                Policy(3,curraindex,:,:,N_j)=ceil(pol_d3_a1/N_d3);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,n_z,n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
                [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_z*eBind;
                pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                Policy(2,curraindex,:,:,N_j)=rem(pol_d3_a1-1,N_d3)+1;
                Policy(3,curraindex,:,:,N_j)=ceil(pol_d3_a1/N_d3);
            end
        end
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            [~,maxindex1]=max(ReturnMatrix_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_e,[N_d3*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            pol_d3_a1=shiftdim(maxindex2,1);
            Policy(2,curraindex,:,e_c,N_j)=rem(pol_d3_a1-1,N_d3)+1;
            Policy(3,curraindex,:,e_c,N_j)=ceil(pol_d3_a1/N_d3);

            % Divide-and-conquer layer 2
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                    pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    Policy(2,curraindex,:,e_c,N_j)=rem(pol_d3_a1-1,N_d3)+1;
                    Policy(3,curraindex,:,e_c,N_j)=ceil(pol_d3_a1/N_d3);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                    pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    Policy(2,curraindex,:,e_c,N_j)=rem(pol_d3_a1-1,N_d3)+1;
                    Policy(3,curraindex,:,e_c,N_j)=ceil(pol_d3_a1/N_d3);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                [~,maxindex1]=max(ReturnMatrix_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii_ze,[N_d3*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                pol_d3_a1=shiftdim(maxindex2,1); % (npts,1)
                Policy(2,curraindex,z_c,e_c,N_j)=rem(pol_d3_a1-1,N_d3)+1;
                Policy(3,curraindex,z_c,e_c,N_j)=ceil(pol_d3_a1/N_d3);

                % Divide-and-conquer layer 2
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    end
                    Policy(2,curraindex,z_c,e_c,N_j)=rem(pol_d3_a1-1,N_d3)+1;
                    Policy(3,curraindex,z_c,e_c,N_j)=ceil(pol_d3_a1/N_d3);
                end
            end
        end
    end

    % d2, which was not in ReturnFn
    Policy(1,:,:,:,N_j)=ones(1,N_a,N_z,N_e,'gpuArray'); % d2 (terminal: d2 doesn't matter since it's only in the expectations term)

else % V_Jplus1

    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j));

    % Build a2primeIndex and a2primeProbs for RisykAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Get EV in terms of next period endogenous states
    EVpree=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);
    for amb_ce=1:n_ambiguity(N_j) % evaluate the iid-e expectation under each of the multiple priors (running worst case)
        EVe=sum(EVpree.*shiftdim(ambiguity_pi_e_J(:,N_j+1,amb_ce),-2),3);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        if amb_ce==1
            EVnext=EVe;
        else
            EVnext=min(EVnext,EVe);
        end
    end % EVnext is now the worst case over the e-priors; the z expectation is next
    ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
    for amb_c0=1:n_ambiguity(N_j)
        EV=EVnext.*shiftdim(ambiguity_pi_z_J(:,:,N_j,amb_c0)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        ambEVstack=cat(4,ambEVstack,EV);
    end

    % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
    % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
    for amb_c=1:n_ambiguity(N_j)
        EV=ambEVstack(:,:,:,amb_c);
        a2primeProbsK=a2primeProbs;
        EV=reshape(EV,[N_a,N_z]);

        % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
        skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
        aprimeProbsK=repmat(a2primeProbsK,N_a1,N_z);
        aprimeProbsK(skipinterp)=0;
        aprimeProbsK=reshape(aprimeProbsK,[N_d23*N_a1,N_u,N_z]);
        % Take the expectation over the between period iid u shock
        EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbsK;
        EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbsK);
        if amb_c==1
            Mmin=EV1+EV2; EV1sel=EV1; EV2sel=EV2;
        else
            Mk=EV1+EV2;
            newmin=(Mk<Mmin);
            Mmin(newmin)=Mk(newmin);
            EV1sel(newmin)=EV1(newmin);
            EV2sel(newmin)=EV2(newmin);
        end
    end
    % Worst case over the u-priors (the ambiguous risky return distribution)
    EV=sum(EV1sel.*ambiguity_pi_u(:,1)',2)+sum(EV2sel.*ambiguity_pi_u(:,1)',2);
    for amb_cu=2:n_ambiguity(N_j)
        EV=min(EV,sum(EV1sel.*ambiguity_pi_u(:,amb_cu)',2)+sum(EV2sel.*ambiguity_pi_u(:,amb_cu)',2));
    end
    EV=reshape(EV,[N_d23*N_a1,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,N_z]),[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
        pol_d3_a1=shiftdim(maxindex2,1); % (npts,N_z,N_e)
        d3part=rem(pol_d3_a1-1,N_d3)+1;
        a1primepart=ceil(pol_d3_a1/N_d3);
        Policy(2,curraindex,:,:,N_j)=d3part;
        Policy(3,curraindex,:,:,N_j)=a1primepart;
        % Get the d2Policy
        [npts,nz,ne]=size(pol_d3_a1);
        zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
        Policy(1,curraindex,:,:,N_j)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_z*eBind;
                pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                d3part=rem(pol_d3_a1-1,N_d3)+1;
                a1primepart=ceil(pol_d3_a1/N_d3);
                Policy(2,curraindex,:,:,N_j)=d3part;
                Policy(3,curraindex,:,:,N_j)=a1primepart;
                % Get the d2Policy
                [npts,nz,ne]=size(pol_d3_a1);
                zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(1,curraindex,:,:,N_j)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,n_z,n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,N_j)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_z*eBind;
                pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                d3part=rem(pol_d3_a1-1,N_d3)+1;
                a1primepart=ceil(pol_d3_a1/N_d3);
                Policy(2,curraindex,:,:,N_j)=d3part;
                Policy(3,curraindex,:,:,N_j)=a1primepart;
                % Get the d2Policy
                [npts,nz,ne]=size(pol_d3_a1);
                zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(1,curraindex,:,:,N_j)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d3*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
            pol_d3_a1=shiftdim(maxindex2,1); % (npts,N_z)
            d3part=rem(pol_d3_a1-1,N_d3)+1;
            a1primepart=ceil(pol_d3_a1/N_d3);
            Policy(2,curraindex,:,e_c,N_j)=d3part;
            Policy(3,curraindex,:,e_c,N_j)=a1primepart;
            % Get the d2Policy
            [npts,nz]=size(pol_d3_a1);
            zidx=repmat(gpuArray(1:nz),npts,1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            Policy(1,curraindex,:,e_c,N_j)=d2index_resh(lin);

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                    pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    Policy(2,curraindex,:,e_c,N_j)=d3part;
                    Policy(3,curraindex,:,e_c,N_j)=a1primepart;
                    % Get the d2Policy
                    [npts,nz]=size(pol_d3_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    Policy(1,curraindex,:,e_c,N_j)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,N_j)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                    pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    Policy(2,curraindex,:,e_c,N_j)=d3part;
                    Policy(3,curraindex,:,e_c,N_j)=a1primepart;
                    % Get the d2Policy
                    [npts,nz]=size(pol_d3_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    Policy(1,curraindex,:,e_c,N_j)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c); % [N_d3,N_a1]
            d2index_z=d2index_resh(:,:,z_c);          % [N_d3,N_a1]
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_z;

                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d3*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                pol_d3_a1=shiftdim(maxindex2,1); % (npts,1)
                d3part=rem(pol_d3_a1-1,N_d3)+1;
                a1primepart=ceil(pol_d3_a1/N_d3);
                Policy(2,curraindex,z_c,e_c,N_j)=d3part;
                Policy(3,curraindex,z_c,e_c,N_j)=a1primepart;
                Policy(1,curraindex,z_c,e_c,N_j)=d2index_z(d3part+N_d3*(a1primepart-1));

                % Divide and conquer layer 2
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,N_j)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    end
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    Policy(2,curraindex,z_c,e_c,N_j)=d3part;
                    Policy(3,curraindex,z_c,e_c,N_j)=a1primepart;
                    Policy(1,curraindex,z_c,e_c,N_j)=d2index_z(d3part+N_d3*(a1primepart-1));
                end
            end
        end
    end
end

%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj));

    % Build a2primeIndex and a2primeProbs for RisykAsset
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2);
    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1);

    % Get EV in terms of next period endogenous states
    EVpree=V(:,:,:,jj+1); % iid-e expectation of V is taken first
    for amb_ce=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors (running worst case)
        EVe=sum(EVpree.*shiftdim(ambiguity_pi_e_J(:,jj+1,amb_ce),-2),3);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        if amb_ce==1
            EVnext=EVe;
        else
            EVnext=min(EVnext,EVe);
        end
    end % EVnext is now the worst case over the e-priors; the z expectation is next
    ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
    for amb_c0=1:n_ambiguity(jj)
        EV=EVnext.*shiftdim(ambiguity_pi_z_J(:,:,jj,amb_c0)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        ambEVstack=cat(4,ambEVstack,EV);
    end

    % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
    % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
    for amb_c=1:n_ambiguity(jj)
        EV=ambEVstack(:,:,:,amb_c);
        a2primeProbsK=a2primeProbs;
        EV=reshape(EV,[N_a,N_z]);

        % Interpolate EV onto aprime, use skipinterp to avoid numerical errors where the lower and upper points are identical
        skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)));
        aprimeProbsK=repmat(a2primeProbsK,N_a1,N_z);
        aprimeProbsK(skipinterp)=0;
        aprimeProbsK=reshape(aprimeProbsK,[N_d23*N_a1,N_u,N_z]);
        % Take the expectation over the between period iid u shock
        EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*aprimeProbsK;
        EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)),[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbsK);
        if amb_c==1
            Mmin=EV1+EV2; EV1sel=EV1; EV2sel=EV2;
        else
            Mk=EV1+EV2;
            newmin=(Mk<Mmin);
            Mmin(newmin)=Mk(newmin);
            EV1sel(newmin)=EV1(newmin);
            EV2sel(newmin)=EV2(newmin);
        end
    end
    % Worst case over the u-priors (the ambiguous risky return distribution)
    EV=sum(EV1sel.*ambiguity_pi_u(:,1)',2)+sum(EV2sel.*ambiguity_pi_u(:,1)',2);
    for amb_cu=2:n_ambiguity(jj)
        EV=min(EV,sum(EV1sel.*ambiguity_pi_u(:,amb_cu)',2)+sum(EV2sel.*ambiguity_pi_u(:,amb_cu)',2));
    end
    EV=reshape(EV,[N_d23*N_a1,N_z]);

    % Refine d2 out of EV before combining with ReturnFn
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,N_z]),[],1);
    EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a1,N_z]);
    d2index_resh=reshape(d2index,[N_d3,N_a1,N_z]);

    % DiscountedEV
    DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,1,1,N_z]);

    if vfoptions.lowmemory==0
        % Layer 1
        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
        entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

        [~,maxindex1]=max(entireRHS_ii,[],2);
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d3*N_a1,vfoptions.level1n*N_a2,N_z,N_e]),[],1);
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
        V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
        pol_d3_a1=shiftdim(maxindex2,1);
        d3part=rem(pol_d3_a1-1,N_d3)+1;
        a1primepart=ceil(pol_d3_a1/N_d3);
        Policy(2,curraindex,:,:,jj)=d3part;
        Policy(3,curraindex,:,:,jj)=a1primepart;
        % Get the d2Policy
        [npts,nz,ne]=size(pol_d3_a1);
        zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
        lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
        Policy(1,curraindex,:,:,jj)=d2index_resh(lin);

        % Divide and conquer layer 2
        maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                a1primeindexes=loweredge+(0:1:maxgap(ii));
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_z*eBind;
                pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                d3part=rem(pol_d3_a1-1,N_d3)+1;
                a1primepart=ceil(pol_d3_a1/N_d3);
                Policy(2,curraindex,:,:,jj)=d3part;
                Policy(3,curraindex,:,:,jj)=a1primepart;
                % Get the d2Policy
                [npts,nz,ne]=size(pol_d3_a1);
                zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(1,curraindex,:,:,jj)=d2index_resh(lin);
            else
                loweredge=maxindex1(:,1,ii,:,:,:);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,n_z,n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_z,N_e]);
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,:,jj)=shiftdim(Vtempii,1);
                dind=(rem(maxindex-1,N_d3)+1);
                allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind+N_d3*N_a2*N_z*eBind;
                pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                d3part=rem(pol_d3_a1-1,N_d3)+1;
                a1primepart=ceil(pol_d3_a1/N_d3);
                Policy(2,curraindex,:,:,jj)=d3part;
                Policy(3,curraindex,:,:,jj)=a1primepart;
                % Get the d2Policy
                [npts,nz,ne]=size(pol_d3_a1);
                zidx=repmat(gpuArray(reshape(1:nz,[1,nz,1])),npts,1,ne);
                lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                Policy(1,curraindex,:,:,jj)=d2index_resh(lin);
            end
        end

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % Layer 1
            ReturnMatrix_ii_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii_e,[],2);
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii_e,[N_d3*N_a1,vfoptions.level1n*N_a2,N_z]),[],1);
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
            V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
            pol_d3_a1=shiftdim(maxindex2,1);
            d3part=rem(pol_d3_a1-1,N_d3)+1;
            a1primepart=ceil(pol_d3_a1/N_d3);
            Policy(2,curraindex,:,e_c,jj)=d3part;
            Policy(3,curraindex,:,e_c,jj)=a1primepart;
            % Get the d2Policy
            [npts,nz]=size(pol_d3_a1);
            zidx=repmat(gpuArray(1:nz),npts,1);
            lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
            Policy(1,curraindex,:,e_c,jj)=d2index_resh(lin);

            % Divide and conquer layer 2
            maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(a1primeindexes-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                    pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    Policy(2,curraindex,:,e_c,jj)=d3part;
                    Policy(3,curraindex,:,e_c,jj)=a1primepart;
                    % Get the d2Policy
                    [npts,nz]=size(pol_d3_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    Policy(1,curraindex,:,e_c,jj)=d2index_resh(lin);
                else
                    loweredge=maxindex1(:,1,ii,:,:);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,n_z,special_n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                    d3aprimez=d3ind+N_d3*(loweredge-1)+N_d3*N_a1*shiftdim(zBind,-2);
                    entireRHS_ii_e=reshape(ReturnMatrix_ii+DiscountedEV(d3aprimez),[N_d3,level1iidiff(ii)*N_a2,N_z]);
                    [Vtempii,maxindex]=max(entireRHS_ii_e,[],1);
                    V(curraindex,:,e_c,jj)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d3)+1);
                    allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii))+N_d3*N_a2*zBind;
                    pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    Policy(2,curraindex,:,e_c,jj)=d3part;
                    Policy(3,curraindex,:,e_c,jj)=a1primepart;
                    % Get the d2Policy
                    [npts,nz]=size(pol_d3_a1);
                    zidx=repmat(gpuArray(1:nz),npts,1);
                    lin=d3part+N_d3*(a1primepart-1)+N_d3*N_a1*(zidx-1);
                    Policy(1,curraindex,:,e_c,jj)=d2index_resh(lin);
                end
            end
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c); % [N_d3,N_a1]
            d2index_z=d2index_resh(:,:,z_c);          % [N_d3,N_a1]
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                % Layer 1
                ReturnMatrix_ii_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,n_a1,vfoptions.level1n,n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                entireRHS_ii_ze=ReturnMatrix_ii_ze+DiscountedEV_z;

                [~,maxindex1]=max(entireRHS_ii_ze,[],2);
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii_ze,[N_d3*N_a1,vfoptions.level1n*N_a2]),[],1);
                curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
                V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                pol_d3_a1=shiftdim(maxindex2,1); % (npts,1)
                d3part=rem(pol_d3_a1-1,N_d3)+1;
                a1primepart=ceil(pol_d3_a1/N_d3);
                Policy(2,curraindex,z_c,e_c,jj)=d3part;
                Policy(3,curraindex,z_c,e_c,jj)=a1primepart;
                Policy(1,curraindex,z_c,e_c,jj)=d2index_z(d3part+N_d3*(a1primepart-1));

                % Divide and conquer layer 2
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        d3aprime=d3ind+N_d3*(a1primeindexes-1);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3*(maxgap(ii)+1),level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d3,1,level1iidiff(ii),n_a2,special_n_z,special_n_e, d3_gridvals, a1_gridvals(loweredge), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                        d3aprime=d3ind+N_d3*(loweredge-1);
                        entireRHS_ii_ze=reshape(ReturnMatrix_ii+DiscountedEV_z(d3aprime),[N_d3,level1iidiff(ii)*N_a2]);
                        [Vtempii,maxindex]=max(entireRHS_ii_ze,[],1);
                        V(curraindex,z_c,e_c,jj)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d3)+1);
                        allind=dind+N_d3*repelem(a2Bind,1,level1iidiff(ii));
                        pol_d3_a1=shiftdim(maxindex+N_d3*(loweredge(allind)-1),1);
                    end
                    d3part=rem(pol_d3_a1-1,N_d3)+1;
                    a1primepart=ceil(pol_d3_a1/N_d3);
                    Policy(2,curraindex,z_c,e_c,jj)=d3part;
                    Policy(3,curraindex,z_c,e_c,jj)=a1primepart;
                    Policy(1,curraindex,z_c,e_c,jj)=d2index_z(d3part+N_d3*(a1primepart-1));
                end
            end
        end
    end
end


end
