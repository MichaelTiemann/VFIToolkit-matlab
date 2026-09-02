function [V,Policy]=ValueFnIter_FHorz_AmbAverse_RiskyAsset_nod1_noz_e_raw(n_ambiguity, n_d2,n_d3,n_a1,n_a2,n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, e_gridvals_J, u_grid, ambiguity_pi_e_J, ambiguity_pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Ambiguity aversion: multiple priors over ambiguity_pi_u (the risky return distribution is ambiguous, not known risk) and pi_e;
% the continuation is the worst case at each expectation stage, with the aprime lottery conditional on the prior.
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_e=prod(n_e);
N_u=prod(n_u);

N_d=N_d2*N_d3;
N_a=N_a1*N_a2;

% For ReturnFn
% n_d3
% N_d3
% d3_grid
% For aprimeFn
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_e,N_j,'gpuArray'); % d1,d2,d3,a1prime

%%

d3a1_gridvals=CreateGridvals([n_d3,n_a1],[d3_grid;a1_grid],1);
a1a2_gridvals=CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_e, d3a1_gridvals, a1a2_gridvals,e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(1,:,:,N_j)=1; % is meaningless anyway
        Policy(2,:,:,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_e, d3a1_gridvals, a1a2_gridvals, e_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(1,:,e_c,N_j)=1; % is meaningless anyway
            Policy(2,:,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    ambEVstack=[]; % one slice per e-prior (the aprime lottery below is conditional on the prior)
    for amb_c0=1:n_ambiguity(N_j)
        EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*ambiguity_pi_e_J(:,N_j+1,amb_c0)',2);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        ambEVstack=cat(3,ambEVstack,EV);
    end

    % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
    % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
    for amb_c=1:n_ambiguity(N_j)
        EV=ambEVstack(:,:,amb_c);
        a2primeProbsK=a2primeProbs;

        % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbsK is [N_d,N_u]

        aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
        % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)
        aprimeProbsK=repmat(a2primeProbsK,N_a1,1);  % [N_d*N_a1,N_u]

        % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        EVlower=reshape(EV(aprimeIndex),[N_d23*N_a1,N_u]); % the lower aprime
        EVupper=reshape(EV(aprimeplus1Index),[N_d23*N_a1,N_u]); % the upper aprime
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(EVlower==EVupper);
        aprimeProbsK(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=aprimeProbsK.*EVlower+(1-aprimeProbsK).*EVupper; % (d23 & a1prime,u,zprime)
        % Already applied the probabilities from interpolating onto grid
        if amb_c==1
            Mmin=EV;
        else
            Mmin=min(Mmin,EV);
        end
    end
    % Worst case over the u-priors (the ambiguous risky return distribution)
    EV=squeeze(sum((Mmin.*shiftdim(ambiguity_pi_u(:,1),-1)),2));
    for amb_cu=2:n_ambiguity(N_j)
        EV=min(EV,squeeze(sum((Mmin.*shiftdim(ambiguity_pi_u(:,amb_cu),-1)),2)));
    end

    % Time to refine EV, we can refine out d2
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1]),[],1);

    DiscountedEV_onlyd3=DiscountFactorParamsVec*shiftdim(EV_onlyd3,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2],n_e, d3a1_gridvals, a1a2_gridvals,e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,a,e)

        % Time to refine ReturnMatrix, we can refine out d1
        % no d1 here

        % Now put together entireRHS, which just depends on d3
        entireRHS=ReturnMatrix+DiscountedEV_onlyd3;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(2,:,:,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy(1,:,:,N_j)=shiftdim(d2index(maxindex),1);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values

       for e_c=1:N_e
           e_val=e_gridvals_J(e_c,:,N_j);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_e, d3a1_gridvals, a1a2_gridvals, e_val, ReturnFnParamsVec);

           % Time to refine ReturnMatrix, we can refine out d1
           % no d1 here

           % Now put together entireRHS, which just depends on d3
           entireRHS_e=ReturnMatrix_e+DiscountedEV_onlyd3;

           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e,[],1);
           V(:,e_c,N_j)=Vtemp;
           Policy(2,:,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
           Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
           Policy(1,:,e_c,N_j)=shiftdim(d2index(maxindex),1);
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
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    ambEVstack=[]; % one slice per e-prior (the aprime lottery below is conditional on the prior)
    for amb_c0=1:n_ambiguity(jj)
        EV=sum(V(:,:,jj+1).*ambiguity_pi_e_J(:,jj+1,amb_c0)',2);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        ambEVstack=cat(3,ambEVstack,EV);
    end

    % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
    % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
    for amb_c=1:n_ambiguity(jj)
        EV=ambEVstack(:,:,amb_c);
        a2primeProbsK=a2primeProbs;

        % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbsK is [N_d,N_u]


        aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
        % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)
        aprimeProbsK=repmat(a2primeProbsK,N_a1,1);  % [N_d*N_a1,N_u]

        % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
        EVlower=reshape(EV(aprimeIndex),[N_d23*N_a1,N_u]); % the lower aprime
        EVupper=reshape(EV(aprimeplus1Index),[N_d23*N_a1,N_u]); % the upper aprime
        % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
        skipinterp=(EVlower==EVupper);
        aprimeProbsK(skipinterp)=0; % effectively skips interpolation

        % Switch EV from being in terms of a2prime to being in terms of d2 and a2
        EV=aprimeProbsK.*EVlower+(1-aprimeProbsK).*EVupper; % (d23 & a1prime,u,zprime)
        % Already applied the probabilities from interpolating onto grid
        if amb_c==1
            Mmin=EV;
        else
            Mmin=min(Mmin,EV);
        end
    end
    % Worst case over the u-priors (the ambiguous risky return distribution)
    EV=squeeze(sum((Mmin.*shiftdim(ambiguity_pi_u(:,1),-1)),2));
    for amb_cu=2:n_ambiguity(jj)
        EV=min(EV,squeeze(sum((Mmin.*shiftdim(ambiguity_pi_u(:,amb_cu),-1)),2)));
    end

    % Time to refine EV, we can refine out d2
    [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1]),[],1);

    DiscountedEV_onlyd3=DiscountFactorParamsVec*shiftdim(EV_onlyd3,1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], n_e, d3a1_gridvals, a1a2_gridvals, e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,a,e)

        % Time to refine ReturnMatrix, we can refine out d1
        % no d1 here

        % Now put together entireRHS, which just depends on d3
        entireRHS=ReturnMatrix+DiscountedEV_onlyd3;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(2,:,:,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy(3,:,:,jj)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy(1,:,:,jj)=shiftdim(d2index(maxindex),1);

    elseif vfoptions.lowmemory>=1 % lm1 already does the most-looped variant, so it also serves the higher lowmemory values

       for e_c=1:N_e
           e_val=e_gridvals_J(e_c,:,jj);
           ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d3,n_a1], [n_a1,n_a2], special_n_e, d3a1_gridvals, a1a2_gridvals, e_val, ReturnFnParamsVec);

           % Time to refine ReturnMatrix, we can refine out d1
           % no d1 here

           % Now put together entireRHS, which just depends on d3
           entireRHS_e=ReturnMatrix_e+DiscountedEV_onlyd3;

           %Calc the max and it's index
           [Vtemp,maxindex]=max(entireRHS_e,[],1);

           V(:,e_c,jj)=Vtemp;
           Policy(2,:,e_c,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
           Policy(3,:,e_c,jj)=shiftdim(ceil(maxindex/N_d3),-1);
           Policy(1,:,e_c,jj)=shiftdim(d2index(maxindex),1);
        end
    end
end




end
