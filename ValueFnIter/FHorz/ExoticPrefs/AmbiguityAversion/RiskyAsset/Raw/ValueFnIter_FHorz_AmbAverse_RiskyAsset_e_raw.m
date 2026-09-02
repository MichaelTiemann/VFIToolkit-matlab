function [V,Policy]=ValueFnIter_FHorz_AmbAverse_RiskyAsset_e_raw(n_ambiguity, n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, e_gridvals_J, u_grid, ambiguity_pi_z_J, ambiguity_pi_e_J, ambiguity_pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Ambiguity aversion: multiple priors over ambiguity_pi_u (the risky return distribution is ambiguous, not known risk), pi_z and pi_e;
% the continuation is the worst case at each expectation stage, with the aprime lottery conditional on the prior.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(n_e);
N_u=prod(n_u);

N_d=N_d1*N_d2*N_d3;
N_a=N_a1*N_a2;

% For ReturnFn
n_d13=[n_d1,n_d3];
% For aprimeFn
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_z,N_e,N_j,'gpuArray'); % d1,d2,d3,a1prime

%%
u_grid=gpuArray(u_grid);

d13a1_gridvals=CreateGridvals([n_d1,n_d3,n_a1],[d1_grid;d3_grid;a1_grid],1);
a12_gridvals=CreateGridvals([n_a1,n_a2],[a1_grid;a2_grid],1);

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
end

aind=gpuArray(0:1:N_a-1);
zind=shiftdim(gpuArray(0:1:N_z-1),-1);
eind=shiftdim(gpuArray(0:1:N_e-1),-2);


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

ambiguity_pi_e_J=shiftdim(ambiguity_pi_e_J,-2); % Move to third dimension

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], n_z, n_e, d13a1_gridvals, a12_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        dindex=rem(maxindex-1,N_d)+1;
        Policy(1,:,:,:,N_j)=shiftdim(rem(dindex-1,N_d1)+1,-1);
        Policy(2,:,:,:,N_j)=1; % is meaningless anyway
        Policy(3,:,:,:,N_j)=shiftdim(ceil(dindex/N_d1),-1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d),-1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], n_z, special_n_e, d13a1_gridvals, a12_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            dindex=rem(maxindex-1,N_d)+1;
            Policy(1,:,:,e_c,N_j)=shiftdim(rem(dindex-1,N_d1)+1,-1);
            Policy(2,:,:,e_c,N_j)=1; % is meaningless anyway
            Policy(3,:,:,e_c,N_j)=shiftdim(ceil(dindex/N_d1),-1);
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
        end

    elseif vfoptions.lowmemory==2

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_z
                z_val=z_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], special_n_z, special_n_e, d13a1_gridvals, a12_gridvals, z_val, e_val, ReturnFnParamsVec);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                dindex=rem(maxindex-1,N_d)+1;
                Policy(1,:,z_c,e_c,N_j)=shiftdim(rem(dindex-1,N_d1)+1,-1);
                Policy(2,:,z_c,e_c,N_j)=1; % is meaningless anyway
                Policy(3,:,z_c,e_c,N_j)=shiftdim(ceil(dindex/N_d1),-1);
                Policy(4,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/N_d),-1);
            end
        end

    end
else
    % Using V_Jplus1
    EVpree=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
    % aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)

    for amb_ce=1:n_ambiguity(N_j) % evaluate the iid-e expectation under each of the multiple priors (running worst case)
        EVe=sum(EVpree.*ambiguity_pi_e_J(1,1,:,N_j+1,amb_ce),3);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        if amb_ce==1
            EVpre=EVe;
        else
            EVpre=min(EVpre,EVe);
        end
    end % EVpre is now the worst case over the e-priors; the z expectation is next

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], n_z, n_e, d13a1_gridvals, a12_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        % (d,aprime,a,z,e)

        ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
        for amb_c0=1:n_ambiguity(N_j)
            EV=EVpre.*shiftdim(ambiguity_pi_z_J(:,:,N_j,amb_c0)',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            ambEVstack=cat(4,ambEVstack,EV);
        end

        % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
        % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
        for amb_c=1:n_ambiguity(N_j)
            EV=ambEVstack(:,:,:,amb_c);
            a2primeProbsK=a2primeProbs;

            % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
            % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbsK to zero)
            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1))); % Note, probably just do this off of a2prime values
            aprimeProbsK=repmat(a2primeProbsK,N_a1,N_z);  % [N_d*N_a1,N_u]
            aprimeProbsK(skipinterp)=0;
            aprimeProbsK=reshape(aprimeProbsK,[N_d23*N_a1,N_u,N_z]);

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)); % (d&a1prime,u,z), the lower aprime
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)); % (d&a1prime,u,z), the upper aprime

            % Apply the aprimeProbsK
            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_z]).*aprimeProbsK; % probability of lower grid point
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbsK); % probability of upper grid point

            % Expectation over u (using ambiguity_pi_u), and then add the lower and upper
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
        % EV is over (d&a1prime,1,z)

        % Time to refine
        % First: ReturnMatrix, we can refine out d1
        [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix,[N_d1,N_d3*N_a1,N_a,N_z,N_e]),[],1);
        % Second: EV, we can refine out d2
        [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_z,1]),[],1);
        % Now put together entireRHS, which just depends on d3
        entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(3,:,:,:,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy(4,:,:,:,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy(1,:,:,:,N_j)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*zind+N_d3*N_a1*N_a*N_z*eind),1);
        Policy(2,:,:,:,N_j)=shiftdim(d2index(maxindex+N_d3*zind),1);

    elseif vfoptions.lowmemory==1
        ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
        for amb_c0=1:n_ambiguity(N_j)
            EV=EVpre.*shiftdim(ambiguity_pi_z_J(:,:,N_j,amb_c0)',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            ambEVstack=cat(4,ambEVstack,EV);
        end

        % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
        % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
        for amb_c=1:n_ambiguity(N_j)
            EV=ambEVstack(:,:,:,amb_c);
            a2primeProbsK=a2primeProbs;

            % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
            % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbsK to zero)
            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1))); % Note, probably just do this off of a2prime values
            aprimeProbsK=repmat(a2primeProbsK,N_a1,N_z);  % [N_d*N_a1,N_u]
            aprimeProbsK(skipinterp)=0;
            aprimeProbsK=reshape(aprimeProbsK,[N_d23*N_a1,N_u,N_z]);

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)); % (d&a1prime,u,z), the lower aprime
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)); % (d&a1prime,u,z), the upper aprime

            % Apply the aprimeProbsK
            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_z]).*aprimeProbsK; % probability of lower grid point
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbsK); % probability of upper grid point

            % Expectation over u (using ambiguity_pi_u), and then add the lower and upper
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
        % EV is over (d&a1prime,1,z)

        betaEV=DiscountFactorParamsVec*EV;

        % Time to refine
        % Second (out of order): EV, we can refine out d2
        [EV_onlyd3,d2index]=max(reshape(betaEV,[N_d2,N_d3*N_a1,1,N_z]),[],1);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], n_z, special_n_e, d13a1_gridvals, a12_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_e,[N_d1,N_d3*N_a1,N_a,N_z]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS_e=shiftdim(ReturnMatrix_onlyd3+EV_onlyd3,1);

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(3,:,:,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
            Policy(4,:,:,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
            Policy(1,:,:,e_c,N_j)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*zind),1);
            Policy(2,:,:,e_c,N_j)=shiftdim(d2index(maxindex+N_d3*zind),1);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);

            %Calc the condl expectation term (except beta) which depends on z but not control variables
            ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
            for amb_c0=1:n_ambiguity(N_j)
                EV_z=EVpre.*ambiguity_pi_z_J(z_c,:,N_j,amb_c0);
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension
                ambEVstack=cat(3,ambEVstack,EV_z);
            end

            % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
            % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
            for amb_c=1:n_ambiguity(N_j)
                EV_z=ambEVstack(:,:,amb_c);
                a2primeProbsK=a2primeProbs;

                % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
                % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbsK to zero)
                skipinterp=logical(EV_z(aprimeIndex)==EV_z(aprimeplus1Index)); % Note, probably just do this off of a2prime values
                aprimeProbsK=repmat(a2primeProbsK,N_a1,1);  % [N_d*N_a1,N_u]
                aprimeProbsK(skipinterp)=0;

                % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
                EV1_z=aprimeProbsK.*reshape(EV_z(aprimeIndex),[N_d23*N_a1,N_u]); % (d&a1prime,u), the lower aprime
                EV2_z=(1-aprimeProbsK).*reshape(EV_z(aprimeplus1Index),[N_d23*N_a1,N_u]); % (d&a1prime,u), the upper aprime
                % Already applied the probabilities from interpolating onto grid

                % Expectation over u (using ambiguity_pi_u), and then add the lower and upper
                if amb_c==1
                    Mmin=EV1_z+EV2_z; EV1_zsel=EV1_z; EV2_zsel=EV2_z;
                else
                    Mk=EV1_z+EV2_z;
                    newmin=(Mk<Mmin);
                    Mmin(newmin)=Mk(newmin);
                    EV1_zsel(newmin)=EV1_z(newmin);
                    EV2_zsel(newmin)=EV2_z(newmin);
                end
            end
            % Worst case over the u-priors (the ambiguous risky return distribution)
            EV_z=sum(EV1_zsel.*ambiguity_pi_u(:,1)',2)+sum(EV2_zsel.*ambiguity_pi_u(:,1)',2);
            for amb_cu=2:n_ambiguity(N_j)
                EV_z=min(EV_z,sum(EV1_zsel.*ambiguity_pi_u(:,amb_cu)',2)+sum(EV2_zsel.*ambiguity_pi_u(:,amb_cu)',2));
            end
            % EV is over (d&a1prime,1)

            betaEV_z=DiscountFactorParamsVec*EV_z;

            % Time to refine
            % Second (out of order): EV, we can refine out d2
            [EV_onlyd3,d2index]=max(reshape(betaEV_z,[N_d2,N_d3*N_a1,1]),[],1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], special_n_z, special_n_e, d13a1_gridvals, a12_gridvals, z_val, e_val, ReturnFnParamsVec);

                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_ze,[N_d1,N_d3*N_a1,N_a]),[],1);
                % Now put together entireRHS, which just depends on d3
                entireRHS_ze=shiftdim(ReturnMatrix_onlyd3+EV_onlyd3,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);

                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(3,:,z_c,e_c,N_j)=shiftdim(rem(maxindex-1,N_d3)+1,1);
                Policy(4,:,z_c,e_c,N_j)=shiftdim(ceil(maxindex/N_d3),-1);
                Policy(1,:,z_c,e_c,N_j)=shiftdim(d1index(maxindex+N_d3*N_a1*aind),1);
                Policy(2,:,z_c,e_c,N_j)=shiftdim(d2index(maxindex),1);
            end
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

    EVpree=V(:,:,:,jj+1);

    for amb_ce=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors (running worst case)
        EVe=sum(EVpree.*ambiguity_pi_e_J(1,1,:,jj+1,amb_ce),3);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        if amb_ce==1
            EVpre=EVe;
        else
            EVpre=min(EVpre,EVe);
        end
    end % EVpre is now the worst case over the e-priors; the z expectation is next

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a2, n_u, d23_grid, a2_grid, u_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: a2primeIndex is [N_d,N_u], whereas a2primeProbs is [N_d,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex-1,N_a1,1); % [N_d*N_a1,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d23,N_u)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d*N_a1,N_u]
    % aprimeProbs=repmat(a2primeProbs,N_a1,1);  % [N_d*N_a1,N_u]
    % Note: aprimeIndex corresponds to value of (a1, a2), but has dimension (d,a1)

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], n_z, n_e, d13a1_gridvals, a12_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);
        % (d,aprime,a,z,e)

        ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
        for amb_c0=1:n_ambiguity(jj)
            EV=EVpre.*shiftdim(ambiguity_pi_z_J(:,:,jj,amb_c0)',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            ambEVstack=cat(4,ambEVstack,EV);
        end

        % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
        % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
        for amb_c=1:n_ambiguity(jj)
            EV=ambEVstack(:,:,:,amb_c);
            a2primeProbsK=a2primeProbs;

            % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
            % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbsK to zero)
            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1))); % Note, probably just do this off of a2prime values
            aprimeProbsK=repmat(a2primeProbsK,N_a1,N_z);  % [N_d*N_a1,N_u]
            aprimeProbsK(skipinterp)=0;
            aprimeProbsK=reshape(aprimeProbsK,[N_d23*N_a1,N_u,N_z]);

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)); % (d&a1prime,u,z), the lower aprime
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)); % (d&a1prime,u,z), the upper aprime

            % Apply the aprimeProbsK
            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_z]).*aprimeProbsK; % probability of lower grid point
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbsK); % probability of upper grid point

            % Expectation over u (using ambiguity_pi_u), and then add the lower and upper
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
        % EV is over (d,1,z)

        % ReturnMatrix is over (d&a1prime,a,z,e)

        % Time to refine
        % First: ReturnMatrix, we can refine out d1
        [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix,[N_d1,N_d3*N_a1,N_a,N_z,N_e]),[],1);
        % Second: EV, we can refine out d2
        [EV_onlyd3,d2index]=max(reshape(EV,[N_d2,N_d3*N_a1,1,N_z,1]),[],1);
        % Now put together entireRHS, which just depends on d3
        entireRHS=shiftdim(ReturnMatrix_onlyd3+DiscountFactorParamsVec*EV_onlyd3,1);
        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(3,:,:,:,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
        Policy(4,:,:,:,jj)=shiftdim(ceil(maxindex/N_d3),-1);
        Policy(1,:,:,:,jj)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*zind+N_d3*N_a1*N_a*N_z*eind),1);
        Policy(2,:,:,:,jj)=shiftdim(d2index(maxindex+N_d3*zind),1);

    elseif vfoptions.lowmemory==1

        ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
        for amb_c0=1:n_ambiguity(jj)
            EV=EVpre.*shiftdim(ambiguity_pi_z_J(:,:,jj,amb_c0)',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension
            ambEVstack=cat(4,ambEVstack,EV);
        end

        % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
        % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
        for amb_c=1:n_ambiguity(jj)
            EV=ambEVstack(:,:,:,amb_c);
            a2primeProbsK=a2primeProbs;

            % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
            % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbsK to zero)
            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_z)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1))); % Note, probably just do this off of a2prime values
            aprimeProbsK=repmat(a2primeProbsK,N_a1,N_z);  % [N_d*N_a1,N_u]
            aprimeProbsK(skipinterp)=0;
            aprimeProbsK=reshape(aprimeProbsK,[N_d23*N_a1,N_u,N_z]);

            % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
            EV1=EV(aprimeIndex(:)+N_a*((1:1:N_z)-1)); % (d,u,z), the lower aprime
            EV2=EV(aprimeplus1Index(:)+N_a*((1:1:N_z)-1)); % (d,u,z), the upper aprime

            % Apply the aprimeProbsK
            EV1=reshape(EV1,[N_d23*N_a1,N_u,N_z]).*aprimeProbsK; % probability of lower grid point
            EV2=reshape(EV2,[N_d23*N_a1,N_u,N_z]).*(1-aprimeProbsK); % probability of upper grid point

            % Expectation over u (using ambiguity_pi_u), and then add the lower and upper
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
        % EV is over (d&a1prime,1,z)

        betaEV=DiscountFactorParamsVec*EV;

        % Time to refine
        % Second (out of order): EV, we can refine out d2
        [EV_onlyd3,d2index]=max(reshape(betaEV,[N_d2,N_d3*N_a1,1,N_z]),[],1);

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], n_z, special_n_e, d13a1_gridvals, a12_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);
            % (d,aprime,a,z)

            % Time to refine
            % First: ReturnMatrix, we can refine out d1
            [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_e,[N_d1,N_d3*N_a1,N_a,N_z]),[],1);
            % Now put together entireRHS, which just depends on d3
            entireRHS_e=shiftdim(ReturnMatrix_onlyd3+EV_onlyd3,1);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(3,:,:,e_c,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
            Policy(4,:,:,e_c,jj)=shiftdim(ceil(maxindex/N_d3),-1);
            Policy(1,:,:,e_c,jj)=shiftdim(d1index(maxindex+N_d3*N_a1*aind+N_d3*N_a1*N_a*zind),1);
            Policy(2,:,:,e_c,jj)=shiftdim(d2index(maxindex+N_d3*zind),1);
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);

            %Calc the condl expectation term (except beta) which depends on z but not control variables
            ambEVstack=[]; % one slice per z-prior (the aprime lottery below is conditional on the prior)
            for amb_c0=1:n_ambiguity(jj)
                EV_z=EVpre.*ambiguity_pi_z_J(z_c,:,jj,amb_c0);
                EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension
                ambEVstack=cat(3,ambEVstack,EV_z);
            end

            % Worst case over the stacked priors, with the aprime lottery conditional on the prior (running argmin,
            % tracking the winning prior's components so the u-stage arithmetic matches the exponential donor)
            for amb_c=1:n_ambiguity(jj)
                EV_z=ambEVstack(:,:,amb_c);
                a2primeProbsK=a2primeProbs;

                % Seems like interpolation has trouble due to numerical precision rounding errors when the two points being interpolated are equal
                % So I will add a check for when this happens, and then overwrite those (by setting aprimeProbsK to zero)
                skipinterp=logical(EV_z(aprimeIndex)==EV_z(aprimeplus1Index)); % Note, probably just do this off of a2prime values
                aprimeProbsK=repmat(a2primeProbsK,N_a1,1);  % [N_d*N_a1,N_u]
                aprimeProbsK(skipinterp)=0;

                % Switch EV from being in terms of aprime to being in terms of d (in expectation because of the u shocks)
                EV1_z=aprimeProbsK.*reshape(EV_z(aprimeIndex),[N_d23*N_a1,N_u]); % (d&a1prime,u), the lower aprime
                EV2_z=(1-aprimeProbsK).*reshape(EV_z(aprimeplus1Index),[N_d23*N_a1,N_u]); % (d&a1prime,u), the upper aprime
                % Already applied the probabilities from interpolating onto grid

                % Expectation over u (using ambiguity_pi_u), and then add the lower and upper
                if amb_c==1
                    Mmin=EV1_z+EV2_z; EV1_zsel=EV1_z; EV2_zsel=EV2_z;
                else
                    Mk=EV1_z+EV2_z;
                    newmin=(Mk<Mmin);
                    Mmin(newmin)=Mk(newmin);
                    EV1_zsel(newmin)=EV1_z(newmin);
                    EV2_zsel(newmin)=EV2_z(newmin);
                end
            end
            % Worst case over the u-priors (the ambiguous risky return distribution)
            EV_z=sum(EV1_zsel.*ambiguity_pi_u(:,1)',2)+sum(EV2_zsel.*ambiguity_pi_u(:,1)',2);
            for amb_cu=2:n_ambiguity(jj)
                EV_z=min(EV_z,sum(EV1_zsel.*ambiguity_pi_u(:,amb_cu)',2)+sum(EV2_zsel.*ambiguity_pi_u(:,amb_cu)',2));
            end
            % EV_z is over (d&a1prime,1)

            betaEV_z=DiscountFactorParamsVec*EV_z;

            % Time to refine
            % Second (out of order): EV, we can refine out d2
            [EV_onlyd3,d2index]=max(reshape(betaEV_z,[N_d2,N_d3*N_a1,1]),[],1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d13,n_a1], [n_a1,n_a2], special_n_z, special_n_e, d13a1_gridvals, a12_gridvals, z_val, e_val, ReturnFnParamsVec);

                % Time to refine
                % First: ReturnMatrix, we can refine out d1
                [ReturnMatrix_onlyd3,d1index]=max(reshape(ReturnMatrix_ze,[N_d1,N_d3*N_a1,N_a]),[],1);
                % Now put together entireRHS, which just depends on d3
                entireRHS_ze=shiftdim(ReturnMatrix_onlyd3+EV_onlyd3,1);

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);

                V(:,z_c,e_c,jj)=Vtemp;
                Policy(3,:,z_c,e_c,jj)=shiftdim(rem(maxindex-1,N_d3)+1,1);
                Policy(4,:,z_c,e_c,jj)=shiftdim(ceil(maxindex/N_d3),-1);
                Policy(1,:,z_c,e_c,jj)=shiftdim(d1index(maxindex+N_d3*N_a1*aind),1);
                Policy(2,:,z_c,e_c,jj)=shiftdim(d2index(maxindex),1);
            end
        end
    end

end



end
