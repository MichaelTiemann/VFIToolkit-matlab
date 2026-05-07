function [PolicyPath_ForAgentDistIter,PolicyProbsPath,PolicyValuesPath]=TransitionPath_InfHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath,T,Parameters,n_d,n_a,n_z,l_d,l_aprime,N_a,N_z,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions)


%%
if simoptions.experienceasset==1
    
    whichisdforexpasset=length(n_d)-simoptions.setup_experienceasset.l_dexperienceasset+1:length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
    if N_z==0
        a2primeIndexesPath=zeros(N_a,T-1,'gpuArray');
        a2primeProbsPath=zeros(N_a,T-1,'gpuArray');
        for tt=1:T-1
            aprimeFnParamsVec=CreateVectorFromParams(Parameters, simoptions.setup_experienceasset.aprimeFnParamNames);

            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_Case1(PolicyIndexesPath(:,:,tt),simoptions.setup_experienceasset.aprimeFn, whichisdforexpasset, n_d, simoptions.setup_experienceasset.n_a1,simoptions.setup_experienceasset.n_a2, 0, simoptions.setup_experienceasset.d_grid, simoptions.setup_experienceasset.a2_grid, aprimeFnParamsVec);
            % Note: a2primeIndexes and a2primeProbs are both [N_a]
            % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            a2primeIndexesPath(:,tt)=a2primeIndexes;
            a2primeProbsPath(:,tt)=a2primeProbs;
        end
    else
        a2primeIndexesPath=zeros(N_a,N_z,T-1,'gpuArray');
        a2primeProbsPath=zeros(N_a,N_z,T-1,'gpuArray');
        for tt=1:T-1
            aprimeFnParamsVec=CreateVectorFromParams(Parameters, simoptions.setup_experienceasset.aprimeFnParamNames);

            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_Case1(PolicyIndexesPath(:,:,:,tt),simoptions.setup_experienceasset.aprimeFn, whichisdforexpasset, n_d, simoptions.setup_experienceasset.n_a1,simoptions.setup_experienceasset.n_a2, N_z, simoptions.setup_experienceasset.d_grid, simoptions.setup_experienceasset.a2_grid, aprimeFnParamsVec);
            % Note: a2primeIndexes and a2primeProbs are both [N_a,N_z] for InfHorz
            % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            a2primeIndexesPath(:,:,tt)=a2primeIndexes;
            a2primeProbsPath(:,:,tt)=a2primeProbs;
        end
    end
    
    if N_z==0
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,1,T-1]);
        a2primeIndexesPath=repmat(a2primeIndexesPath,1,2,1);
        a2primeIndexesPath(:,2,:)=a2primeIndexesPath(:,2,:)+1; % upper index
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a,1,T-1]);
        a2primeProbsPath=repmat(a2primeProbsPath,1,2,1);
        a2primeProbsPath(:,2,:)=1-a2primeProbsPath(:,2,:); % upper prob
    else
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,N_z,1,T-1]);
        a2primeIndexesPath=repmat(a2primeIndexesPath,1,1,2,1);
        a2primeIndexesPath(:,:,2,:)=a2primeIndexesPath(:,:,2,:)+1; % upper index
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a,N_z,1,T-1]);
        a2primeProbsPath=repmat(a2primeProbsPath,1,1,2,1);
        a2primeProbsPath(:,:,2,:)=1-a2primeProbsPath(:,:,2,:); % upper prob
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a*N_z,2,T-1]);
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a*N_z,2,T-1]);
    end

    a2primeIndexesPath=gather(a2primeIndexesPath);
    a2primeProbsPath=gather(a2primeProbsPath);
end


if simoptions.experienceasset==1
    n_a1=simoptions.setup_experienceasset.n_a1;
else
    n_a1=n_a;
end

if N_z==0
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,0,T-1,d_gridvals,aprime_gridvals,vfoptions,1); % [size(PolicyValuesPath,1),N_a,T-1]
    % PolicyValuesPath=permute(PolicyValuesPath,[2,1,3]); %[N_a,l_d+l_aprime,T-1] % special ordering is needed for AggVars
    % Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime
    % When using grid interpolation layer also PolicyProbsPath
    if isscalar(n_a1)
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:),[N_a,T-1]); % aprime index
    elseif length(n_a1)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:)-1),[N_a,T-1]);
    elseif length(n_a1)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:)-1),[N_a,T-1]);
    elseif length(n_a1)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:)-1),[N_a,T-1]);
    end
    PolicyaprimePath=gather(PolicyaprimePath);
    clear PolicyIndexesPath
    if simoptions.gridinterplayer==1
        error("need to restore gridinterp functionality");
    end
    if simoptions.experienceasset==1
        if simoptions.setup_experienceasset.N_a1==0
            PolicyaprimePath=reshape(PolicyaprimePath,[N_a,1,T])+a2primeIndexesPath;
        else
            PolicyaprimePath=reshape(PolicyaprimePath,[N_a,1,T])+simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1);
        end
        PolicyProbsPath=a2primeProbsPath;
    end
else
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,n_z,T-1,d_gridvals,aprime_gridvals,vfoptions,1); % [size(PolicyValuesPath,1),N_a,N_z,T-1]
    PolicyValuesPath=permute(PolicyValuesPath,[2,3,1,4]); %[N_a,N_z,l_d+l_aprime,T-1] % special ordering is needed for AggVars
    % Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime
    % When using grid interpolation layer also PolicyProbsPath
    if isscalar(n_a1)
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:),[N_a*N_z,T-1]); % aprime index
    elseif length(n_a1)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1),[N_a*N_z,T-1]);
    elseif length(n_a1)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:)-1),[N_a*N_z,T-1]);
    elseif length(n_a1)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:,:)-1),[N_a*N_z,T-1]);
    end
    PolicyaprimezPath=gather(PolicyaprimePath+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1));
    clear PolicyIndexesPath PolicyaprimePath
    if simoptions.gridinterplayer==1
        error("need to restore gridinterp functionality");
    end
    if simoptions.experienceasset==1
        if simoptions.setup_experienceasset.N_a1==0
            PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z,1,T-1])+a2primeIndexesPath;
        else
            PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z,1,T-1])+simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1);
        end
        PolicyProbsPath=a2primeProbsPath;
    end
end

% clear a2primeIndexesPath a2primeProbsPath % Free up some memory


%% Clean up output
if N_z==0
    PolicyPath_ForAgentDistIter=PolicyaprimePath;
elseif N_z>0
    PolicyPath_ForAgentDistIter=PolicyaprimezPath;
end


if isempty(N_probs) || N_probs==1 % =1 means not being used
    PolicyProbsPath=[];
end


end
