function [PricePathOld,GEcondnPath]=TransitionPath_InfHorz_shooting(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d,n_a,n_z,n_e, N_d,N_a,N_z,N_e, l_d,l_aprime,l_a,l_z,l_e, d_gridvals,aprime_gridvals,a_gridvals,a_grid,z_gridvals,e_gridvals,ze_gridvals,pi_z,pi_z_sparse,pi_e, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, vfoptions, simoptions,transpathoptions)
% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

if transpathoptions.verbose==1
    % Set up some things to be used later
    pathnametitles=strjoin(PricePathNames,' ');
    wpathnametitle=10*length(PricePathNames); % roughly the space that will use to print the prices themselves
    % fprintf('%-*s || %-*s \n',wpathnametitle,'Old',wpathnametitle,'New')
    % fprintf('%-*s || %-*s \n',wpathnametitle,pathnametitles,wpathnametitle,pathnametitles)
end

%%
PricePathNew=zeros(size(PricePathOld),'gpuArray');
PricePathNew(T,:)=PricePathOld(T,:);
AggVarsPath=zeros(T-1,length(AggVarNames),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect
GEcondnPath=zeros(T-1,length(GEeqnNames),'gpuArray');

% Setup, the shapes of various of these objects vary depending on the setting
[PolicyIndexesPath,N_probs,II1,II2]=TransitionPath_InfHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_z,N_e,T,transpathoptions,vfoptions,simoptions);

%%
PricePathDist=Inf;
GEcondnPathDist=Inf;
itercounter=1;
while (PricePathDist>transpathoptions.toleranceGEprices || GEcondnPathDist>transpathoptions.toleranceGEcondns) && itercounter<=transpathoptions.maxiter

    %% Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    [~,PolicyIndexesPath]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_z,N_e,d_gridvals, a_grid, z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,DiscountFactorParamNames, ReturnFnParamNames, transpathoptions,vfoptions);

    %% Modify PolicyIndexesPath into forms needed for forward iteration
    [PolicyPath_ForAgentDistIter,PolicyProbsPath,PolicyValuesPath]=TransitionPath_InfHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath,T,Parameters,n_d,n_a,n_z,n_e,l_d,l_aprime,N_a,N_z,N_e,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions);

   %% Iterate forward over t: iterate agent dist, calculate aggvars, evaluate general eqm
    % Call AgentDist the current periods distn and AgentDistnext the next periods distn which we must calculate
    AgentDist=AgentDist_initial;

    % Initialise _tminus1 entries in Parameters from initialvalues (used at tt=1)
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
        end
    end
    if use_tminus1params==1
        for pp=1:length(tminus1paramNames)
            Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
        end
    end
    if use_tminus1AggVars==1
        for pp=1:length(tminus1AggVarsNames)
            Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
        end
    end
    if use_stockvars==1
        for pp=1:length(stockvarsNames)
            Parameters.([stockvarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(stockvarsNames{pp});
        end
    end
    
    for tt=1:T-1
        %% Setup the Parameters for period tt

        % Get t-1 PricePath, ParamPath and AggVars before we update them
        if tt>1
            if use_tminus1price==1
                for pp=1:length(tminus1priceNames)
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                end
            end
            if use_tminus1params==1
                for pp=1:length(tminus1paramNames)
                    Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                end
            end
            if use_tminus1AggVars==1
                for pp=1:length(tminus1AggVarsNames)
                    % The AggVars have not yet been updated, so they still contain previous period values
                    Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
                end
            end
            if use_stockvars==1 % Comes from PricePathNew, unlike the _tminus1price, which comes from the PricePathOld
                for pp=1:length(stockvarsNames)
                    Parameters.([stockvarsNames{pp},'_tminus1'])=PricePathNew(tt-1,PricePathSizeVec(1,stockvarInPricePathNames(pp)):PricePathSizeVec(2,stockvarInPricePathNames(pp)));
                end
            end
        end

        % Update current PricePath and ParamPath
        for pp=1:length(PricePathNames)
            Parameters.(PricePathNames{pp})=PricePathOld(tt,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
        end
        for pp=1:length(ParamPathNames)
            Parameters.(ParamPathNames{pp})=ParamPath(tt,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
        end

        % Get t+1 PricePath
        if use_tplus1price==1
            for pp=1:length(tplus1priceNames)
                kk=tplus1pricePathkk(pp);
                Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
            end
        end

        %% Get the current optimal policy, and iterate the agent dist
        AgentDistnext=TransitionPath_InfHorz_substeps_Step3tt_IterAgentDist(AgentDist,PolicyPath_ForAgentDistIter,PolicyProbsPath,tt,N_a,N_z,N_e,N_probs,pi_z_sparse,pi_e,II1,II2,transpathoptions,simoptions);

        %% AggVars
        if N_z==0 && N_e==0
            AggVars=TransitionPath_InfHorz_substeps_Step4tt_AggVars(AgentDist,PolicyValuesPath(:,:,tt),tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,n_a,n_z,n_e,N_z,N_e,a_gridvals,ze_gridvals,transpathoptions);
        else
            AggVars=TransitionPath_InfHorz_substeps_Step4tt_AggVars(AgentDist,PolicyValuesPath(:,:,:,tt),tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,n_a,n_z,n_e,N_z,N_e,a_gridvals,ze_gridvals,transpathoptions);
        end

        for ff=1:length(AggVarNames) % Note: needed for _tminus1 as well as GeneralEqmEqns
            Parameters.(AggVarNames{ff})=AggVars.(AggVarNames{ff}).Mean;
        end

        %% Intermediate Eqns
        if transpathoptions.useintermediateEqns==1
            % Note: intermediateEqns just take in things from the Parameters structure, as do GeneralEqmEqns (AggVars get put into structure), hence just use the GeneralEqmConditions_Case1_v3g().
            intEqnnames=fieldnames(transpathoptions.intermediateEqns);
            intermediateEqnsVec=zeros(1,length(intEqnnames));
            % Do the intermediateEqns, in order
            for gg=1:length(intEqnnames)
                intermediateEqnsVec(gg)=real(GeneralEqmConditions_Case1_v3g(transpathoptions.intermediateEqnsCell{gg}, transpathoptions.intermediateEqnParamNames(gg).Names, Parameters));
                Parameters.(intEqnnames{gg})=intermediateEqnsVec(gg);
            end
        end
    
        %% General Eqm Eqns
        if transpathoptions.updatepert==1
            % Evaluate the general eqm conditions, and based on them create PricePathNew (interpretation depends on transpathoptions)
            [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_tt(Parameters,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld(tt,:),itercounter,transpathoptions);
            PricePathNew(tt,:)=PricePathNew_tt;
            GEcondnPath(tt,:)=GEcondnPath_tt;
        else
            % Just evaluate the general eqm conditions for this period. Creating PricePathNew from
            % them happens after the tt loop, in updatePricePathNew_TPath_T, because the Newton
            % options need the conditions from every period before they can produce an update.
            for gg=1:length(GeneralEqmEqnsCell)
                % Note: _v3 rather than _v3g, so on CPU rather than GPU
                GEcondnPath(tt,gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
                % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
            end
        end
        
        % Sometimes, want to keep the AggVars to plot them
        if transpathoptions.graphaggvarspath==1
            for ii=1:length(AggVarNames)
                AggVarsPath(tt,ii)=AggVars.(AggVarNames{ii}).Mean;
            end
        end

        AgentDist=AgentDistnext;
    end
    

    %% Now update prices, give verbose feedback, and check for convergence
    if transpathoptions.updatepert==0
        % Every general eqm condition, for every period, is now known, so the update can use them together
        PricePathNew=updatePricePathNew_TPath_T(GEcondnPath,PricePathOld,T,itercounter,transpathoptions);
    end

    % See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
    % And how far the general eqm conditions are from zero. Scalarize across the general eqm eqns in each
    % time period the same way the stationary general eqm does, then take the L-Infinity norm over time
    % (the same norm as is used for the prices). GEcondnPath is the raw conditions, before the permute
    % and before updateaccuracycutoff is applied.
    if transpathoptions.multiGEcriterion==0
        GEcondnPathDist=max(sum(abs(transpathoptions.multiGEweights.*GEcondnPath),2));
    elseif transpathoptions.multiGEcriterion==1
        GEcondnPathDist=max(sqrt(sum(transpathoptions.multiGEweights.*(GEcondnPath.^2),2)));
    end

    if transpathoptions.verbose==1
        fprintf(' \n')
        fprintf('%-*s || %-*s \n',wpathnametitle,'Old',wpathnametitle,'New')
        fprintf('%-*s || %-*s \n',wpathnametitle,pathnametitles,wpathnametitle,pathnametitles)

        % Would be nice to have a way to get the iteration count without having the whole printout of path values (I think that would be useful?)
        [PricePathOld,PricePathNew]
    end

    % Create plots of the transition path (before we update pricepath)
    createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,transpathoptions);

    PricePathOld=updatePricePath(PricePathOld,PricePathNew,transpathoptions,T);

    TransPathConvergence=max(PricePathDist/transpathoptions.toleranceGEprices,GEcondnPathDist/transpathoptions.toleranceGEcondns); % So when this gets to 1 we have convergence, we require convergence in both
    if transpathoptions.verbose==1
        fprintf('Number of iterations on transition path: %i \n',itercounter)
        if isfinite(transpathoptions.toleranceGEprices)
            fprintf('Current distance between old and new price path (in L-Infinity norm): %8.6f \n', PricePathDist)
        end
        fprintf('Current distance of the general eqm conditions from zero: %8.6f \n', GEcondnPathDist)
        fprintf('Ratio of current distance to the convergence tolerance: %.2f (convergence when reaches 1) \n',TransPathConvergence)
    end

    if transpathoptions.historyofpricepath==1
        % Store the whole history of the price path and save it every ten iterations
        PricePathHistory{itercounter,1}=PricePathDist;
        PricePathHistory{itercounter,2}=PricePathOld;
        if rem(itercounter,10)==1
            save ./SavedOutput/TransPath_Internal.mat PricePathHistory
        end
    end

    itercounter=itercounter+1;


end



end
