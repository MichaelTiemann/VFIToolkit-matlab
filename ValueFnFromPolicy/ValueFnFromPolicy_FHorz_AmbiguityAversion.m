function V=ValueFnFromPolicy_FHorz_AmbiguityAversion(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,ReturnFn,Parameters,DiscountFactorParamNames,vfoptions)
% Ambiguity aversion variant of ValueFnFromPolicy_FHorz: values the given Policy under the
% min-over-priors continuation (maxmin). The priors arrive pre-processed in
% vfoptions.ambiguity_pi_z_J/ambiguity_pi_e_J and vfoptions.n_ambiguity is [1,N_j]
% (ValueFnFromPolicy_FHorz already ran ExogShockSetup_FHorz, whose AmbiguityAversion branch did
% this), so z_gridvals_J and the vfoptions fields are taken as given here; the regular pi_z/pi_e
% are not used (the priors replace them).
%
% The gridinterplayer==1 case mirrors ValueFnFromPolicy_FHorz_GI, with the interpolation done
% conditional on the prior (same convention as the AmbAverse GI raws): each prior's EV is read at
% the two adjacent aprime grid points with the L2 weights, and the worst case (min over priors)
% is taken afterwards. Handles both GI1 (scalar n_a) and GI2A (two standard endogenous states;
% a2prime is folded into the linear index so the lookups are shared).

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

n_ambiguity=vfoptions.n_ambiguity; % [1,N_j]

%% Risky asset routes to its own subfn (u is treated as ambiguity: ambiguity_pi_u mandatory;
% a riskyasset model with no z and no e is a sensible ambiguity model, so the no-shocks error
% below does not apply to it)
if vfoptions.riskyasset==1
    V=ValueFnFromPolicy_FHorz_AmbAverse_RiskyAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

if N_z==0 && N_e==0
    error('Cannot use Ambiguity Aversion without any shocks (what is the point?); you have n_z=0 and no e variables')
end
if prod(vfoptions.n_semiz)>0
    error('AmbiguityAversion is not implemented for semi-exogenous states (vfoptions.n_semiz)')
end
if vfoptions.gridinterplayer==1 && length(n_a)>2
    error('AmbiguityAversion with gridinterplayer is not implemented for more than two standard endogenous states')
end

%% Implement new way of handling ReturnFn inputs
ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

%%
a_gridvals=CreateGridvals(n_a,a_grid,1);

%%
if vfoptions.gridinterplayer==1
    % Slot index for the aprime lower index in the gridinterplayer==1 Kron'd Policy
    index_a1=1+(N_d>0); % 1 if no d, 2 if d
    l_a=length(n_a);
    n_a1=n_a(1);
    % GI2A (l_a>=2): interpolation is applied to the first endogenous state only, so
    % PolicyIndexesKron carries a separate a2prime row (index_a1+1). We fold the a2prime offset
    % straight into alower, turning it into a linear index into (N_a1*N_a2); alower+1 then still
    % steps a1 by one, so every (per-prior) lookup below is identical to the l_a==1 case.

    if N_z==0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_e,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_e,N_j]);
        if l_a>=2 % GI2A: fold a2prime into the linear index
            a2prime=reshape(PolicyIndexesKron(index_a1+1,:,:,:),[N_a,N_e,N_j]);
            alower=alower+n_a1*(a2prime-1);
        end
        PolicyProbs=zeros(N_a,N_e,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
        PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));

            if jj==N_j
                V(:,:,jj)=FofPolicy_jj;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                ambEVnextAtPolicy=zeros(N_a,N_e,n_ambiguity(jj),'gpuArray'); % (a,e,prior)
                for amb_c=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors
                    EVnext_amb=sum(V(:,:,jj+1).*shiftdim(vfoptions.ambiguity_pi_e_J(:,jj+1,amb_c),-1),2); % (N_a,1)
                    EVnext_amb(isnan(EVnext_amb))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                    % Look up this prior's EV at lower & upper aprime with the L2 weights (the interpolation is conditional on the prior)
                    EVlower=reshape(EVnext_amb(alower(:,:,jj)),[N_a,N_e]);
                    EVupper=reshape(EVnext_amb(alower(:,:,jj)+1),[N_a,N_e]);
                    ambEVnextAtPolicy(:,:,amb_c)=PolicyProbs(:,:,jj,1).*EVlower+PolicyProbs(:,:,jj,2).*EVupper;
                end
                EVnextAtPolicy=min(ambEVnextAtPolicy,[],3); % take the worst-case over the priors
                EVnextAtPolicy(isnan(EVnextAtPolicy))=0; % zero corner weights times -Inf next-states give NaN
                V(:,:,jj)=FofPolicy_jj+beta*EVnextAtPolicy;
            end
        end

        V=reshape(V,[n_a,vfoptions.n_e,N_j]);

    elseif N_z>0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_z,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_z,N_j]);
        if l_a>=2 % GI2A: fold a2prime into the linear index
            a2prime=reshape(PolicyIndexesKron(index_a1+1,:,:,:),[N_a,N_z,N_j]);
            alower=alower+n_a1*(a2prime-1);
        end
        PolicyProbs=zeros(N_a,N_z,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
        PolicyProbs(:,:,:,1)=1-PolicyProbs(:,:,:,2);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_z,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));

            if jj==N_j
                V(:,:,jj)=FofPolicy_jj;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                % EVnext_amb(aprime, z) = sum_{zprime} pi_z(z,zprime) * V(aprime, zprime, jj+1), per prior
                % Linear index into (N_a, N_z) at (alower(a,z), z) and (alower+1, z)
                zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)
                lower_lin=alower(:,:,jj)+zidxoffset;
                ambEVnextAtPolicy=zeros(N_a,N_z,n_ambiguity(jj),'gpuArray'); % (a,z,prior)
                for amb_c=1:n_ambiguity(jj) % evaluate the expectation under each of the multiple priors
                    EVnext_amb=V(:,:,jj+1)*vfoptions.ambiguity_pi_z_J(:,:,jj,amb_c)'; % (N_a, N_z)
                    EVnext_amb(isnan(EVnext_amb))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                    % Look up this prior's EV at (alower(a,z), z) and (alower+1, z) with the L2 weights (the interpolation is conditional on the prior)
                    ambEVnextAtPolicy(:,:,amb_c)=PolicyProbs(:,:,jj,1).*EVnext_amb(lower_lin)+PolicyProbs(:,:,jj,2).*EVnext_amb(lower_lin+1);
                end
                EVnextAtPolicy=min(ambEVnextAtPolicy,[],3); % take the worst-case over the priors
                EVnextAtPolicy(isnan(EVnextAtPolicy))=0; % zero corner weights times -Inf next-states give NaN
                V(:,:,jj)=FofPolicy_jj+beta*EVnextAtPolicy;
            end
        end

        V=reshape(V,[n_a,n_z,N_j]);

    elseif N_z>0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_daprime,N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions); % rows: alower (=index_a1), L2 (=end). L2flag dropped by Kron.

        alower=reshape(PolicyIndexesKron(index_a1,:,:,:),[N_a,N_z,N_e,N_j]);
        L2=reshape(PolicyIndexesKron(end,:,:,:),[N_a,N_z,N_e,N_j]);
        if l_a>=2 % GI2A: fold a2prime into the linear index
            a2prime=reshape(PolicyIndexesKron(index_a1+1,:,:,:),[N_a,N_z,N_e,N_j]);
            alower=alower+n_a1*(a2prime-1);
        end
        PolicyProbs=zeros(N_a,N_z,N_e,N_j,2,'gpuArray');
        PolicyProbs(:,:,:,:,2)=(L2-1)/(vfoptions.ngridinterp+1);
        PolicyProbs(:,:,:,:,1)=1-PolicyProbs(:,:,:,:,2);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_z,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j;

            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

            if jj==N_j
                V(:,:,:,jj)=FofPolicy_jj;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                % Integrate over iid e (worst case over the priors), then over zprime|z (worst case over the priors)
                ambEVe=zeros(N_a,N_z,n_ambiguity(jj),'gpuArray'); % (aprime,z,prior)
                for amb_c=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors
                    EVe_amb=sum(V(:,:,:,jj+1).*shiftdim(vfoptions.ambiguity_pi_e_J(:,jj+1,amb_c),-2),3); % (N_a, N_z)
                    EVe_amb(isnan(EVe_amb))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                    ambEVe(:,:,amb_c)=EVe_amb;
                end
                EVnextpre=min(ambEVe,[],3); % take the worst-case over the priors (iid e; pointwise on the grid, so no interpolation is involved yet)
                % For each (a, z, e), look up each z-prior's EV at (alower(a,z,e), z) and (alower+1, z)
                zidxoffset=N_a*gpuArray(0:N_z-1); % (1, N_z)
                lower_lin=alower(:,:,:,jj)+zidxoffset; % (N_a, N_z, N_e) — broadcasting
                ambEVnextAtPolicy=zeros(N_a,N_z,N_e,n_ambiguity(jj),'gpuArray'); % (a,z,e,prior)
                for amb_c=1:n_ambiguity(jj) % evaluate the expectation under each of the multiple priors
                    EVnext_amb=EVnextpre*vfoptions.ambiguity_pi_z_J(:,:,jj,amb_c)'; % (N_a, N_z)
                    EVnext_amb(isnan(EVnext_amb))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                    % Look up this prior's EV with the L2 weights (the interpolation is conditional on the prior)
                    ambEVnextAtPolicy(:,:,:,amb_c)=PolicyProbs(:,:,:,jj,1).*EVnext_amb(lower_lin)+PolicyProbs(:,:,:,jj,2).*EVnext_amb(lower_lin+1);
                end
                EVnextAtPolicy=min(ambEVnextAtPolicy,[],4); % take the worst-case over the priors
                EVnextAtPolicy(isnan(EVnextAtPolicy))=0; % zero corner weights times -Inf next-states give NaN
                V(:,:,:,jj)=FofPolicy_jj+beta*EVnextAtPolicy;
            end
        end

        V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
    end

else % no grid interpolation layer

    if N_z==0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_e,l_d+l_a,N_j]
        % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, vfoptions.n_e, N_j, vfoptions);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j; % current period, counts backwards from J-1

            % Evaluate Return Fn
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,vfoptions.n_e,a_gridvals,vfoptions.e_gridvals_J(:,:,jj));

            if jj==N_j
                V(:,:,jj)=FofPolicy_jj;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                ambEVnext=zeros(N_a,n_ambiguity(jj),'gpuArray'); % (aprime,prior)
                for amb_c=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors
                    EVnext_amb=sum(V(:,:,jj+1).*shiftdim(vfoptions.ambiguity_pi_e_J(:,jj+1,amb_c),-1),2); % (N_a,1)
                    EVnext_amb(isnan(EVnext_amb))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                    ambEVnext(:,amb_c)=EVnext_amb;
                end
                EVnext=min(ambEVnext,[],2); % take the worst-case over the priors

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                % e is iid -> EVnext shape [N_a,1] depends only on aprime
                EVnextOfPolicy=EVnext(reshape(optaprime,[N_a*N_e,1]));

                V(:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_e]);
            end
        end

        %Transforming Value Fn out of Kronecker Form
        V=reshape(V,[n_a,vfoptions.n_e,N_j]);

    elseif N_z>0 && N_e==0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); %[N_a,N_z,l_d+l_a,N_j]
        % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_z,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j; % current period, counts backwards from J-1

            % Evaluate Return Fn
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));

            if jj==N_j
                V(:,:,jj)=FofPolicy_jj;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                % EVnext(a, z_from) = sum_{z_to} pi(z_from, z_to) * V(a, z_to, jj+1), worst case over the priors
                ambEVnext=zeros(N_a,N_z,n_ambiguity(jj),'gpuArray'); % (aprime,z,prior)
                for amb_c=1:n_ambiguity(jj) % evaluate the expectation under each of the multiple priors
                    EVnext_amb=V(:,:,jj+1)*vfoptions.ambiguity_pi_z_J(:,:,jj,amb_c)'; % [N_a, N_z_from]
                    EVnext_amb(isnan(EVnext_amb))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                    ambEVnext(:,:,amb_c)=EVnext_amb;
                end
                EVnext=min(ambEVnext,[],3); % take the worst-case over the priors

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                aprimez_index=reshape(optaprime,[N_a*N_z,1])+N_a*(kron((1:1:N_z)',ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron

                EVnextOfPolicy=EVnext(aprimez_index);

                V(:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_z]);
            end
        end

        %Transforming Value Fn out of Kronecker Form
        V=reshape(V,[n_a,n_z,N_j]);

    elseif N_z>0 && N_e>0

        PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_e
        l_daprime=size(PolicyValues,1);
        PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a,N_z*N_e,l_d+l_a,N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
        % The following will also be needed to calculate the expectation of next period value fn, evaluated based on the policy.
        PolicyIndexesKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, [n_z,vfoptions.n_e], N_j, vfoptions);

        %% Calculate the Value Fn by backward iteration
        V=zeros(N_a,N_z,N_e,N_j,'gpuArray');

        for reverse_j=0:N_j-1
            jj=N_j-reverse_j; % current period, counts backwards from J-1

            % Evaluate Return Fn
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
            FofPolicy_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell,PolicyValuesPermute(:,:,:,jj),l_daprime,n_a,[n_z,vfoptions.n_e],a_gridvals,[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)]),[N_a,N_z,N_e]);

            if jj==N_j
                V(:,:,:,jj)=FofPolicy_jj;
            else
                beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
                % Integrate over iid e (worst case over the priors), then over zprime|z (worst case over the priors)
                ambEVe=zeros(N_a,N_z,n_ambiguity(jj),'gpuArray'); % (aprime,z,prior)
                for amb_c=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors
                    EVe_amb=sum(V(:,:,:,jj+1).*shiftdim(vfoptions.ambiguity_pi_e_J(:,jj+1,amb_c),-2),3); % (N_a, N_z)
                    EVe_amb(isnan(EVe_amb))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                    ambEVe(:,:,amb_c)=EVe_amb;
                end
                EVnextpre=min(ambEVe,[],3); % take the worst-case over the priors (iid e)
                ambEVnext=zeros(N_a,N_z,n_ambiguity(jj),'gpuArray'); % (aprime,z,prior)
                for amb_c=1:n_ambiguity(jj) % evaluate the expectation under each of the multiple priors
                    EVnext_amb=EVnextpre*vfoptions.ambiguity_pi_z_J(:,:,jj,amb_c)'; % (N_a, N_z)
                    EVnext_amb(isnan(EVnext_amb))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                    ambEVnext(:,:,amb_c)=EVnext_amb;
                end
                EVnext=min(ambEVnext,[],3); % take the worst-case over the priors

                if N_d==0
                    optaprime=PolicyIndexesKron(1,:,:,jj);
                else
                    optaprime=shiftdim(PolicyIndexesKron(2,:,:,jj),1);
                end

                aprimez_index=reshape(optaprime,[N_a*N_z*N_e,1])+N_a*(kron(kron(ones(N_e,1,'gpuArray'),(1:1:N_z)'),ones(N_a,1,'gpuArray'))-1); % N_a*(z_index-1), but just with lots of kron

                EVnextOfPolicy=EVnext(aprimez_index);

                V(:,:,:,jj)=FofPolicy_jj+beta*reshape(EVnextOfPolicy,[N_a,N_z,N_e]);
            end

        end

        % Transforming Value Fn out of Kronecker Form
        V=reshape(V,[n_a,n_z,vfoptions.n_e,N_j]);
    end

end

end
