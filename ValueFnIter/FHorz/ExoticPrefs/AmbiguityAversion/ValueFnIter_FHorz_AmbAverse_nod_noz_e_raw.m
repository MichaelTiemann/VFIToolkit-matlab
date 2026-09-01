function  [V,Policy]=ValueFnIter_FHorz_AmbAverse_nod_noz_e_raw(n_ambiguity, n_a, n_e, N_j, a_grid, e_gridvals_J, ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Ambiguity aversion: multiple priors over the shock process; the continuation EV is the worst case over the priors.
% Note: have no z variable, do have e variables

N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(1,N_a,N_e,N_j,'gpuArray'); % no d variable


%%
if vfoptions.lowmemory==0
    % Parallel over all e

    %% N_j
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

    ambiguity_pi_e_J=shiftdim(ambiguity_pi_e_J,-1); % Move to second dimensionfor e_c=1:n_e (normally -2, but no z so -1)

    if ~isfield(vfoptions,'V_Jplus1')
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0); % Because no z, can treat e like z and call Par2 rather than Par2e
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        Policy(1,:,:,N_j)=maxindex;
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_e]); % Using V_Jplus1
        ambEV=zeros(N_a,n_ambiguity(N_j),'gpuArray'); % (aprime,prior)
        for amb_c=1:n_ambiguity(N_j) % evaluate the iid-e expectation under each of the multiple priors
            EV=EVpre.*ambiguity_pi_e_J(1,:,N_j+1,amb_c);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over e', leaving a singular second dimension
            ambEV(:,amb_c)=EV;
        end
        EV=min(ambEV,[],2); % take the worst-case over the priors

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % autofill a&e in EV

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp,1);
        Policy(1,:,:,N_j)=shiftdim(maxindex,1);
    end


    %% Loop backward over age
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;

        if vfoptions.verbose==1
            fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
        end

        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EVpre=V(:,:,jj+1);
        ambEV=zeros(N_a,n_ambiguity(jj),'gpuArray'); % (aprime,prior)
        for amb_c=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors
            EV=EVpre.*ambiguity_pi_e_J(1,:,jj+1,amb_c);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over e', leaving a singular second dimension
            ambEV(:,amb_c)=EV;
        end
        EV=min(ambEV,[],2); % take the worst-case over the priors

        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, n_e, 0, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*EV; % autofill a&e in EV

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp,1);
        Policy(1,:,:,jj)=shiftdim(maxindex,1);
    end
elseif vfoptions.lowmemory==1
    %% Loop over all e
    special_n_e=ones(1,length(n_e));

    %% N_j
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

    ambiguity_pi_e_J=shiftdim(ambiguity_pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

    if ~isfield(vfoptions,'V_Jplus1')
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec,0); % Because no z, can treat e like z and call Par2 rather than Par2e
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,e_c,N_j)=Vtemp;
            Policy(1,:,e_c,N_j)=maxindex;
        end
    else
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form
        ambEV=zeros(N_a,n_ambiguity(N_j),'gpuArray'); % (aprime,prior)
        for amb_c=1:n_ambiguity(N_j) % evaluate the iid-e expectation under each of the multiple priors
            EV=EVpre.*ambiguity_pi_e_J(1,:,N_j+1,amb_c);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over e', leaving a singular second dimension
            ambEV(:,amb_c)=EV;
        end
        EV=min(ambEV,[],2); % take the worst-case over the priors

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec,0);

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*EV; % autofill a in EV

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(1,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    end

    %% Loop backward over age
    for reverse_j=1:N_j-1
        jj=N_j-reverse_j;

        if vfoptions.verbose==1
            fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
        end

        % Create a vector containing all the return function parameters (in order)
        ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
        DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

        EVpre=V(:,:,jj+1);
        ambEV=zeros(N_a,n_ambiguity(jj),'gpuArray'); % (aprime,prior)
        for amb_c=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors
            EV=EVpre.*ambiguity_pi_e_J(1,:,jj+1,amb_c);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over e', leaving a singular second dimension
            ambEV(:,amb_c)=EV;
        end
        EV=min(ambEV,[],2); % take the worst-case over the priors

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, 0, n_a, special_n_e, 0, a_grid, e_val, ReturnFnParamsVec,0);

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*EV; % autofill a in EV

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(1,:,e_c,jj)=shiftdim(maxindex,1);
        end
    end
end


end
