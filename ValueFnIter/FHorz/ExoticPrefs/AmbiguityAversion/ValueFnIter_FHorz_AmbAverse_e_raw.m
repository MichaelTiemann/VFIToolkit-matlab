function [V,Policy]=ValueFnIter_FHorz_AmbAverse_e_raw(n_ambiguity, n_d,n_a,n_z,n_e,N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J,ambiguity_pi_z_J, ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Ambiguity aversion: multiple priors over the shock process; the continuation EV is the worst case over the priors.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);


V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(1,N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

ambiguity_pi_e_J=shiftdim(ambiguity_pi_e_J,-2); % Move to third dimension

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(1,:,:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(1,:,:,e_c,N_j)=maxindex;
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                % Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(1,:,z_c,e_c,N_j)=maxindex;
            end
        end

    end
else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpree=reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]);    % First, switch V_Jplus1 into Kron form
    ambEVe=zeros(N_a,N_z,n_ambiguity(N_j),'gpuArray'); % (aprime,z,prior)
    for amb_c=1:n_ambiguity(N_j) % evaluate the iid-e expectation under each of the multiple priors
        EVe=EVpree.*ambiguity_pi_e_J(1,1,:,N_j+1,amb_c);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EVe=sum(EVe,3); % sum over e', leaving a singular third dimension
        ambEVe(:,:,amb_c)=EVe;
    end
    EVpre=min(ambEVe,[],3); % take the worst-case over the priors (iid e); the z expectation is next

    ambEV=zeros(N_a,1,N_z,n_ambiguity(N_j),'gpuArray'); % (aprime,1,z,prior)
    for amb_c=1:n_ambiguity(N_j) % evaluate the expectation under each of the multiple priors
        EV=EVpre.*shiftdim(ambiguity_pi_z_J(:,:,N_j,amb_c)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        ambEV(:,:,:,amb_c)=EV;
    end
    EV=min(ambEV,[],4); % take the worst-case over the priors
    % From here, can just use EV as normal

    entireEV=repelem(EV,N_d,1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);
        % (d,aprime,a,z,e)

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV; % autofill a&e into EV

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(1,:,:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0);
            % (d,aprime,a,z)

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV; % autofill a into EV

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(1,:,:,e_c,N_j)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            entireEV_z=entireEV(:,:,z_c);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                % (d,aprime,a)

                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z; % autofill a into EV

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(1,:,z_c,e_c,N_j)=maxindex;
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
    ambEVe=zeros(N_a,N_z,n_ambiguity(jj),'gpuArray'); % (aprime,z,prior)
    for amb_c=1:n_ambiguity(jj) % evaluate the iid-e expectation under each of the multiple priors
        EVe=EVpree.*ambiguity_pi_e_J(1,1,:,jj+1,amb_c);
        EVe(isnan(EVe))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EVe=sum(EVe,3); % sum over e', leaving a singular third dimension
        ambEVe(:,:,amb_c)=EVe;
    end
    EVpre=min(ambEVe,[],3); % take the worst-case over the priors (iid e); the z expectation is next

    ambEV=zeros(N_a,1,N_z,n_ambiguity(jj),'gpuArray'); % (aprime,1,z,prior)
    for amb_c=1:n_ambiguity(jj) % evaluate the expectation under each of the multiple priors
        EV=EVpre.*shiftdim(ambiguity_pi_z_J(:,:,jj,amb_c)',-1);
        EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        ambEV(:,:,:,amb_c)=EV;
    end
    EV=min(ambEV,[],4); % take the worst-case over the priors
    % From here, can just use EV as normal

    entireEV=repelem(EV,N_d,1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, d_gridvals, a_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);
        % (d,aprime,a,z,e)

        entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV; %  autofill a&e into EV

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(1,:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, n_z, special_n_e, d_gridvals, a_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0);
            % (d,aprime,a,z)

            entireRHS_e=ReturnMatrix_e+DiscountFactorParamsVec*entireEV; % autofill a into EV

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(1,:,:,e_c,jj)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            entireEV_z=entireEV(:,:,z_c);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_ze=CreateReturnFnMatrix_Disc_e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_gridvals, a_grid, z_val, e_val, ReturnFnParamsVec,0);
                % (d,aprime,a)

                entireRHS_ze=ReturnMatrix_ze+DiscountFactorParamsVec*entireEV_z; % autofill a into EV

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                V(:,z_c,e_c,jj)=Vtemp;
                Policy(1,:,z_c,e_c,jj)=maxindex;
            end
        end
    end
end


end
