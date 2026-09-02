function [V,Policy2]=ValueFnIter_FHorz_GulPesendorfer_noz_e_raw(n_d, n_a, n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Note: have no z variable, do have e variables

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_e,N_j,'gpuArray'); % first dim indexes the optimal choice for d and aprime (joint index, d fastest)

%%
if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

%% j=N_j
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0); % Because no z, can treat e like z

        TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix;

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,:,N_j)=Vtemp-MostTempting;
        Policy(:,:,N_j)=maxindex;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0);

            TemptationMatrix_e=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, TemptationFnParamsVec,0);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);
            V(:,e_c,N_j)=Vtemp-MostTempting_e;
            Policy(:,e_c,N_j)=maxindex;
        end
    end
else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    V_Jplus1=sum(V_Jplus1.*pi_e_J(1,:,N_j+1),2); % expectation over e'

    entireEV=kron(V_Jplus1,ones(N_d,1)); % expand to the joint (d,aprime) rows

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0);

        TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,N_j), TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*entireEV;

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,N_j)=shiftdim(Vtemp-MostTempting,1);
        Policy(:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0);

            TemptationMatrix_e=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, TemptationFnParamsVec,0);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e+DiscountFactorParamsVec*entireEV;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,e_c,N_j)=shiftdim(Vtemp-MostTempting_e,1);
            Policy(:,e_c,N_j)=shiftdim(maxindex,1);
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=V(:,:,jj+1);

    EV=sum(EV.*pi_e_J(1,:,jj+1),2); % expectation over e'

    entireEV=kron(EV,ones(N_d,1)); % expand to the joint (d,aprime) rows

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,jj), ReturnFnParamsVec,0);

        TemptationMatrix=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, n_e, d_gridvals, a_grid, e_gridvals_J(:,:,jj), TemptationFnParamsVec,0);
        MostTempting=max(TemptationMatrix,[],1);
        entireRHS=ReturnMatrix+TemptationMatrix+DiscountFactorParamsVec*entireEV;

        % Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,jj)=shiftdim(Vtemp-MostTempting,1);
        Policy(:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, ReturnFnParamsVec,0);

            TemptationMatrix_e=CreateReturnFnMatrix_Disc(TemptationFn, n_d, n_a, special_n_e, d_gridvals, a_grid, e_val, TemptationFnParamsVec,0);
            MostTempting_e=max(TemptationMatrix_e,[],1);
            entireRHS_e=ReturnMatrix_e+TemptationMatrix_e+DiscountFactorParamsVec*entireEV;

            % Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_e,[],1);

            V(:,e_c,jj)=shiftdim(Vtemp-MostTempting_e,1);
            Policy(:,e_c,jj)=shiftdim(maxindex,1);
        end
    end
end

%%
% Return the joint (d,aprime) Kron index (d fastest); the dispatcher's UnKron does the split
Policy2=shiftdim(Policy,-1);

end
