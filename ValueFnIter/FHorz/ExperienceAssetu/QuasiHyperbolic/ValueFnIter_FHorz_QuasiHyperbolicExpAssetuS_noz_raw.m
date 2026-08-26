function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_noz_raw(n_d1, n_d2,n_a1,n_a2,n_u,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, u_gridvals, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2; % argmax row dim: the ReturnMatrix is built over [n_d1,n_d2]
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_u=prod(n_u);

Vhat=zeros(N_a,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    Vhat(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,N_j)=Vhat(:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a2, n_u, d2_gridvals, a2_grid, u_gridvals, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_u]

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,1]);

    Vlower=reshape(EVpre(aprimeIndex(:)),[N_d2*N_a1,N_a2,N_u]);
    Vupper=reshape(EVpre(aprimeplus1Index(:)),[N_d2*N_a1,N_a2,N_u]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u)
    % Already applied the probabilities from interpolating onto grid
    EV=sum((EV.*pi_u),3); % (d2,a1prime,a2)
    EV(isnan(EV))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0

    % hat: argmax of F + beta0*beta*EV; under: the beta-RHS gathered at that argmax
    entireRHS_hat=ReturnMatrix+beta0beta*repelem(EV,N_d1,N_a1);
    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
    entireRHS_under=ReturnMatrix+beta*repelem(EV,N_d1,N_a1);
    maxindexfull=maxindex+N_d12*N_a1*(0:1:N_a-1);
    Vhat(:,N_j)=shiftdim(Vtemp,1);
    Vunderbar(:,N_j)=shiftdim(entireRHS_under(maxindexfull),1);
    Policy(:,N_j)=shiftdim(maxindex,1);

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
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a2, n_u, d2_gridvals, a2_grid, u_gridvals, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2,N_u], whereas aprimeProbs is [N_d2,N_a2,N_u]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2,N_u]
    aprimeProbs=repmat(a2primeProbs,N_a1,1,1);  % [N_d2*N_a1,N_a2,N_u]

    Vlower=reshape(Vunderbar(aprimeIndex(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
    Vupper=reshape(Vunderbar(aprimeplus1Index(:),jj+1),[N_d2*N_a1,N_a2,N_u]);
    % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
    skipinterp=(Vlower==Vupper);
    aprimeProbs(skipinterp)=0; % effectively skips interpolation

    % Switch EV from being in terms of a2prime to being in terms of d2 and a2
    EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper; % (d2,a1prime,a2,u)
    % Already applied the probabilities from interpolating onto grid
    EV=sum((EV.*pi_u),3); % (d2,a1prime,a2)
    EV(isnan(EV))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

    ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_noz(ReturnFn, n_d1,n_d2, n_a1, n_a1,n_a2, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, ReturnFnParamsVec,0,0); % Level=0, Refine=0

    % hat: argmax of F + beta0*beta*EV; under: the beta-RHS gathered at that argmax
    entireRHS_hat=ReturnMatrix+beta0beta*repelem(EV,N_d1,N_a1);
    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
    entireRHS_under=ReturnMatrix+beta*repelem(EV,N_d1,N_a1);
    maxindexfull=maxindex+N_d12*N_a1*(0:1:N_a-1);
    Vhat(:,jj)=shiftdim(Vtemp,1);
    Vunderbar(:,jj)=shiftdim(entireRHS_under(maxindexfull),1);
    Policy(:,jj)=shiftdim(maxindex,1);

end


%%
Policy=shiftdim(Policy,-1);


end
