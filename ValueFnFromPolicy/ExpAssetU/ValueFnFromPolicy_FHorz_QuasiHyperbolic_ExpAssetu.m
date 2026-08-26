function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has experienceassetu (vfoptions.experienceassetu>=1)
% and quasi-hyperbolic discounting.
%
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext) is ALWAYS built from the recursion-driver value (Vdrive).
%
% Structural base: ValueFnFromPolicy_FHorz_ExpAssetu (the a2prime lottery and the u integration are
% carried over untouched). QH bookkeeping mirrors ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset.
%
% experienceassetu: a2prime=aprimeFn(d_expasset, a2, u), with u an iid between-period shock that does
% NOT enter Policy. a2primeIndex/a2primeProbs therefore carry a trailing N_u dimension ([N_a, N_u] with
% no shocks, [N_a, N_ze, N_u] otherwise) and u is integrated out with pi_u inside the pass loop, before
% the pi_z weighting, before the isnan clear, and before any discount factor is applied.
%
% The interpolated lookup is written once and run over a pass loop ({Policy} or {Policy,Policyalt}),
% rather than duplicated, so there is a single copy of the four-shock-case interpolation to check
% against the exponential source.

%% Dispatch to SemiExo subfn if n_semiz>0
if prod(vfoptions.n_semiz)>0
    % ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu_SemiExo{,_GI} are not written yet, so say so
    % rather than failing with an undefined-function message. ValueFnIter_FHorz raises the matching
    % error for this combination, so in practice you never reach here.
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu: QuasiHyperbolic experienceassetu with a semi-exogenous state is not yet implemented (the ExpAssetuSemiExo QH ValueFnFromPolicy subfns are not written). QuasiHyperbolic experienceassetu without semiz is supported.')
end
%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAssetu)
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetu you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
if ~isfield(vfoptions,'n_u')
    error('To use experienceassetu you must define vfoptions.n_u')
end
if ~isfield(vfoptions,'u_grid')
    error('To use experienceassetu you must define vfoptions.u_grid')
end
if ~isfield(vfoptions,'pi_u')
    error('To use experienceassetu you must define vfoptions.pi_u')
end
n_u=vfoptions.n_u;
u_grid=gpuArray(vfoptions.u_grid);
pi_u=gpuArray(vfoptions.pi_u);
N_u=prod(n_u);
l_u=length(n_u);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu: experienceassetu requires at least one decision variable (the one driving a2prime)')
end
l_d=length(n_d);
l_a=length(n_a);

if isscalar(n_a)
    n_a1=0;
    N_a1=0;
    l_a1=0;
else
    n_a1=n_a(1:end-1);
    N_a1=prod(n_a1);
    l_a1=length(n_a1);
end
n_a2=n_a(end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a2=length(n_a2);
l_aprime=l_a1;

if isfield(vfoptions,'l_dexperienceassetu')
    l_d2=vfoptions.l_dexperienceassetu;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;
n_d2=n_d(end-l_d2+1:end);

temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_u)
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+1:end}};
else
    aprimeFnParamNames={};
end

if N_z==0 && N_e==0
    N_ze=0;
elseif N_z>0 && N_e==0
    N_ze=N_z;
elseif N_z==0 && N_e>0
    N_ze=N_e;
else
    N_ze=N_z*N_e;
end

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues, Policy in Kron form, and the a1prime joint index -- for Policy, and Policyalt if Naive
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]);
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_j]);
    a1prime_idx=ones(N_a, N_j, 'gpuArray');
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
    if N_z>0 && N_e==0
        Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z, N_j]);
    elseif N_z==0 && N_e>0
        Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_e, N_j]);
    else
        Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z*N_e, N_j]);
    end
    a1prime_idx=ones(N_a, N_ze, N_j, 'gpuArray');
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    if N_z==0 && N_e==0
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,1,3]);
        Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_j]);
        a1prime_idx_alt=ones(N_a, N_j, 'gpuArray');
    else
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
        if N_z>0 && N_e==0
            Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_z, N_j]);
        elseif N_z==0 && N_e>0
            Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_e, N_j]);
        else
            Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_z*N_e, N_j]);
        end
        a1prime_idx_alt=ones(N_a, N_ze, N_j, 'gpuArray');
    end
    for ii=1:l_a1
        comp=shiftdim(Policyalt_k(l_d+ii, :, :, :),1);
        a1prime_idx_alt=a1prime_idx_alt+cumprods_a1(ii)*(comp-1);
    end
end

%% Joint z+e gridvals for ReturnFn when both present
if N_z>0 && N_e>0
    joint_zegridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% Two value functions (Vdrive uses beta and drives the recursion; Vrep uses beta0*beta and is reported as V)
if N_z==0 && N_e==0
    Vdrive=zeros(N_a, N_j, 'gpuArray');       Vrep=zeros(N_a, N_j, 'gpuArray');
elseif N_z==0 && N_e>0
    Vdrive=zeros(N_a, N_e, N_j, 'gpuArray');  Vrep=zeros(N_a, N_e, N_j, 'gpuArray');
elseif N_z>0 && N_e==0
    Vdrive=zeros(N_a, N_z, N_j, 'gpuArray');  Vrep=zeros(N_a, N_z, N_j, 'gpuArray');
else
    Vdrive=zeros(N_a, N_z, N_e, N_j, 'gpuArray'); Vrep=zeros(N_a, N_z, N_e, N_j, 'gpuArray');
end

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z==0 && N_e==0
        Policy_slice=Policy_k(:,:,jj);
    else
        Policy_slice=Policy_k(:,:,:,jj);
    end

    % Step 1: a2primeIndex, a2primeProbs at Policy (and at Policyalt if Naive)
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetu(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_ze, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    if isNaive
        if N_z==0 && N_e==0
            Policyalt_slice=Policyalt_k(:,:,jj);
        else
            Policyalt_slice=Policyalt_k(:,:,:,jj);
        end
        [a2primeIndex_alt, a2primeProbs_alt]=CreateaprimePolicyExperienceAssetu(Policyalt_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_ze, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    end

    % Step 2: ReturnFn at policy (u does not enter Return) (and at Policyalt if Naive)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_z==0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
    elseif N_z==0 && N_e>0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
    elseif N_z>0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
    end
    if isNaive
        if N_z==0 && N_e==0
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
        elseif N_z==0 && N_e>0
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
        elseif N_z>0 && N_e==0
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        else
            F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
        end
    end

    if jj==N_j
        % Terminal period: no continuation, so each value is just the return at its own policy
        if N_z==0 && N_e==0
            if isNaive, Vdrive(:,jj)=F_alt_jj; else, Vdrive(:,jj)=F_jj; end
            Vrep(:,jj)=F_jj;
        elseif N_z==0 && N_e>0
            if isNaive, Vdrive(:,:,jj)=F_alt_jj; else, Vdrive(:,:,jj)=F_jj; end
            Vrep(:,:,jj)=F_jj;
        elseif N_z>0 && N_e==0
            if isNaive, Vdrive(:,:,jj)=F_alt_jj; else, Vdrive(:,:,jj)=F_jj; end
            Vrep(:,:,jj)=F_jj;
        else
            if isNaive, Vdrive(:,:,:,jj)=F_alt_jj; else, Vdrive(:,:,:,jj)=F_jj; end
            Vrep(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
        beta0beta=beta0*beta;

        % Step 3: EVnext, built from the recursion-driver value (Vdrive)
        if N_z==0 && N_e==0
            EVnext=Vdrive(:,jj+1);
        elseif N_z==0 && N_e>0
            EVnext=sum(Vdrive(:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -1), 2); % [N_a, 1]
        elseif N_z>0 && N_e==0
            EVnext=Vdrive(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        else
            EVnext=sum(Vdrive(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_z, 1]
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        end

        % Step 4: interpolated lookup of EVnext at each policy's a2prime. Pass 1 is Policy (giving
        % the reported value's continuation); pass 2, when Naive, is Policyalt (the recursion
        % driver's). The interpolation itself is identical, hence the pass loop.
        for pass=1:(1+isNaive)
            if pass==1
                a2pi=a2primeIndex; a2pp=a2primeProbs; a1pi=a1prime_idx;
            else
                a2pi=a2primeIndex_alt; a2pp=a2primeProbs_alt; a1pi=a1prime_idx_alt;
            end

            if N_z==0 && N_e==0
                if N_a1==0
                    aprime_low=a2pi;
                    aprime_up =a2pi+1;
                else
                    a1p=a1pi(:,jj); % [N_a]
                    aprime_low=a1p+N_a1*(a2pi-1); % [N_a, N_u]
                    aprime_up =a1p+N_a1*(a2pi);
                end
                Vlower=reshape(EVnext(aprime_low(:)),[N_a,N_u]);
                Vupper=reshape(EVnext(aprime_up(:)), [N_a,N_u]);
                a2pPrb=a2pp;
                a2pPrb(Vlower==Vupper)=0; % skipinterp
                EV=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
                EVnext_pass=sum(EV .* shiftdim(pi_u,-1), 2); % sum over u -> [N_a, 1]
                EVnext_pass(isnan(EVnext_pass))=0;
            elseif N_z==0 && N_e>0
                if N_a1==0
                    aprime_low=a2pi;
                    aprime_up =a2pi+1;
                else
                    a1p=a1pi(:,:,jj); % [N_a, N_e]
                    aprime_low=a1p+N_a1*(a2pi-1); % broadcast -> [N_a, N_e, N_u]
                    aprime_up =a1p+N_a1*(a2pi);
                end
                % EVnext already has pi_e summed out (e is iid); matches standard VFI noz+e EVpre.
                Vlower=reshape(EVnext(aprime_low(:)),[N_a,N_e,N_u]);
                Vupper=reshape(EVnext(aprime_up(:)), [N_a,N_e,N_u]);
                a2pPrb=a2pp;
                a2pPrb(Vlower==Vupper)=0; % skipinterp on pi_e-collapsed EVnext
                EV=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
                EVnext_pass=sum(EV .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_e]
                EVnext_pass(isnan(EVnext_pass))=0;
            elseif N_z>0 && N_e==0
                if N_a1==0
                    aprime_low=a2pi;
                    aprime_up =a2pi+1;
                else
                    a1p=a1pi(:,:,jj);
                    aprime_low=a1p+N_a1*(a2pi-1); % broadcast -> [N_a, N_z, N_u]
                    aprime_up =a1p+N_a1*(a2pi);
                end
                % Per-z' interpolation against Vdrive(:,:,jj+1) directly (not EVnext, which has pi_z already summed).
                Vnext=Vdrive(:,:,jj+1); % [N_a, N_z']
                zprimeoffset=reshape(N_a*gpuArray(0:N_z-1),[1,1,1,N_z]); % [1,1,1,N_z']
                Vlower=reshape(Vnext(aprime_low+zprimeoffset),[N_a,N_z,N_u,N_z]);
                Vupper=reshape(Vnext(aprime_up +zprimeoffset),[N_a,N_z,N_u,N_z]);
                a2pPrb4=repmat(a2pp,[1,1,1,N_z]);
                a2pPrb4(Vlower==Vupper)=0; % skipinterp
                EV4=a2pPrb4.*Vlower+(1-a2pPrb4).*Vupper; % [N_a, N_z, N_u, N_z']
                EV4=sum(EV4 .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_z, 1, N_z']
                EV4=EV4 .* reshape(pi_z_J(:,:,jj),[1,N_z,1,N_z]); % weight by pi_z(z, z')
                EV4(isnan(EV4))=0;
                EVnext_pass=reshape(sum(EV4,4),[N_a,N_z]); % sum over z'
            else
                a2pIdx=reshape(a2pi,[N_a, N_z, N_e, N_u]);
                a2pPrb=reshape(a2pp,[N_a, N_z, N_e, N_u]);
                if N_a1==0
                    aprime_low=a2pIdx;     % [N_a, N_z, N_e, N_u]
                    aprime_up =a2pIdx+1;
                else
                    a1p=reshape(a1pi(:,:,jj),[N_a, N_z, N_e]);
                    aprime_low=a1p+N_a1*(a2pIdx-1); % broadcast a1p over u -> [N_a, N_z, N_e, N_u]
                    aprime_up =a1p+N_a1*(a2pIdx);
                end
                % Pre-collapse pi_e (e iid), keep z' for per-z' skipinterp (matches standard VFI z+e branch).
                EVpre=reshape(sum(Vdrive(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1),-2), 3),[N_a,N_z]); % [N_a, N_z']
                zprimeoffset=reshape(N_a*gpuArray(0:N_z-1),[1,1,1,1,N_z]); % [1,1,1,1,N_z']
                Vlower=reshape(EVpre(aprime_low+zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
                Vupper=reshape(EVpre(aprime_up +zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
                a2pPrb5=repmat(a2pPrb,[1,1,1,1,N_z]);
                a2pPrb5(Vlower==Vupper)=0; % skipinterp
                EV5=a2pPrb5.*Vlower+(1-a2pPrb5).*Vupper; % [N_a, N_z, N_e, N_u, N_z']
                EV5=sum(EV5 .* shiftdim(pi_u,-3), 4); % sum over u -> [N_a, N_z, N_e, 1, N_z']
                EV5=EV5 .* reshape(pi_z_J(:,:,jj),[1,N_z,1,1,N_z]); % weight by pi_z(z, z')
                EV5(isnan(EV5))=0;
                EVnext_pass=reshape(sum(EV5,5),[N_a,N_z,N_e]); % sum over z'
            end

            if pass==1
                EVnext_atP=EVnext_pass;
            else
                EVnext_atPa=EVnext_pass;
            end
        end

        % Step 5: Vdrive carries beta (and, when Naive, Policyalt's return and continuation);
        % Vrep carries beta0*beta at Policy and is what gets reported as V.
        if N_z==0 && N_e==0
            if isNaive
                Vdrive(:,jj)=F_alt_jj+beta*EVnext_atPa;
            else
                Vdrive(:,jj)=F_jj+beta*EVnext_atP;
            end
            Vrep(:,jj)=F_jj+beta0beta*EVnext_atP;
        elseif N_z==0 && N_e>0
            if isNaive
                Vdrive(:,:,jj)=F_alt_jj+beta*EVnext_atPa;
            else
                Vdrive(:,:,jj)=F_jj+beta*EVnext_atP;
            end
            Vrep(:,:,jj)=F_jj+beta0beta*EVnext_atP;
        elseif N_z>0 && N_e==0
            if isNaive
                Vdrive(:,:,jj)=F_alt_jj+beta*EVnext_atPa;
            else
                Vdrive(:,:,jj)=F_jj+beta*EVnext_atP;
            end
            Vrep(:,:,jj)=F_jj+beta0beta*EVnext_atP;
        else
            if isNaive
                Vdrive(:,:,:,jj)=F_alt_jj+beta*EVnext_atPa;
            else
                Vdrive(:,:,:,jj)=F_jj+beta*EVnext_atP;
            end
            Vrep(:,:,:,jj)=F_jj+beta0beta*EVnext_atP;
        end
    end
end

%% Output: V is the reported (beta0*beta) value; Valt is the recursion-driver (beta) value
if N_z==0 && N_e==0
    V   =reshape(Vrep,   [n_a, N_j]);
    Valt=reshape(Vdrive, [n_a, N_j]);
elseif N_z==0 && N_e>0
    V   =reshape(Vrep,   [n_a, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V   =reshape(Vrep,   [n_a, n_z, N_j]);
    Valt=reshape(Vdrive, [n_a, n_z, N_j]);
else
    V   =reshape(Vrep,   [n_a, n_z, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, n_z, vfoptions.n_e, N_j]);
end


end
