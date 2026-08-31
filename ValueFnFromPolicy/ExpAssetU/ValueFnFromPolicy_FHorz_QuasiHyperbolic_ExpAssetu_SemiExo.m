function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu_SemiExo(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, beta0)
% Compute V from a given Policy when the model has experienceassetu (vfoptions.experienceassetu>=1),
% semi-exogenous shocks (n_semiz>0), and quasi-hyperbolic discounting.
%
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext_byd2) is ALWAYS built from the recursion-driver value (Vdrive).
%
% Structural base: ValueFnFromPolicy_FHorz_ExpAssetu_SemiExo (the a2prime lottery, the skipinterp
% rule, the d_semiz indirection and the u integration are carried over untouched). QH bookkeeping
% mirrors ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset_SemiExo and
% ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu.
%
% experienceassetu: a2prime=aprimeFn(d_expasset, a2, u), with u an iid between-period shock that does
% NOT enter Policy. a2primeIndex/a2primeProbs therefore carry a trailing N_u dimension, and u is
% integrated out with pi_u inside the pass loop -- before the isnan clear, and before any discount
% factor is applied, so both QH passes share one u-integrated continuation.
%
% The per-state lookup is written once and run over a pass loop ({Policy} or {Policy,Policyalt}),
% so there is a single copy of it to check against the exponential source.

%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu_SemiExo_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, beta0);
    return
end

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAssetu_SemiExo)
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetu you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
if ~isfield(vfoptions,'n_u'),    error('To use experienceassetu you must define vfoptions.n_u'),    end
if ~isfield(vfoptions,'u_grid'), error('To use experienceassetu you must define vfoptions.u_grid'), end
if ~isfield(vfoptions,'pi_u'),   error('To use experienceassetu you must define vfoptions.pi_u'),   end
n_u=vfoptions.n_u;
u_grid=gpuArray(vfoptions.u_grid);
pi_u=gpuArray(vfoptions.pi_u);
N_u=prod(n_u);
l_u=length(n_u);

n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);

if isfield(vfoptions,'l_dsemiz')
    l_dsemiz=vfoptions.l_dsemiz;
else
    l_dsemiz=1;
end
n_dsemiz=n_d(end-l_dsemiz+1:end);
N_dsemiz=prod(n_dsemiz);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetu_SemiExo: experienceassetu+semiz requires at least one decision variable')
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
whichisdforexpasset=(l_d-l_dsemiz-l_d2+1):(l_d-l_dsemiz);
n_d2=n_d(whichisdforexpasset);

temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_u)
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+1:end}};
else
    aprimeFnParamNames={};
end

if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*max(N_z,1);

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues, Policy in Kron form, d_semiz index and a1prime index -- for Policy, and Policyalt if Naive
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
if N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_shocks, N_j]);
    d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
    a1prime_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_shocks, N_e, N_j]);
    d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    a1prime_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
    if N_e==0
        Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_shocks, N_j]);
        d_semiz_idx_alt=ones(N_a,N_shocks,N_j,'gpuArray');
        a1prime_idx_alt=ones(N_a,N_shocks,N_j,'gpuArray');
    else
        Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_shocks, N_e, N_j]);
        d_semiz_idx_alt=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
        a1prime_idx_alt=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    end
    for ii=1:l_dsemiz
        comp=shiftdim(Policyalt_k(l_d-l_dsemiz+ii, :, :, :, :),1);
        d_semiz_idx_alt=d_semiz_idx_alt+cumprods_dsemiz(ii)*(comp-1);
    end
    for ii=1:l_a1
        comp=shiftdim(Policyalt_k(l_d+ii, :, :, :, :),1);
        a1prime_idx_alt=a1prime_idx_alt+cumprods_a1(ii)*(comp-1);
    end
end

%% Joint shock gridvals for ReturnFn
if N_z==0
    joint_gridvals_J=semiz_gridvals_J;
else
    joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
    for jj=1:N_j
        joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
    end
end

%% Two value functions (Vdrive uses beta and drives the recursion; Vrep uses beta0*beta and is reported as V)
if N_e==0
    Vdrive=zeros(N_a, N_shocks, N_j, 'gpuArray');      Vrep=zeros(N_a, N_shocks, N_j, 'gpuArray');
else
    Vdrive=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray'); Vrep=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
end

[~, SZ_grid_noz]=ndgrid(1:N_a, 1:N_semiz);
if N_z>0
    [~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);
end

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at Policy (and at Policyalt if Naive) -- helper adds the u dim
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_e==0
        Policy_slice=Policy_k(:,:,:,jj);
    else
        Policy_slice=reshape(Policy_k(:,:,:,:,jj), [l_d+l_a1, N_a, N_shocks*N_e]);
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetu(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_shocks*max(N_e,1), n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % Shapes (helper passes N_z arg=N_shocks*max(N_e,1)>0 so the N_z>0 branch is always taken):
    %   N_e==0: [N_a, N_shocks,    N_u]
    %   N_e>0:  [N_a, N_shocks*N_e, N_u]
    if isNaive
        if N_e==0
            Policyalt_slice=Policyalt_k(:,:,:,jj);
        else
            Policyalt_slice=reshape(Policyalt_k(:,:,:,:,jj), [l_d+l_a1, N_a, N_shocks*N_e]);
        end
        [a2primeIndex_alt, a2primeProbs_alt]=CreateaprimePolicyExperienceAssetu(Policyalt_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_shocks*max(N_e,1), n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    end

    % Step 2: ReturnFn at policy (u does not enter Return) (and at Policyalt if Naive)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
    end
    if isNaive
        if N_e==0
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
        else
            F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
        end
    end

    if jj==N_j
        % Terminal period: no continuation, so each value is just the return at its own policy
        if N_e==0
            if isNaive, Vdrive(:,:,jj)=F_alt_jj; else, Vdrive(:,:,jj)=F_jj; end
            Vrep(:,:,jj)=F_jj;
        else
            if isNaive, Vdrive(:,:,:,jj)=F_alt_jj; else, Vdrive(:,:,:,jj)=F_jj; end
            Vrep(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        beta0beta=beta0*beta;

        % Step 3a: integrate the recursion-driver value over e' (if any)
        if N_e==0
            V_next=Vdrive(:,:,jj+1);
        else
            V_next=Vdrive(:,:,:,jj+1);
            V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Step 3b: integrate over z' (markov, does not depend on d_semiz)
        if N_z==0
            EV_after_z=V_next;
        else
            V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
            EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3); % [N_a, N_semiz_to, 1, N_z_from]
            EV_after_z(isnan(EV_after_z))=0;
            EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]);
        end

        % Step 3c: for each d_semiz, integrate over semiz' -> EVnext_byd2
        if N_z==0
            EVnext_byd2=zeros(N_a, N_semiz, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
                EVd2c=sum(EV_after_z .* shiftdim(pi_d2c, -1), 2);
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,d2_c)=reshape(EVd2c, [N_a, N_semiz]);
            end
        else
            EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
                pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
                EVd2c=sum(EV_after_z .* pi_reshape, 2);
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
            end
        end

        % Step 4: per-state lookup. Mirrors standard SemiExo VFI's skipinterp+isnan
        % so policies landing on infeasible-on-both-sides next-states give the same
        % finite V as the argmax V. EVnext_byd2 is already pre-collapsed over (z', semiz')
        % per d_semiz choice; skipinterp triggers when EV_lo == EV_up at policy state.
        % Order: skipinterp -> interpolate -> sum over u -> isnan clear.
        % Pass 1 is Policy (giving the reported value's continuation); pass 2, when Naive, is
        % Policyalt (the recursion driver's). Both passes integrate u out with pi_u, so the
        % continuation each pass hands to Step 5 is already u-integrated and undiscounted.
        for pass=1:(1+isNaive)
            if pass==1
                a2pi=a2primeIndex; a2pp=a2primeProbs; a1pi=a1prime_idx; d2i=d_semiz_idx;
            else
                a2pi=a2primeIndex_alt; a2pp=a2primeProbs_alt; a1pi=a1prime_idx_alt; d2i=d_semiz_idx_alt;
            end

            if N_e==0
                a1p=a1pi(:,:,jj);   % [N_a, N_shocks]
                d2_jj=d2i(:,:,jj); % [N_a, N_shocks]
                if N_z==0
                    d2_r =reshape(d2_jj,[N_a, N_semiz]);
                    a2pIdx=a2pi; a2pPrb=a2pp;        % [N_a, N_semiz, N_u]
                    if N_a1==0
                        aprime_low=a2pIdx;                            % [N_a, N_semiz, N_u]
                        aprime_up =a2pIdx+1;
                    else
                        a1p_r=reshape(a1p,[N_a, N_semiz]);
                        aprime_low=a1p_r+N_a1*(a2pIdx-1);             % [N_a, N_semiz, N_u]
                        aprime_up =a1p_r+N_a1*(a2pIdx);
                    end
                    base_off=reshape(N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_r(:)-1), [N_a, N_semiz]);
                    lo_idx=aprime_low+base_off; % broadcast over u
                    up_idx=aprime_up +base_off;
                    EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_u]);
                    EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_u]);
                    a2pPrb(EV_lo==EV_up)=0; % skipinterp
                    per_u=a2pPrb.*EV_lo+(1-a2pPrb).*EV_up;
                    EVnext_pass=sum(per_u .* shiftdim(pi_u,-2), 3); % [N_a, N_semiz]
                    EVnext_pass(isnan(EVnext_pass))=0;
                else
                    d2_r =reshape(d2_jj,[N_a, N_semiz, N_z]);
                    a2pIdx=reshape(a2pi,[N_a, N_semiz, N_z, N_u]);
                    a2pPrb=reshape(a2pp,[N_a, N_semiz, N_z, N_u]);
                    if N_a1==0
                        aprime_low=a2pIdx;
                        aprime_up =a2pIdx+1;
                    else
                        a1p_r=reshape(a1p,[N_a, N_semiz, N_z]);
                        aprime_low=a1p_r+N_a1*(a2pIdx-1);
                        aprime_up =a1p_r+N_a1*(a2pIdx);
                    end
                    base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r(:)-1), [N_a, N_semiz, N_z]);
                    lo_idx=aprime_low+base_off;
                    up_idx=aprime_up +base_off;
                    EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_z, N_u]);
                    EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_z, N_u]);
                    a2pPrb(EV_lo==EV_up)=0; % skipinterp
                    per_u=a2pPrb.*EV_lo+(1-a2pPrb).*EV_up;
                    EVnext_pass=sum(per_u .* shiftdim(pi_u,-3), 4); % [N_a, N_semiz, N_z]
                    EVnext_pass(isnan(EVnext_pass))=0;
                end
            else
                if N_z==0
                    EVnext_pass=zeros(N_a, N_semiz, N_e, 'gpuArray');
                    for e_c=1:N_e
                        block=(e_c-1)*N_shocks + (1:N_shocks);
                        d2_e =reshape(d2i(:,:,e_c,jj),[N_a, N_semiz]);
                        a2pIdx_e=reshape(a2pi(:,block,:),[N_a, N_semiz, N_u]);
                        a2pPrb_e=reshape(a2pp(:,block,:),[N_a, N_semiz, N_u]);
                        if N_a1==0
                            aprime_low_e=a2pIdx_e;
                            aprime_up_e =a2pIdx_e+1;
                        else
                            a1p_e=reshape(a1pi(:,:,e_c,jj),[N_a, N_semiz]);
                            aprime_low_e=a1p_e+N_a1*(a2pIdx_e-1);
                            aprime_up_e =a1p_e+N_a1*(a2pIdx_e);
                        end
                        base_off=reshape(N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1), [N_a, N_semiz]);
                        lo_idx=aprime_low_e+base_off;
                        up_idx=aprime_up_e +base_off;
                        EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_u]);
                        EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_u]);
                        a2pPrb_e(EV_lo==EV_up)=0; % skipinterp
                        per_u=a2pPrb_e.*EV_lo+(1-a2pPrb_e).*EV_up;
                        EV_summed=sum(per_u .* shiftdim(pi_u,-2), 3);
                        EV_summed(isnan(EV_summed))=0;
                        EVnext_pass(:,:,e_c)=EV_summed;
                    end
                else
                    EVnext_pass=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
                    for e_c=1:N_e
                        block=(e_c-1)*N_shocks + (1:N_shocks);
                        d2_e =reshape(d2i(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                        a2pIdx_e=reshape(a2pi(:,block,:),[N_a, N_semiz, N_z, N_u]);
                        a2pPrb_e=reshape(a2pp(:,block,:),[N_a, N_semiz, N_z, N_u]);
                        if N_a1==0
                            aprime_low_e=a2pIdx_e;
                            aprime_up_e =a2pIdx_e+1;
                        else
                            a1p_e=reshape(a1pi(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                            aprime_low_e=a1p_e+N_a1*(a2pIdx_e-1);
                            aprime_up_e =a1p_e+N_a1*(a2pIdx_e);
                        end
                        base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1), [N_a, N_semiz, N_z]);
                        lo_idx=aprime_low_e+base_off;
                        up_idx=aprime_up_e +base_off;
                        EV_lo=reshape(EVnext_byd2(lo_idx(:)),[N_a, N_semiz, N_z, N_u]);
                        EV_up=reshape(EVnext_byd2(up_idx(:)),[N_a, N_semiz, N_z, N_u]);
                        a2pPrb_e(EV_lo==EV_up)=0; % skipinterp
                        per_u=a2pPrb_e.*EV_lo+(1-a2pPrb_e).*EV_up;
                        EV_summed=sum(per_u .* shiftdim(pi_u,-3), 4);
                        EV_summed(isnan(EV_summed))=0;
                        EVnext_pass(:,:,:,e_c)=EV_summed;
                    end
                end
            end

            if pass==1
                EVnext_atP=EVnext_pass;
            else
                EVnext_atPa=EVnext_pass;
            end
        end

        % Step 5: Vdrive carries beta (and, when Naive, Policyalt's return and continuation);
        % Vrep carries beta0*beta at Policy and is what gets reported as V.
        if N_e==0
            if isNaive
                Vdrive(:,:,jj)=F_alt_jj+beta*reshape(EVnext_atPa, [N_a, N_shocks]);
            else
                Vdrive(:,:,jj)=F_jj+beta*reshape(EVnext_atP, [N_a, N_shocks]);
            end
            Vrep(:,:,jj)=F_jj+beta0beta*reshape(EVnext_atP, [N_a, N_shocks]);
        else
            if isNaive
                Vdrive(:,:,:,jj)=F_alt_jj+beta*reshape(EVnext_atPa, [N_a, N_shocks, N_e]);
            else
                Vdrive(:,:,:,jj)=F_jj+beta*reshape(EVnext_atP, [N_a, N_shocks, N_e]);
            end
            Vrep(:,:,:,jj)=F_jj+beta0beta*reshape(EVnext_atP, [N_a, N_shocks, N_e]);
        end
    end
end

%% Output: V is the reported (beta0*beta) value; Valt is the recursion-driver (beta) value
if N_z==0 && N_e==0
    V   =reshape(Vrep,   [n_a, n_semiz, N_j]);
    Valt=reshape(Vdrive, [n_a, n_semiz, N_j]);
elseif N_z==0 && N_e>0
    V   =reshape(Vrep,   [n_a, n_semiz, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, n_semiz, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V   =reshape(Vrep,   [n_a, n_semiz, n_z, N_j]);
    Valt=reshape(Vdrive, [n_a, n_semiz, n_z, N_j]);
else
    V   =reshape(Vrep,   [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
end


end
