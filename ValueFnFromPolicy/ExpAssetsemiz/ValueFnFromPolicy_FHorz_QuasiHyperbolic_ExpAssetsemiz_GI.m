function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetsemiz_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, beta0)
% Compute V from a given Policy when the model has experienceassetsemiz (vfoptions.experienceassetsemiz>=1),
% uses the grid interpolation layer (vfoptions.gridinterplayer==1), and quasi-hyperbolic discounting.
% The experience asset a2 is driven by the semi-exogenous state: a2prime=aprimeFn(d_expasset,a2,semiz,...).
% semiz is always present; z is an OPTIONAL ordinary Markov shock.
% Convention on d ordering: d = [...other d..., d_expasset, d_semiz]. d_semiz is the last l_dsemiz
% components; d_expasset is the l_d2 components immediately before them.
% Joint shock = [semiz, z] (semiz fastest), matches ValueFnIter_FHorz_ExpAssetsemiz.
%
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext_byd2) is ALWAYS built from the recursion-driver value (Vdrive).
%
% Structural base: ValueFnFromPolicy_FHorz_ExpAssetsemiz_GI (the 2x2 corner interpolation, the
% d_semiz indirection and every gather stride are carried over untouched). QH bookkeeping mirrors
% ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset_SemiExo_GI and _QuasiHyperbolic_ExpAssetsemiz.
%
% Per-state EVnext lookup combines three pieces:
%   - a1 fine-grid interpolation (lower/upper a1 grid point + L2 weight)
%   - a2 interpolation via a2primeIndex/a2primeProbs (from CreateaprimePolicyExperienceAssetsemiz)
%   - d_semiz indirection: pick the d2-slice of EVnext_byd2 via d_semiz_idx
%
% The interpolated lookup is written once and run over a pass loop ({Policy} or {Policy,Policyalt}),
% since the interpolation is identical for the two policies -- only the indices/weights differ.
% Under GI each policy carries BOTH an a1 coarse index and an L2 weight, so both passes swap
% a1_lower/a1_upper AND w_a1_lower/w_a1_upper, alongside a2primeIndex/a2primeProbs and d_semiz_idx.

%% Setup
% Semiz gridvals + pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetsemiz you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;

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
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetsemiz_GI: experienceassetsemiz requires at least one decision variable')
end
l_d=length(n_d);
l_semiz=length(n_semiz);

% z is optional. Use N_zloc (=1 when no z) for the reshapes; build a trivial pi_z when there is no z.
N_zloc=max(N_z,1);
if N_z==0
    z_gridvals_J=[];
    pi_z_J=ones(1,1,N_j,'gpuArray'); % single 'z' that transitions to itself (integration over z' is a no-op)
else
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);
end

if isscalar(n_a)
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetsemiz_GI: with noa1 there is no standard endogenous asset to interpolate (turn off vfoptions.gridinterplayer)')
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);

if isfield(vfoptions,'l_dexperienceassetsemiz')
    l_d2=vfoptions.l_dexperienceassetsemiz;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_dsemiz-l_d2+1):(l_d-l_dsemiz);

temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_semiz)
    aprimeFnParamNames={temp{l_d2+l_a2+l_semiz+1:end}};
else
    aprimeFnParamNames={};
end

% Joint shock = [semiz, z] (semiz fastest); when no z it is just semiz
if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*N_zloc;

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetsemiz+GI internally)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
end

%% Strip trailing L2flag channel from Policy if present
size_first=l_d+l_a1+1; % under GI: d, a1mid, L2 -> l_d + l_a1 + 1 channels
if size(Policy,1) > size_first
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:size_first,:), [size_first, tempsize(2:end)]);
end
if isNaive
    if size(Policyalt,1) > size_first
        tempsize=size(Policyalt);
        Policyalt=reshape(Policyalt,[tempsize(1), prod(tempsize)/tempsize(1)]);
        Policyalt=reshape(Policyalt(1:size_first,:), [size_first, tempsize(2:end)]);
    end
end

%% Reshape Policy to canonical Kron form
if N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_j]);
else
    Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_e, N_j]);
end
if isNaive
    if N_e==0
        Policyalt_k=reshape(Policyalt,[size_first, N_a, N_shocks, N_j]);
    else
        Policyalt_k=reshape(Policyalt,[size_first, N_a, N_shocks, N_e, N_j]);
    end
end

%% d_semiz_idx: last l_dsemiz components of d
if N_e==0
    d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end
if isNaive
    if N_e==0
        d_semiz_idx_alt=ones(N_a,N_shocks,N_j,'gpuArray');
    else
        d_semiz_idx_alt=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    end
    for ii=1:l_dsemiz
        comp=shiftdim(Policyalt_k(l_d-l_dsemiz+ii, :, :, :, :),1);
        d_semiz_idx_alt=d_semiz_idx_alt+cumprods_dsemiz(ii)*(comp-1);
    end
end

%% a1prime: lower grid index (position l_d+1) + L2 (last). Other a1 components at l_d+2..l_d+l_a1.
% ValueFnIter converts the midpoint to the lower grid index before returning Policy (the adjust
% block at the end of the GI raws), so this row is the lower index and not the midpoint.
a1_lowerind=shiftdim(Policy_k(l_d+1,:,:,:,:),1);
L2    =shiftdim(Policy_k(l_d+l_a1+1,:,:,:,:),1);
w_a1_upper=(L2-1)/(n2short+1);
w_a1_lower=1-w_a1_upper;
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_lower=a1_lowerind;
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
a1_upper=a1_lower+1;
a1_top_clamp=(a1_lowerind>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp);
if isNaive
    a1_lowerind_alt=shiftdim(Policyalt_k(l_d+1,:,:,:,:),1);
    L2_alt    =shiftdim(Policyalt_k(l_d+l_a1+1,:,:,:,:),1);
    w_a1_upper_alt=(L2_alt-1)/(n2short+1);
    w_a1_lower_alt=1-w_a1_upper_alt;
    a1_lower_alt=a1_lowerind_alt;
    for ii=2:l_a1
        comp=shiftdim(Policyalt_k(l_d+ii,:,:,:,:),1);
        a1_lower_alt=a1_lower_alt+cumprods_a1(ii)*(comp-1);
    end
    a1_upper_alt=a1_lower_alt+1;
    a1_top_clamp_alt=(a1_lowerind_alt>=n_a1(1));
    a1_upper_alt(a1_top_clamp_alt)=a1_lower_alt(a1_top_clamp_alt);
end

%% Joint shock gridvals for ReturnFn
if N_z==0
    joint_gridvals_J=semiz_gridvals_J; % bothz = semiz
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

[~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_zloc);

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at Policy (and at Policyalt if Naive). aprime driven by semiz.
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_e==0
        Policy_slice=Policy_k(:,:,:,jj); % [size_first, N_a, N_shocks]
    else
        Policy_slice=reshape(Policy_k(:,:,:,:,jj), [size_first, N_a, N_shocks*N_e]);
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetsemiz(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_semiz, N_semiz, N_zloc, N_e, d_grid, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % Shapes:
    %   N_e==0: [N_a, N_shocks]
    %   N_e>0:  [N_a, N_shocks*N_e]
    if isNaive
        if N_e==0
            Policyalt_slice=Policyalt_k(:,:,:,jj);
        else
            Policyalt_slice=reshape(Policyalt_k(:,:,:,:,jj), [size_first, N_a, N_shocks*N_e]);
        end
        [a2primeIndex_alt, a2primeProbs_alt]=CreateaprimePolicyExperienceAssetsemiz(Policyalt_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_semiz, N_semiz, N_zloc, N_e, d_grid, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec);
    end

    % Step 2: ReturnFn at policy (and at Policyalt if Naive)
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

        % Step 3b: integrate over z' (markov). Trivial when no z.
        V_next_r=reshape(V_next, [N_a, N_semiz, N_zloc]);
        EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3);
        EV_after_z(isnan(EV_after_z))=0;
        EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_zloc]);

        % Step 3c: for each d_semiz, integrate over semiz' -> EVnext_byd2(a, semiz_from, z_from, d_semiz)
        EVnext_byd2=zeros(N_a, N_semiz, N_zloc, N_dsemiz, 'gpuArray');
        for d2_c=1:N_dsemiz
            pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
            pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
            EVd2c=sum(EV_after_z .* pi_reshape, 2);
            EVd2c(isnan(EVd2c))=0;
            EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_zloc]);
        end

        % Step 4: per-state 2x2 corner interpolation on EVnext_byd2 at d_semiz-selected slice.
        % Corners: (a1_low, a2_low), (a1_low, a2_up), (a1_up, a2_low), (a1_up, a2_up)
        % Pass 1 is Policy (giving the reported value's continuation); pass 2, when Naive, is
        % Policyalt (the recursion driver's). Under GI each pass swaps BOTH policy channels
        % (the coarse a1 index a1Lo/a1Up and the L2 weights wLo/wUp), plus a2prime and d_semiz.
        for pass=1:(1+isNaive)
            if pass==1
                a1Lo=a1_lower; a1Up=a1_upper; wLo=w_a1_lower; wUp=w_a1_upper;
                a2pi=a2primeIndex; a2pp=a2primeProbs; d2i=d_semiz_idx;
            else
                a1Lo=a1_lower_alt; a1Up=a1_upper_alt; wLo=w_a1_lower_alt; wUp=w_a1_upper_alt;
                a2pi=a2primeIndex_alt; a2pp=a2primeProbs_alt; d2i=d_semiz_idx_alt;
            end

            if N_e==0
                a1l=a1Lo(:,:,jj); a1u=a1Up(:,:,jj);
                wa1l=wLo(:,:,jj); wa1u=wUp(:,:,jj);
                d2_jj=d2i(:,:,jj);
                a1l_r=reshape(a1l,[N_a, N_semiz, N_zloc]); a1u_r=reshape(a1u,[N_a, N_semiz, N_zloc]);
                wa1l_r=reshape(wa1l,[N_a, N_semiz, N_zloc]); wa1u_r=reshape(wa1u,[N_a, N_semiz, N_zloc]);
                a2l=reshape(a2pi,[N_a, N_semiz, N_zloc]); a2u=a2l+1;
                wa2l=reshape(a2pp,[N_a, N_semiz, N_zloc]); wa2u=1-wa2l;
                d2_r=reshape(d2_jj,[N_a, N_semiz, N_zloc]);
                base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_zloc*(d2_r(:)-1), [N_a, N_semiz, N_zloc]);
                lin_LL=a1l_r+N_a1*(a2l-1)+base_off;
                lin_LU=a1l_r+N_a1*(a2u-1)+base_off;
                lin_UL=a1u_r+N_a1*(a2l-1)+base_off;
                lin_UU=a1u_r+N_a1*(a2u-1)+base_off;
                EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_zloc]);
                EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_zloc]);
                EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_zloc]);
                EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_zloc]);
                EVnext_pass=wa1l_r.*wa2l.*EV_LL + wa1l_r.*wa2u.*EV_LU + wa1u_r.*wa2l.*EV_UL + wa1u_r.*wa2u.*EV_UU;
            else
                EVnext_pass=zeros(N_a, N_semiz, N_zloc, N_e, 'gpuArray');
                for e_c=1:N_e
                    block=(e_c-1)*N_shocks + (1:N_shocks);
                    a1l_e=reshape(a1Lo(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                    a1u_e=reshape(a1Up(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                    wa1l_e=reshape(wLo(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                    wa1u_e=reshape(wUp(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                    a2l_e=reshape(a2pi(:,block),[N_a, N_semiz, N_zloc]); a2u_e=a2l_e+1;
                    wa2l_e=reshape(a2pp(:,block),[N_a, N_semiz, N_zloc]); wa2u_e=1-wa2l_e;
                    d2_e=reshape(d2i(:,:,e_c,jj),[N_a, N_semiz, N_zloc]);
                    base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_zloc*(d2_e(:)-1), [N_a, N_semiz, N_zloc]);
                    lin_LL=a1l_e+N_a1*(a2l_e-1)+base_off;
                    lin_LU=a1l_e+N_a1*(a2u_e-1)+base_off;
                    lin_UL=a1u_e+N_a1*(a2l_e-1)+base_off;
                    lin_UU=a1u_e+N_a1*(a2u_e-1)+base_off;
                    EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_zloc]);
                    EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_zloc]);
                    EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_zloc]);
                    EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_zloc]);
                    EVnext_pass(:,:,:,e_c)=wa1l_e.*wa2l_e.*EV_LL + wa1l_e.*wa2u_e.*EV_LU + wa1u_e.*wa2l_e.*EV_UL + wa1u_e.*wa2u_e.*EV_UU;
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
if N_z==0
    if N_e==0
        V   =reshape(Vrep,   [n_a, n_semiz, N_j]);
        Valt=reshape(Vdrive, [n_a, n_semiz, N_j]);
    else
        V   =reshape(Vrep,   [n_a, n_semiz, vfoptions.n_e, N_j]);
        Valt=reshape(Vdrive, [n_a, n_semiz, vfoptions.n_e, N_j]);
    end
else
    if N_e==0
        V   =reshape(Vrep,   [n_a, n_semiz, n_z, N_j]);
        Valt=reshape(Vdrive, [n_a, n_semiz, n_z, N_j]);
    else
        V   =reshape(Vrep,   [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
        Valt=reshape(Vdrive, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
    end
end


end
