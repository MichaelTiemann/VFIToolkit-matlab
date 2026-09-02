function V=ValueFnFromPolicy_FHorz_AmbAverse_RiskyAsset_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% AmbiguityAversion + gridinterplayer variant of ValueFnFromPolicy_FHorz_RiskyAsset_GI: values
% the given Policy under the min-over-priors continuation, mirroring the AmbAverse riskyasset GI
% raws' order of operations: sequential mins (e on the grid, then z at the a2 lottery -- the
% lottery is conditional on the prior -- then u) are all taken at the COARSE a1prime nodes, and
% the a1 grid-interp layer then interpolates the worst-case EV between the two coarse corners.
% (The a1 interp layer cannot be per-prior in this family: the d2 refinement max sits between the
% mins and the interp in the raws, matching the vNM donor's max-at-coarse-then-interp; see the
% proposal's STATUS note.) u is treated as ambiguity: ambiguity_pi_u mandatory.
% Compute V from a given Policy with riskyasset AND grid interpolation layer (vfoptions.gridinterplayer==1).
% riskyasset: a2prime = aprimeFn(d2,d3, u) -- u is iid between-period shock, NOT current a2.
% Under GI, Policy carries an L2 fine-grid index for a1prime; lookup is 2x2 in (a1,a2), then summed over u with pi_u.
%
% Structural cousin of ValueFnFromPolicy_FHorz_ExpAssetu_GI. The only substantive differences are:
% (i) the aprime-from-policy helper is CreateaprimePolicyRiskyAsset (riskyasset's aprimeFn ignores
% current a2); (ii) which d feed the aprimeFn is set by vfoptions.refine_d (d2 and d3), not by
% l_dexperienceassetu; (iii) the riskyasset ReturnFn drops d2, so PolicyValues keeps only d1,d3,a1prime.

%% Setup
% z_gridvals_J passed in; the ambiguity priors were processed by ExogShockSetup_FHorz_AmbiguityAversion
n_ambiguity=vfoptions.n_ambiguity; % [1,N_j]
ambiguity_pi_u=gpuArray(vfoptions.ambiguity_pi_u); % [N_u,max(n_ambiguity)]

if ~isfield(vfoptions,'aprimeFn')
    error('To use riskyasset you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
if ~isfield(vfoptions,'n_u'),    error('To use riskyasset you must define vfoptions.n_u'),    end
if ~isfield(vfoptions,'u_grid'), error('To use riskyasset you must define vfoptions.u_grid'), end
if ~isfield(vfoptions,'pi_u'),   error('To use riskyasset you must define vfoptions.pi_u'),   end
n_u=vfoptions.n_u;
u_grid=gpuArray(vfoptions.u_grid);
% (the regular pi_u is not used here: u is ambiguity, and the u-priors replace it)
N_u=prod(n_u);
l_u=length(n_u);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_RiskyAsset_GI: riskyasset requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);

% noa1 case (n_a is scalar -- risky asset is the only endogenous state): GI refines a1, which
% doesn't apply when there's no a1. Fall back to non-GI version (which handles noa1 correctly).
% Matches the upstream VFI convention (noa1 has no GI/DC/DC+GI raw files).
if isscalar(n_a)
    vfoptions.gridinterplayer=0;
    V=ValueFnFromPolicy_FHorz_AmbAverse_RiskyAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);
l_aprime=l_a1;

% Which d feed the aprimeFn: d2 and d3 under vfoptions.refine_d (matches StationaryDist_FHorz_RiskyAsset).
if ~isfield(vfoptions,'refine_d')
    vfoptions.refine_d=[0,0,length(n_d)]; % everything implicitly a d3 (in both aprimeFn and ReturnFn)
end
whichisdforriskyasset=(vfoptions.refine_d(1)+1):1:sum(vfoptions.refine_d(1:3));
l_drisky=length(whichisdforriskyasset);

% aprimeFnParamNames: first inputs to the riskyasset aprimeFn are (d2,d3, u) -- NO current a2.
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_drisky+l_u)
    aprimeFnParamNames={temp{l_drisky+l_u+1:end}};
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

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues (PolicyInd2Val_FHorz handles riskyasset internally)
% The riskyasset ReturnFn takes (d1, d3, a1prime, a1, a2, ...): d2 (aprimeFn-only) is NOT an input.
% This is the 'refine' split the VFI codes use (vfoptions.refine_d): d1 in ReturnFn only,
% d2 in aprimeFn only, d3 in both -- so keep only the d1 and d3 decision rows (then a1prime).
d1rows=1:vfoptions.refine_d(1);
d3rows=(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1):(vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3));
returnrows=[d1rows, d3rows, (l_d+1):(l_d+l_a1)]; % ReturnFn d's (d1 then d3), then a1prime
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
PolicyValues=PolicyValues(returnrows,:,:,:); % drop d2 (not in the ReturnFn)
l_daprime=size(PolicyValues,1); % = n_d1 + n_d3 + l_a1
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]);
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
end

%% Strip trailing L2flag channel if present
size_first=l_d+l_a1+1;
if size(Policy,1) > size_first
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:size_first,:), [size_first, tempsize(2:end)]);
end

%% Reshape Policy
if N_z==0 && N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_j]);
elseif N_z>0 && N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_z, N_j]);
elseif N_z==0 && N_e>0
    Policy_k=reshape(Policy,[size_first, N_a, N_e, N_j]);
else
    Policy_k=reshape(Policy,[size_first, N_a, N_z*N_e, N_j]);
end

%% Extract a1prime lower grid index and L2
% ValueFnIter converts the midpoint to the lower grid index before returning Policy (the adjust
% block at the end of the GI raws), so this row is the lower index and not the midpoint.
a1_lowerind=shiftdim(Policy_k(l_d+1,:,:,:),1);
L2=shiftdim(Policy_k(l_d+l_a1+1,:,:,:),1);
w_a1_upper=(L2-1)/(n2short+1);
w_a1_lower=1-w_a1_upper;
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_lower=a1_lowerind;
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
a1_upper=a1_lower+1;
a1_top_clamp=(a1_lowerind>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp);

%% Joint z+e gridvals for ReturnFn when both present
if N_z>0 && N_e>0
    joint_zegridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% V allocation
if N_z==0 && N_e==0
    V=zeros(N_a, N_j, 'gpuArray');
elseif N_z==0 && N_e>0
    V=zeros(N_a, N_e, N_j, 'gpuArray');
elseif N_z>0 && N_e==0
    V=zeros(N_a, N_z, N_j, 'gpuArray');
else
    V=zeros(N_a, N_z, N_e, N_j, 'gpuArray');
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

    % Step 1: a2primeIndex, a2primeProbs -- helper adds the u dim (riskyasset: aprime ignores a2)
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyRiskyAsset(Policy_slice, aprimeFn, whichisdforriskyasset, n_d, n_a1, n_a2, N_ze, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % shape: N_z==0 && N_e==0 -> [N_a, N_u]; else -> [N_a, N_ze, N_u]

    % Step 2: ReturnFn at policy
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

    if jj==N_j
        if N_z==0 && N_e==0
            V(:,jj)=F_jj;
        elseif N_z==0 && N_e>0
            V(:,:,jj)=F_jj;
        elseif N_z>0 && N_e==0
            V(:,:,jj)=F_jj;
        else
            V(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3 is folded into Step 4: the e/z expectations are per-prior, and the mins are
        % taken at each coarse a1prime corner (the raws' order: mins at coarse, then the a1
        % grid-interp layer interpolates the worst-case EV).

        % Step 4: per-corner mins (a2 lottery conditional on the prior), then the a1 mix
        if N_z==0 && N_e==0
            % a1l, a1u, wa1l, wa1u: [N_a]; a2primeIndex/Probs: [N_a, N_u]
            a1l=a1_lower(:,jj); a1u=a1_upper(:,jj);
            wa1l=w_a1_lower(:,jj); wa1u=w_a1_upper(:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            % Broadcast a1l/u (N_a) over u → [N_a, N_u]
            EVnext=V(:,jj+1);
            % Worst case over the u-priors at each coarse a1 corner, then the a1 mix
            EV1=wa2l.*reshape(EVnext(a1l+N_a1*(a2l-1)),[N_a,N_u]);
            EV2=wa2u.*reshape(EVnext(a1l+N_a1*(a2u-1)),[N_a,N_u]);
            W_L=sum(EV1.*ambiguity_pi_u(:,1)',2)+sum(EV2.*ambiguity_pi_u(:,1)',2);
            for amb_cu=2:n_ambiguity(jj)
                W_L=min(W_L,sum(EV1.*ambiguity_pi_u(:,amb_cu)',2)+sum(EV2.*ambiguity_pi_u(:,amb_cu)',2));
            end
            EV1=wa2l.*reshape(EVnext(a1u+N_a1*(a2l-1)),[N_a,N_u]);
            EV2=wa2u.*reshape(EVnext(a1u+N_a1*(a2u-1)),[N_a,N_u]);
            W_U=sum(EV1.*ambiguity_pi_u(:,1)',2)+sum(EV2.*ambiguity_pi_u(:,1)',2);
            for amb_cu=2:n_ambiguity(jj)
                W_U=min(W_U,sum(EV1.*ambiguity_pi_u(:,amb_cu)',2)+sum(EV2.*ambiguity_pi_u(:,amb_cu)',2));
            end
            EVnext_atpolicy=wa1l.*W_L+wa1u.*W_U;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
            V(:,jj)=F_jj+beta*EVnext_atpolicy;
        elseif N_z==0 && N_e>0
            % a1l/u, wa1l/u: [N_a, N_e]; a2primeIndex/Probs: [N_a, N_e, N_u]
            a1l=a1_lower(:,:,jj); a1u=a1_upper(:,:,jj);
            wa1l=w_a1_lower(:,:,jj); wa1u=w_a1_upper(:,:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            lin_LL=a1l+N_a1*(a2l-1); lin_LU=a1l+N_a1*(a2u-1);
            lin_UL=a1u+N_a1*(a2l-1); lin_UU=a1u+N_a1*(a2u-1);
            % Per-corner mins, mirroring the raws: at each coarse a1 corner, each e-prior is
            % carried through the a2 lottery (conditional on the prior; running argmin with
            % component tracking so the u-stage arithmetic matches the raws), then the worst
            % case over the u-priors; the a1 mix interpolates the worst-case values
            for corner_c=1:2
                if corner_c==1
                    lin_lo=lin_LL; lin_up=lin_LU; % a1 lower corner
                else
                    lin_lo=lin_UL; lin_up=lin_UU; % a1 upper corner
                end
                for amb_c=1:n_ambiguity(jj)
                    EVnext=sum(V(:,:,jj+1) .* shiftdim(vfoptions.ambiguity_pi_e_J(:,jj+1,amb_c), -1), 2);
                    EVnext(isnan(EVnext))=0; % zero pi_e entries times -Inf give NaN
                    EV1=wa2l.*reshape(EVnext(lin_lo(:)),[N_a,N_e,N_u]);
                    EV2=wa2u.*reshape(EVnext(lin_up(:)),[N_a,N_e,N_u]);
                    if amb_c==1
                        Mmin=EV1+EV2; EV1sel=EV1; EV2sel=EV2;
                    else
                        Mk=EV1+EV2;
                        newmin=(Mk<Mmin);
                        Mmin(newmin)=Mk(newmin);
                        EV1sel(newmin)=EV1(newmin);
                        EV2sel(newmin)=EV2(newmin);
                    end
                end
                W_c=sum(EV1sel.*shiftdim(ambiguity_pi_u(:,1),-2),3)+sum(EV2sel.*shiftdim(ambiguity_pi_u(:,1),-2),3);
                for amb_cu=2:n_ambiguity(jj)
                    W_c=min(W_c,sum(EV1sel.*shiftdim(ambiguity_pi_u(:,amb_cu),-2),3)+sum(EV2sel.*shiftdim(ambiguity_pi_u(:,amb_cu),-2),3));
                end
                if corner_c==1
                    W_L=W_c;
                else
                    W_U=W_c;
                end
            end
            EVnext_atpolicy=wa1l.*W_L+wa1u.*W_U;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        elseif N_z>0 && N_e==0
            % a1l/u, wa1l/u: [N_a, N_z]; a2primeIndex/Probs: [N_a, N_z, N_u]
            a1l=a1_lower(:,:,jj); a1u=a1_upper(:,:,jj);
            wa1l=w_a1_lower(:,:,jj); wa1u=w_a1_upper(:,:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]);
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            % Per-corner mins, mirroring the raws: at each coarse a1 corner, each z-prior is
            % carried through the a2 lottery (conditional on the prior; running argmin with
            % component tracking so the u-stage arithmetic matches the raws), then the worst
            % case over the u-priors; the a1 mix interpolates the worst-case values
            for corner_c=1:2
                if corner_c==1
                    lin_lo=lin_LL; lin_up=lin_LU; % a1 lower corner
                else
                    lin_lo=lin_UL; lin_up=lin_UU; % a1 upper corner
                end
                for amb_c=1:n_ambiguity(jj)
                    EVnext=V(:,:,jj+1)*vfoptions.ambiguity_pi_z_J(:,:,jj,amb_c)';
                    EVnext(isnan(EVnext))=0;
                    EV1=wa2l.*reshape(EVnext(lin_lo(:)),[N_a,N_z,N_u]);
                    EV2=wa2u.*reshape(EVnext(lin_up(:)),[N_a,N_z,N_u]);
                    if amb_c==1
                        Mmin=EV1+EV2; EV1sel=EV1; EV2sel=EV2;
                    else
                        Mk=EV1+EV2;
                        newmin=(Mk<Mmin);
                        Mmin(newmin)=Mk(newmin);
                        EV1sel(newmin)=EV1(newmin);
                        EV2sel(newmin)=EV2(newmin);
                    end
                end
                W_c=sum(EV1sel.*shiftdim(ambiguity_pi_u(:,1),-2),3)+sum(EV2sel.*shiftdim(ambiguity_pi_u(:,1),-2),3);
                for amb_cu=2:n_ambiguity(jj)
                    W_c=min(W_c,sum(EV1sel.*shiftdim(ambiguity_pi_u(:,amb_cu),-2),3)+sum(EV2sel.*shiftdim(ambiguity_pi_u(:,amb_cu),-2),3));
                end
                if corner_c==1
                    W_L=W_c;
                else
                    W_U=W_c;
                end
            end
            EVnext_atpolicy=wa1l.*W_L+wa1u.*W_U;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        else
            % a1l/u, wa1l/u: [N_a, N_z*N_e] flat -> [N_a, N_z, N_e]
            % a2primeIndex/Probs: [N_a, N_z*N_e, N_u] -> [N_a, N_z, N_e, N_u]
            a1l=reshape(a1_lower(:,:,jj),[N_a,N_z,N_e]);  a1u=reshape(a1_upper(:,:,jj),[N_a,N_z,N_e]);
            wa1l=reshape(w_a1_lower(:,:,jj),[N_a,N_z,N_e]); wa1u=reshape(w_a1_upper(:,:,jj),[N_a,N_z,N_e]);
            a2l=reshape(a2primeIndex,[N_a,N_z,N_e,N_u]); a2u=a2l+1;
            wa2l=reshape(a2primeProbs,[N_a,N_z,N_e,N_u]); wa2u=1-wa2l;
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1,1]);
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            % iid-e worst case on the grid first (mirrors the AmbAverse riskyasset raws)
            EVpre=[];
            for amb_ce=1:n_ambiguity(jj)
                EVe=reshape(sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.ambiguity_pi_e_J(:,jj+1,amb_ce), -2), 3),[N_a,N_z]);
                EVe(isnan(EVe))=0;
                if amb_ce==1
                    EVpre=EVe;
                else
                    EVpre=min(EVpre,EVe);
                end
            end
            % Per-corner mins, mirroring the raws: at each coarse a1 corner, each z-prior is
            % carried through the a2 lottery (conditional on the prior; running argmin with
            % component tracking so the u-stage arithmetic matches the raws), then the worst
            % case over the u-priors; the a1 mix interpolates the worst-case values
            for corner_c=1:2
                if corner_c==1
                    lin_lo=lin_LL; lin_up=lin_LU; % a1 lower corner
                else
                    lin_lo=lin_UL; lin_up=lin_UU; % a1 upper corner
                end
                for amb_c=1:n_ambiguity(jj)
                    EVnext=EVpre*vfoptions.ambiguity_pi_z_J(:,:,jj,amb_c)';
                    EVnext(isnan(EVnext))=0;
                    EV1=wa2l.*reshape(EVnext(lin_lo(:)),[N_a,N_z,N_e,N_u]);
                    EV2=wa2u.*reshape(EVnext(lin_up(:)),[N_a,N_z,N_e,N_u]);
                    if amb_c==1
                        Mmin=EV1+EV2; EV1sel=EV1; EV2sel=EV2;
                    else
                        Mk=EV1+EV2;
                        newmin=(Mk<Mmin);
                        Mmin(newmin)=Mk(newmin);
                        EV1sel(newmin)=EV1(newmin);
                        EV2sel(newmin)=EV2(newmin);
                    end
                end
                W_c=sum(EV1sel.*shiftdim(ambiguity_pi_u(:,1),-3),4)+sum(EV2sel.*shiftdim(ambiguity_pi_u(:,1),-3),4);
                for amb_cu=2:n_ambiguity(jj)
                    W_c=min(W_c,sum(EV1sel.*shiftdim(ambiguity_pi_u(:,amb_cu),-3),4)+sum(EV2sel.*shiftdim(ambiguity_pi_u(:,amb_cu),-3),4));
                end
                if corner_c==1
                    W_L=W_c;
                else
                    W_U=W_c;
                end
            end
            EVnext_atpolicy=wa1l.*W_L+wa1u.*W_U;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
            V(:,:,:,jj)=F_jj+beta*EVnext_atpolicy;
        end
    end
end

%% Reshape V out of Kron form
if N_z==0 && N_e==0
    V=reshape(V, [n_a, N_j]);
elseif N_z==0 && N_e>0
    V=reshape(V, [n_a, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V=reshape(V, [n_a, n_z, N_j]);
else
    V=reshape(V, [n_a, n_z, vfoptions.n_e, N_j]);
end

end
