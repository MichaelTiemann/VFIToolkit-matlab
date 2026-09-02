function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetze(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0)
% Quasi-hyperbolic discounting with an experienceassetze state (z+e dependent aprimeFn).
% e is structural (always required for ExpAssetze).
% Mirrors ValueFnIter_FHorz_QuasiHyperbolicExpAssetz dispatcher.

%% Semi-exogenous state: hand off to the SemiExo variant
if prod(vfoptions.n_semiz)>0
    [V,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExo(n_d1,n_d2,vfoptions.n_d3,n_a1,n_a2,n_z,vfoptions.n_semiz, N_j, d1_grid, d2_grid, vfoptions.d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
    varargout={V,Policy,Valt,Policyalt};
    return
end

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

l_d2=length(n_d2);
l_a2=length(n_a2);
l_z=length(n_z);
l_e=length(vfoptions.n_e);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z+l_e)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+l_e+1:end}};
else
    aprimeFnParamNames={};
end

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_e==0
    error('experienceassetze requires n_e>0 (e is structural for z+e aprimeFn)')
end

if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
if N_d1>0
    d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
else
    d_gridvals=[];
end

e_gridvals_J=vfoptions.e_gridvals_J;
pi_e_J=vfoptions.pi_e_J;

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% Dispatch to the method sub-dispatchers (each UnKrons its own results)
% Each sub-dispatcher owns both the 1A (scalar n_a1) and the 2A (multi-dim n_a1) tier.
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetze_DC_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
elseif vfoptions.divideandconquer==1
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetze_DC(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
elseif vfoptions.gridinterplayer==1
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetze_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
end


%% Baseline (no DC, no GI)

if isNaive
    if N_a1==0
        if N_d1==0
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_nod1_noa1_e_raw(n_d2,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        else
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_noa1_e_raw(n_d1,n_d2,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        end
    elseif N_d1==0
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_nod1_e_raw(n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    else
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    end
else
    if N_a1==0
        if N_d1==0
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_nod1_noa1_e_raw(n_d2,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_noa1_e_raw(n_d1,n_d2,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        end
    elseif N_d1==0
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_nod1_e_raw(n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    else
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    end
end

%% Unkron (baseline)
if N_a1==0 % noa1: the experience asset a2 is the only endogenous state
    n_a=n_a2;
else
    n_a=[n_a1,n_a2];
end
if N_d1==0
    n_d=n_d2;
else
    n_d=[n_d1,n_d2];
end
if vfoptions.outputkron==1
    V1=V1Kron; Policy=PolicyKron; Valt=ValtKron;
    if isNaive, Policyalt=PolicyaltKron; end
else
    if prod(n_d)==0
        n_daprime=n_a;
    elseif N_a1==0
        n_daprime=n_d; % noa1: the joint index has no a1prime channel (a2prime comes from aprimeFn)
    else
        n_daprime=[n_d,n_a1];
    end
    V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
    Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
    if isNaive
        Policyalt=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    end
end
if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt, []};
end

end
