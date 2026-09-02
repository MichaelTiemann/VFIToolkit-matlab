function [V, Policy]=ValueFnIter_FHorz_GulPesendorfer(n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Gul-Pesendorfer preferences: temptation and self-control. Alongside the return fn u there is
% a temptation fn v (vfoptions.temptationFn, same input signature as the ReturnFn, its own
% parameters), and
%   V_j(a,z) = max_{d,a'} [ u + v + beta*E V_{j+1} ] - max_{d,a'} v
% Policy maximizes the tempted objective u+v+beta*EV; the second term is the most tempting
% alternative, so V nets off the self-control cost. The '-max v' term is a constant w.r.t.
% the choice given the state, so it never affects Policy, only V. The temptation fn should be
% -Inf exactly where the return fn is -Inf (the same budget/feasibility constraints).
%
% Design decisions (implemented in the raws):
% - The most-tempting term is ALWAYS a max over the full choice set. In divide-and-conquer the
%   tempted objective is only evaluated on restricted aprime windows, so the raws compute the
%   most-tempting term separately (from full-choice-set temptation matrices, built one a-slab
%   at a time) and subtract it from V after the max.
% - Under the grid interpolation layer the choice set is the fine grid, so the most-tempting
%   term is the max of v over the FINE grid, found by the same two-stage scheme as the main
%   max but around v's OWN coarse argmax (otherwise the chosen fine point could be more
%   tempting than the coarse max of v, making the self-control cost negative).
% - Divide-and-conquer assumes monotonicity (in a) of the argmax of the TEMPTED objective
%   u+v+beta*EV; with a temptation fn of the usual 'tempted by consumption' kind this holds
%   whenever it holds for u itself.

V=nan;
Policy=nan;

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

% Reject asset types this dispatcher does not handle: every asset type it does handle is
% dispatched below and returns, so an unsupported flag would otherwise be silently ignored.
if vfoptions.experienceasset>=1 || vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
    error('GulPesendorfer preferences are not implemented for the experience assets (only for the standard endogenous states)')
end
if vfoptions.riskyasset==1
    error('GulPesendorfer preferences are not implemented for riskyasset (only for the standard endogenous states)')
end
if vfoptions.residualasset==1
    error('GulPesendorfer preferences are not implemented for residualasset')
end
if vfoptions.dynasty==1
    error('GulPesendorfer preferences are not implemented for dynasty')
end
if isfield(vfoptions,'n_semiz')
    if prod(vfoptions.n_semiz)>0
        error('GulPesendorfer is not implemented for semi-exogenous states (vfoptions.n_semiz)')
    end
end

%% Some Gul-Pesendorfer specific options need to be set if they are not already declared
if ~isfield(vfoptions,'temptationFn')
    error('When using Gul-Pesendorfer preferences you must declare vfoptions.temptationFn (the temptation function)')
end

% Get the temptation function and the parameters needed to evaluate it
% (Note, this is essentially just copy-paste of handling the return fn; the experience assets,
% residualasset and semiz are all ruled out above, so no l_a/l_z adjustments for them here)
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
if N_z==0
    l_z=0;
end
if N_e==0
    l_e=0;
else
    l_e=length(vfoptions.n_e);
end
% Figure out TemptationFnParamNames from TemptationFn
temp=getAnonymousFnInputNames(vfoptions.temptationFn);
if length(temp)>(l_d+l_a+l_a+l_z+l_e) % This is largely pointless, the temptationFn is always going to have some parameters
    TemptationFnParamNames={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z,e)
else
    TemptationFnParamNames={};
end

%% Dispatch on divide-and-conquer/grid-interpolation-layer (level 2)
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
    return
elseif vfoptions.divideandconquer==1
    [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
    return
elseif vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
    return
end

%% Plain (no divide-and-conquer, no grid interpolation)
if N_d==0
    if N_e==0
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    end
else
    if N_e==0
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_noz_raw(n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_noz_e_raw(n_d, n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_e_raw(n_d, n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    end
end


%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

if N_d==0
    n_daprime=n_a;
else
    n_daprime=[n_d,n_a];
end

if N_e==0
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_daprime,n_a,N_j,vfoptions);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    end
else
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    end
end


end
