function [V, Policy]=ValueFnIter_FHorz_AmbiguityAversion(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Ambiguity Aversion: multiple priors over the exogenous shock transition probabilities, and the
% continuation EV is the worst case over the priors (maxmin preferences).
% See appendix to the 'Intro to Life-Cycle models' for an explanation.
%
% Design decision: ambiguity is over pi ONLY -- every prior shares the model shock grid. (A user
% who wants 'grid ambiguity' can build it by hand as a union grid with zero-padded pi's.)
% For the grid interpolation layer, the interpolation is done conditional on the prior (in the
% same way it is conditional on d, z and e): each prior's EV is interpolated over aprime, and the
% worst case (min over priors) is taken afterwards. On grid points the two orders coincide, so
% level 1 of GI, and all of divide-and-conquer (where EV is only ever read at grid points), just
% use the pointwise worst-case EV.
%
% The multiple priors were validated and put into their _J forms by
% ExogShockSetup_FHorz_AmbiguityAversion (called from ExogShockSetup_FHorz), which also
% normalized vfoptions.n_ambiguity to [1,N_j]. The regular pi_z_J input is unused here (the
% priors replace it); it is what the agent distribution etc. will use, and a warning was thrown
% during setup if it is not one of the priors.

N_d=prod(n_d);
% N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

n_ambiguity=vfoptions.n_ambiguity; % [1,N_j], from ExogShockSetup_FHorz_AmbiguityAversion

if isfield(vfoptions,'n_semiz')
    if prod(vfoptions.n_semiz)>0
        error('AmbiguityAversion is not implemented for semi-exogenous states (vfoptions.n_semiz)')
    end
end

%% Risky asset routes to its own subfn (level 3), mirroring how EpsteinZin does it
% (u is treated as ambiguity, not risk: the priors over pi_u are mandatory and were validated by
% ExogShockSetup_FHorz_AmbiguityAversion, as were any priors over pi_z/pi_e. Note that a
% riskyasset model with no z and no e is a perfectly sensible ambiguity model -- the ambiguity
% is over the risky return distribution -- so the no-shocks error below does not apply to it.)
if vfoptions.riskyasset==1
    [V, Policy]=ValueFnIter_FHorz_AmbAverse_RiskyAsset(n_ambiguity, n_d,vfoptions.n_a1,vfoptions.n_a2,n_z,vfoptions.n_u,N_j,d_grid,vfoptions.a1_grid, vfoptions.a2_grid, z_gridvals_J, vfoptions.u_grid, vfoptions.ambiguity_pi_z_J, vfoptions.ambiguity_pi_u, ReturnFn, vfoptions.aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
end

if N_z==0 && N_e==0
    error('Cannot use Ambiguity Aversion without any shocks (what is the point?); you have n_z=0 and no e variables')
end

% Reject asset types this dispatcher does not handle: every asset type it does handle is
% dispatched below and returns, so an unsupported flag would otherwise be silently ignored.
if vfoptions.experienceasset>=1 || vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
    error('AmbiguityAversion preferences are not implemented for the experience assets (only for the standard endogenous states)')
end
if vfoptions.residualasset==1
    error('AmbiguityAversion preferences are not implemented for residualasset')
end
if vfoptions.dynasty==1
    error('AmbiguityAversion preferences are not implemented for dynasty')
end

% Standard endogenous states from here on
if size(d_grid,2)==1
    d_gridvals=CreateGridvals(n_d,d_grid,1);
else % already d_gridvals (or empty when N_d==0)
    d_gridvals=d_grid;
end

%% Dispatch on divide-and-conquer/grid-interpolation-layer (level 2)
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_AmbAverse_DC_GI(n_ambiguity, n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
elseif vfoptions.divideandconquer==1
    [V,Policy]=ValueFnIter_FHorz_AmbAverse_DC(n_ambiguity, n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
elseif vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_AmbAverse_GI(n_ambiguity, n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return
end

%% Plain (no divide-and-conquer, no grid interpolation)
if N_d==0
    if N_e==0
        [VKron,PolicyKron]=ValueFnIter_FHorz_AmbAverse_nod_raw(n_ambiguity, n_a, n_z, N_j, a_grid, z_gridvals_J, vfoptions.ambiguity_pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_AmbAverse_nod_noz_e_raw(n_ambiguity, n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_AmbAverse_nod_e_raw(n_ambiguity, n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, vfoptions.ambiguity_pi_z_J, vfoptions.ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else
    if N_e==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_AmbAverse_raw(n_ambiguity, n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.ambiguity_pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_AmbAverse_noz_e_raw(n_ambiguity, n_d, n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_AmbAverse_e_raw(n_ambiguity, n_d, n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, vfoptions.ambiguity_pi_z_J, vfoptions.ambiguity_pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
