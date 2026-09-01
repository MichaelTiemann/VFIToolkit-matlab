function varargout=ValueFnIter_FHorz_QuasiHyperbolic(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic preferences.
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j=u_t+beta_0*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.
%
% Outputs are returned via varargout. Caller may request:
%   Naive:         [V, Policy] or [V, Policy, Valt] or [V, Policy, Valt, Policyalt]
%                  V       = Vtilde     (QH-discounted, used for the QH-optimal choice)
%                  Policy  = QH-optimal choice (argmax of Vtilde)
%                  Valt    = V_std      (std-discounted continuation, computed at Policyalt)
%                  Policyalt = std-optimal choice (argmax of V_std). Naive needs Policy AND
%                  Policyalt to reconstruct V/Valt from policy alone.
%   Sophisticated: [V, Policy] or [V, Policy, Valt]
%                  V       = Vhat       (QH-discounted from current self's perspective)
%                  Policy  = equilibrium choice
%                  Valt    = Vunderbar  (realised continuation under future selves' own QH choices)
%                  Policy alone suffices to reconstruct V and Valt.
%
% DiscountFactorParamNames is the standard discount factor beta
% beta0 (the additional discount factor) is received as an input; the dispatcher reads it once from vfoptions.QHadditionaldiscount

N_d=prod(n_d);
% N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'quasi_hyperbolic')
    error('You are using quasi-hyperbolic discounting (vfoptions.exoticpreferences) but you have not specified vfoptions.quasi_hyperbolic, which must be either Naive or Sophisticated')
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    % Check that one of the possible options have been used. If not then error.
    error('vfoptions.quasi_hyperbolic must be either Naive or Sophisticated (check spelling and capital letter) \n')
end

if ~isfield(vfoptions,'QHadditionaldiscount')
    error('You must declare vfoptions.QHadditionaldiscount when using quasi-hyperbolic discouting (you have vfoptions.exoticpreferences set to QuasiHyperbolic)')
elseif ~ischar(vfoptions.QHadditionaldiscount)
    error('vfoptions.QHadditionaldiscount must be the name of the additional discount parameter, given as a character vector such as ''beta0''')
end

% Reject asset types this dispatcher does not handle: every asset type it does handle is
% dispatched below and returns, so an unsupported flag would otherwise be silently ignored.
if vfoptions.riskyasset==1
    error('Quasi-hyperbolic discounting is not implemented for riskyasset')
end
if vfoptions.residualasset==1
    error('Quasi-hyperbolic discounting is not implemented for residualasset')
end
if vfoptions.dynasty==1
    error('Quasi-hyperbolic discounting is not implemented for dynasty')
end

% Read the additional discount factor once here, and pass the value (not the parameter name) down to the raws.
beta0=Parameters.(vfoptions.QHadditionaldiscount);
if ~isscalar(beta0)
    error('The quasi-hyperbolic additional discount factor (the parameter named by vfoptions.QHadditionaldiscount) must be a scalar; it cannot depend on age')
end

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% Dispatch on the non-standard endogenous state, if there is one (level 3).
% The splits (n_d1/n_d2/n_a1/n_a2 and their grids) were computed once by
% SetupNonStandardEndoStates_FHorz and are unpacked from vfoptions here.
if vfoptions.experienceasset>=1
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsset(vfoptions.n_d1,vfoptions.n_d2,vfoptions.n_a1,vfoptions.n_a2,n_z, N_j, vfoptions.d1_grid , vfoptions.d2_grid, vfoptions.a1_grid, vfoptions.a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
elseif vfoptions.experienceassetu>=1
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetu(vfoptions.n_d1,vfoptions.n_d2,vfoptions.n_a1,vfoptions.n_a2,n_z, N_j, vfoptions.d1_grid , vfoptions.d2_grid, vfoptions.a1_grid, vfoptions.a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
elseif vfoptions.experienceassete>=1
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssete(vfoptions.n_d1,vfoptions.n_d2,vfoptions.n_a1,vfoptions.n_a2,n_z, N_j, vfoptions.d1_grid, vfoptions.d2_grid, vfoptions.a1_grid, vfoptions.a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
elseif vfoptions.experienceassetz>=1
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetz(vfoptions.n_d1,vfoptions.n_d2,vfoptions.n_a1,vfoptions.n_a2,n_z, N_j, vfoptions.d1_grid, vfoptions.d2_grid, vfoptions.a1_grid, vfoptions.a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
elseif vfoptions.experienceassetze>=1
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetze(vfoptions.n_d1,vfoptions.n_d2,vfoptions.n_a1,vfoptions.n_a2,n_z, N_j, vfoptions.d1_grid, vfoptions.d2_grid, vfoptions.a1_grid, vfoptions.a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
elseif vfoptions.experienceassetsemiz>=1
    % semiz is always present for this family (it drives the experience asset)
    if prod(vfoptions.n_semiz)==0
        error('When using experienceassetsemiz you must set vfoptions.n_semiz (the semi-exogenous state drives the experience asset)')
    end
    [V1,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetsemiz(vfoptions.n_d1,vfoptions.n_d2,vfoptions.n_d3,vfoptions.n_a1,vfoptions.n_a2,n_z,vfoptions.n_semiz, N_j, vfoptions.d1_grid , vfoptions.d2_grid, vfoptions.d3_grid, vfoptions.a1_grid, vfoptions.a2_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
    varargout={V1, Policy, Valt, Policyalt};
    return
end

%% Standard endogenous states from here on
%%
if prod(vfoptions.n_semiz)>0
    % Solve with semi-exogenous state
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo(n_d, n_a, n_z, vfoptions.n_semiz, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo(n_d, n_a, n_z, vfoptions.n_semiz, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.semiz_gridvals_J, pi_z_J, vfoptions.pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt,[]};
    end
    return
end

d_gridvals=CreateGridvals(n_d,d_grid,1);

if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    % Solve by doing Divide-and-Conquer, and then a grid interpolation layer
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolic_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolic_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt,[]};
    end
    return
elseif vfoptions.divideandconquer==1
    % Solve using Divide-and-Conquer algorithm
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolic_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolic_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt,[]};
    end
    return
elseif vfoptions.gridinterplayer==1
    % Solve using grid interpolation layer
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolic_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolic_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt,[]};
    end
    return
end


%%
if isNaive % Output: [V=Vtilde, Policy, Valt=V_std, Policyalt]
    if N_e==0
        if N_z==0
            if N_d==0
                [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            else
                [VtildeKron, PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            end
        else
            if N_d==0
                [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            else
                [VtildeKron, PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            end
        end
    else
        if N_z==0
            if N_d==0
                [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            else
                [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            end
        else
            if N_d==0
                [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            else
                [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            end
        end
    end
else % Sophisticated. Output: [V=Vhat, Policy, Valt=Vunderbar]
    if N_e==0
        if N_z==0
            if N_d==0
                [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            else
                [VhatKron, PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            end
        else
            if N_d==0
                [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            else
                [VhatKron, PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            end
        end
    else
        if N_z==0
            if N_d==0
                [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            else
                [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            end
        else
            if N_d==0
                [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            else
                [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0);
            end
        end
    end
end


%% Bridge semantic Kron names back to generic V1Kron/ValtKron for downstream reshape
if isNaive
    V1Kron=VtildeKron;
else
    V1Kron=VhatKron;
    ValtKron=VunderbarKron;
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V1=V1Kron;
    Policy=PolicyKron;
    Valt=ValtKron;
    if isNaive
        Policyalt=PolicyaltKron;
    end
else
    if N_d==0
        n_daprime=n_a;
    else
        n_daprime=[n_d,n_a];
    end
    if N_e==0
        if N_z==0
            V1=reshape(V1Kron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_daprime,n_a,N_j,vfoptions);
            Valt=reshape(ValtKron,[n_a,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_noz(PolicyaltKron,n_daprime,n_a,N_j,vfoptions);
            end
        else
            V1=reshape(V1Kron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_daprime,n_a,n_z,N_j,vfoptions);
            end
        end
    else
        if N_z==0
            V1=reshape(V1Kron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
            Valt=reshape(ValtKron,[n_a,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);
            end
        else
            V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
            end
        end
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt,[]};
end


end
