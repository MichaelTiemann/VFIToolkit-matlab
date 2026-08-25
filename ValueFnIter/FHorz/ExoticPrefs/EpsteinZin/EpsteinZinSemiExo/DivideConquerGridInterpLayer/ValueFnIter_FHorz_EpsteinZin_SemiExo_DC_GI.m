function [V,Policy]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC_GI(n_d1,n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Epstein-Zin + SemiExo + divide-and-conquer + grid interpolation layer
% sub-dispatcher (mirrors ValueFnIter_FHorz_SemiExo_DC_GI, with the Epstein-Zin
% trailing arguments passed through to the raws). The ezc1-ezc8/sj/warmglow
% preamble is done by ValueFnIter_FHorz_EpsteinZin and just passed through.
% Does its own UnKron (the caller returns immediately).

N_d1=prod(n_d1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isscalar(n_a)
    error('Epstein-Zin with semi-exogenous shocks plus divide-and-conquer plus the grid interpolation layer only supports a single endogenous state (scalar n_a) for now')
end
if ~isfield(vfoptions,'ngridinterp')
    error('You must declare vfoptions.ngridinterp when using the grid interpolation layer')
end

if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=floor(sqrt(n_a(1)));
    if n_a(1)<5
        error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a);

%% 1 endogenous state
if N_d1==0
    if N_e==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_nod1_e_raw(n_d2,n_a,n_z,n_semiz,  vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    end
else
    if N_e==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_noz_raw(n_d1, n_d2,n_a,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_raw(n_d1, n_d2,n_a,n_z,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_noz_e_raw(n_d1,n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_GI1_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
        end
    end
end


%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

% Because of how we have N_semiz*N_z together, use the _z commands to UnKron
if N_z==0
    n_bothz=vfoptions.n_semiz;
else
    n_bothz=[vfoptions.n_semiz,n_z];
end

% First dimension of PolicyKron is (d1,d2,aprime), or if no d1, then (d2,aprime)
if N_d1==0
    if N_e==0
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
    else
        V=reshape(VKron,[n_a,n_bothz, vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    end
else
    if N_e==0
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d1,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
    else
        V=reshape(VKron,[n_a,n_bothz, vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d1,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    end
end


end
