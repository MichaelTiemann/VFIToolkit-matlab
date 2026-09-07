function [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI(n_d1,n_d2,n_a,n_semiz,n_z,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with a semi-exogenous state. The tempted objective u+v+beta*EV goes through
% the standard per-d2 machinery; the most-tempting term is a max over the FULL (d1,d2,aprime)
% choice set (a per-d2 full max collected alongside each inner solve, maxed over d2, and
% subtracted from V after the outer max). See ValueFnIter_FHorz_GulPesendorfer for the design
% decisions (under grid interpolation the most-tempting term is over the FINE grid).

N_d1=prod(n_d1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if length(n_a)>2
    error('Cannot use vfoptions.divideandconquer/gridinterplayer with more than two endogenous states (you have length(n_a)>2)')
end

%% Dispatch: 1 vs 2 endogenous states
if ~isscalar(n_a)
    if ~isscalar(vfoptions.ngridinterp)
        error('vfoptions.gridinterplayer=1 with two endogenous states can only be applied to the first of the two endo states (you have length(vfoptions.ngridinterp)>1)')
    end
    if N_d1==0
        if N_e==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_nod1_e_raw(n_d2,n_a,n_z,n_semiz,  vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_noz_raw(n_d1,n_d2,n_a,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_raw(n_d1,n_d2,n_a,n_z,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_noz_e_raw(n_d1,n_d2,n_a,vfoptions.n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI2A_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        end
    end
else
    if N_d1==0
        if N_e==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_nod1_e_raw(n_d2,n_a,n_z,n_semiz,  vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_noz_raw(n_d1,n_d2,n_a,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_raw(n_d1,n_d2,n_a,n_z,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_noz_e_raw(n_d1,n_d2,n_a,vfoptions.n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_SemiExo_GI1_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
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
if isscalar(n_a)
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
else % two endogenous states
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d1==0
        if N_e==0
            V=reshape(VKron,[n_a,n_bothz,N_j]);
            Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d2,n_a1,n_a2,n_a,n_bothz,N_j,vfoptions);
        else
            V=reshape(VKron,[n_a,n_bothz, vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d2,n_a1,n_a2,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        end
    else
        if N_e==0
            V=reshape(VKron,[n_a,n_bothz,N_j]);
            Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron,n_d1,n_d2,n_a1,n_a2,n_a,n_bothz,N_j,vfoptions);
        else
            V=reshape(VKron,[n_a,n_bothz, vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron,n_d1,n_d2,n_a1,n_a2,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        end
    end
end



end
