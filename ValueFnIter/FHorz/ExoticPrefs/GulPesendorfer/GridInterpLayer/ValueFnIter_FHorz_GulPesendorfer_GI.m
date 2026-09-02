function [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with a grid interpolation layer on aprime. The choice set is the fine grid,
% so the most-tempting term is the max of v over the FINE grid, found by the same two-stage
% scheme as the main max but around v's OWN coarse argmax. See the comments in
% ValueFnIter_FHorz_GulPesendorfer for the design decisions.

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isscalar(n_a)
    error('GulPesendorfer with vfoptions.divideandconquer/gridinterplayer and two endogenous states is not yet implemented (the plain GulPesendorfer solver does handle two standard endogenous states)')
end

%% 1 endogenous state
if N_d==0
    if N_e==0
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_GI1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_GI1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_GI1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    end
else
    if N_e==0
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_GI1_noz_raw(n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_GI1_noz_e_raw(n_d, n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_GI1_e_raw(n_d, n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
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
    if N_e==0
        if N_z==0
            Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_a,n_a,N_j,vfoptions);
        else
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,n_a,n_z,N_j,vfoptions);
        end
    else
        if N_z==0
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
        else
            Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_a,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
        end
    end
else
    if N_e==0
        if N_z==0
            Policy=UnKronPolicyIndexes2_FHorz_noz(PolicyKron,n_d,n_a,n_a,N_j,vfoptions);
        else
            Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,n_a,n_z,N_j,vfoptions);
        end
    else
        if N_z==0
            Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
        else
            Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d,n_a,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
        end
    end
end

if N_e==0
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
    end
else
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
    end
end


end
