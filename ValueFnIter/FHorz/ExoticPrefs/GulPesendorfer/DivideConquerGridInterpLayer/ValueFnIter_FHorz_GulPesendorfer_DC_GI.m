function [V,Policy]=ValueFnIter_FHorz_GulPesendorfer_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with divide-and-conquer plus a grid interpolation layer on aprime. The DC
% pass runs on the tempted objective u+v+beta*EV; the most-tempting term is the max of v over
% the FINE grid (the choice set under GI), with v's coarse argmax computed exactly over the
% full aprime grid one a-slab at a time. See the comments in ValueFnIter_FHorz_GulPesendorfer
% for the design decisions.

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'level1n')
    if isscalar(n_a)
        vfoptions.level1n=floor(sqrt(n_a(1)));
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end

if ~isscalar(n_a)
    error('GulPesendorfer with vfoptions.divideandconquer/gridinterplayer and two endogenous states is not yet implemented (the plain GulPesendorfer solver does handle two standard endogenous states)')
end

%% 1 endogenous state
if N_d==0
    if N_e==0
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    end
else
    if N_e==0
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_noz_raw(n_d, n_a, N_j, d_gridvals, a_grid, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_raw(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_noz_e_raw(n_d, n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_GulPesendorfer_DC1_GI1_e_raw(n_d, n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
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
