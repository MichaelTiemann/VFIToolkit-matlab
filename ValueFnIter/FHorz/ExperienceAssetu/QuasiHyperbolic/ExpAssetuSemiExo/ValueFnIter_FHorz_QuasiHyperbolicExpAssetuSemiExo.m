function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, beta0)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience assetu
% z is exogenous state, semiz is semi-exog state

% vfoptions are already set by ValueFnIter_FHorz()
if vfoptions.parallel~=2
    error('Can only use experience assetu with parallel=2 (gpu)')
end

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience assetu you must define vfoptions.aprimeFn')
end

if isfield(vfoptions,'n_u')
    n_u=vfoptions.n_u;
else
    error('To use an experience assetu you must define vfoptions.n_u')
end
if isfield(vfoptions,'n_u')
    u_grid=gpuArray(vfoptions.u_grid);
else
    error('To use an experience assetu you must define vfoptions.u_grid')
end
if isfield(vfoptions,'pi_u')
    pi_u=gpuArray(vfoptions.pi_u);
else
    error('To use an experience assetu you must define vfoptions.pi_u')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
l_u=length(n_u);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_u)
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+1:end}}; % the first inputs will always be (d2,a2,u)
else
    aprimeFnParamNames={};
end

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
if N_d1>0
    d12_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
else
    d12_gridvals=[]; % not used
end


%% Dispatch
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    % Solve by doing Divide-and-Conquer, and then a grid interpolation layer
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExo_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExo_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, []};
    end
    return
elseif vfoptions.divideandconquer==1
    % Solve by Divide-and-Conquer
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExo_DC(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExo_DC(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, []};
    end
    return
elseif vfoptions.gridinterplayer==1
    % Solve with a grid interpolation layer
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExo_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExo_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        varargout={V1, Policy, Valt, []};
    end
    return
end


%% Plain case: no divide-and-conquer, no grid interpolation layer
if N_a1==0
    if N_e==0
        if N_d1==0
            if N_z==0
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_nod1_noa1_noz_raw(n_d2,n_d3,n_a2,n_semiz,n_u, N_j, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J,u_gridvals, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_nod1_noa1_noz_raw(n_d2,n_d3,n_a2,n_semiz,n_u, N_j, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J,u_gridvals, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            else
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_nod1_noa1_raw(n_d2,n_d3,n_a2,n_z,n_semiz,n_u, N_j, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_nod1_noa1_raw(n_d2,n_d3,n_a2,n_z,n_semiz,n_u, N_j, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            end
        else
            if N_z==0
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_noa1_noz_raw(n_d1,n_d2,n_d3,n_a2,n_semiz,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J,u_gridvals, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_noa1_noz_raw(n_d1,n_d2,n_d3,n_a2,n_semiz,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J,u_gridvals, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            else
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_noa1_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_noa1_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            end
        end
    else % N_e
        if N_d1==0
            if N_z==0
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_nod1_noa1_noz_e_raw(n_d2,n_d3,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_nod1_noa1_noz_e_raw(n_d2,n_d3,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            else
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_nod1_noa1_e_raw(n_d2,n_d3,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_nod1_noa1_e_raw(n_d2,n_d3,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            end
        else
            if N_z==0
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_noa1_noz_e_raw(n_d1,n_d2,n_d3,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_noa1_noz_e_raw(n_d1,n_d2,n_d3,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            else
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_noa1_e_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_noa1_e_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            end
        end
    end
else 
    %% With a1 (a standard endogenous state)
    if N_e==0
        if N_d1==0
            if N_z==0
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_u, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J,u_gridvals, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_u, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J,u_gridvals, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            else
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            end
        else
            if N_z==0
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J,u_gridvals, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J,u_gridvals, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            else
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            end
        end
    else % N_e
        if N_d1==0
            if N_z==0
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            else
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            end
        else
            if N_z==0
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            else
                if isNaive
                    [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoN_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuSemiExoS_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, pi_semiz_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
            end
        end
    end
end


%%
if vfoptions.outputkron==1
    V=VKron; Policy=PolicyKron; Valt=ValtKron;
    if isNaive
        varargout={V,Policy,Valt,PolicyaltKron};
    else
        varargout={V,Policy,Valt,[]};
    end
    return
end

if N_z==0
    n_bothz=n_semiz;
else
    n_bothz=[n_semiz,n_z];
end
if n_a1>0
    n_a=[n_a1,n_a2];
else
    n_a=n_a2;
end

if N_e==0
    V=reshape(VKron,[n_a,n_bothz,N_j]);
    Valt=reshape(ValtKron,[n_a,n_bothz,N_j]);
    if N_a1==0
        if N_d1==0
            Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_d3,n_a,n_bothz,N_j,vfoptions);
            if isNaive
                Policyalt=UnKronPolicyIndexes2_FHorz_z(PolicyaltKron,n_d2,n_d3,n_a,n_bothz,N_j,vfoptions);
            end
        else
            Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d1,n_d2,n_d3,n_a,n_bothz,N_j,vfoptions);
            if isNaive
                Policyalt=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron,n_d1,n_d2,n_d3,n_a,n_bothz,N_j,vfoptions);
            end
        end
    else
        if N_d1==0
            Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
            if isNaive
                Policyalt=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
            end
        else
            Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
            if isNaive
                Policyalt=UnKronPolicyIndexes4_FHorz_z(PolicyaltKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
            end
        end
    end
else
    V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    Valt=reshape(ValtKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    if N_a1==0
        if N_d1==0
            Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d2,n_d3,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
            if isNaive
                Policyalt=UnKronPolicyIndexes2_FHorz_z_e(PolicyaltKron,n_d2,n_d3,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
            end
        else
            Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d1,n_d2,n_d3,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
            if isNaive
                Policyalt=UnKronPolicyIndexes3_FHorz_z_e(PolicyaltKron,n_d1,n_d2,n_d3,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
            end
        end
    else
        if N_d1==0
            Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
            if isNaive
                Policyalt=UnKronPolicyIndexes3_FHorz_z_e(PolicyaltKron,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
            end
        else
            Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
            if isNaive
                Policyalt=UnKronPolicyIndexes4_FHorz_z_e(PolicyaltKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
            end
        end
    end
end




if isNaive
    varargout={V,Policy,Valt,Policyalt};
else
    varargout={V,Policy,Valt,[]};
end

end
