function vfoptions=SetupNonStandardEndoStates_FHorz(n_d,n_a,d_grid,a_grid,vfoptions)
% Splits n_d and n_a (and their grids) into the standard states and the non-standard endogenous
% state, for whichever non-standard endogenous state is in use. At most one is ever in use.
%
% Handles the six experience assets (experienceasset, experienceassetu, experienceassete,
% experienceassetz, experienceassetze, experienceassetsemiz), riskyasset, and residualasset.
% If none of them is in use this does nothing.
%
% The results are stored in vfoptions (n_d1, n_d2, n_d3, n_a1, n_a2, n_r and the matching
% d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, r_grid); the caller unpacks whichever it needs at
% the point of use. n_d3/d3_grid are only set when there is a semi-exogenous state.
%
% Sibling of SetupNonStandardEndoStates_FHorz_TPath, which does the same job for transition paths.

%% Experience assets
% experienceasset: aprime(d,a)
% experienceassetu: aprime(d,a,u)
% experienceassetz: aprime(d,a,z)
% experienceassete: aprime(d,a,e)
% experienceassetze: aprime(d,a,z,e)
if vfoptions.experienceasset>=1 || vfoptions.experienceassetu>=1 || vfoptions.experienceassetz>=1 || vfoptions.experienceassete>=1 || vfoptions.experienceassetze>=1 || vfoptions.experienceassetsemiz>=1
    % It is simply assumed that the experience asset is the last asset, and that the decision that influences it is the last decision.
    % When using both semiexo and experience asset, the last decision variable influences semi-exo and the second last decision variable influences the experience asset

    if vfoptions.experienceasset>=1
        if ~isfield(vfoptions,'l_dexperienceasset')
            vfoptions.l_dexperienceasset=1; % by default, only one decision variable influences the experienceasset
        end
    elseif vfoptions.experienceassetu>=1
        if ~isfield(vfoptions,'l_dexperienceassetu')
            vfoptions.l_dexperienceassetu=1; % by default, only one decision variable influences the experienceassetu
        end
    elseif vfoptions.experienceassete>=1
        if ~isfield(vfoptions,'l_dexperienceassete')
            vfoptions.l_dexperienceassete=1; % by default, only one decision variable influences the experienceassete
        end
    elseif vfoptions.experienceassetz>=1
        if ~isfield(vfoptions,'l_dexperienceassetz')
            vfoptions.l_dexperienceassetz=1; % by default, only one decision variable influences the experienceassetz
        end
    elseif vfoptions.experienceassetze>=1
        if ~isfield(vfoptions,'l_dexperienceassetze')
            vfoptions.l_dexperienceassetze=1; % by default, only one decision variable influences the experienceassetze
        end
    elseif vfoptions.experienceassetsemiz>=1
        if ~isfield(vfoptions,'l_dexperienceassetsemiz')
            vfoptions.l_dexperienceassetsemiz=1; % by default, only one decision variable influences the experienceassetsemiz
        end
    end
    
    if vfoptions.experienceasset>=1
        vfoptions.l_d2=vfoptions.l_dexperienceasset;
        vfoptions.l_a2=vfoptions.experienceasset;
    elseif vfoptions.experienceassetu>=1
        vfoptions.l_d2=vfoptions.l_dexperienceassetu;
        vfoptions.l_a2=vfoptions.experienceassetu;
    elseif vfoptions.experienceassete>=1
        vfoptions.l_d2=vfoptions.l_dexperienceassete;
        vfoptions.l_a2=vfoptions.experienceassete;
    elseif vfoptions.experienceassetz>=1
        vfoptions.l_d2=vfoptions.l_dexperienceassetz;
        vfoptions.l_a2=vfoptions.experienceassetz;
    elseif vfoptions.experienceassetze>=1
        vfoptions.l_d2=vfoptions.l_dexperienceassetze;
        vfoptions.l_a2=vfoptions.experienceassetze;
    elseif vfoptions.experienceassetsemiz>=1
        vfoptions.l_d2=vfoptions.l_dexperienceassetsemiz;
        vfoptions.l_a2=vfoptions.experienceassetsemiz;
    end

    if prod(vfoptions.n_semiz)>0
        if ~isfield(vfoptions,'l_dsemiz')
            vfoptions.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
        end

        % Split decision variables (other, semiexo, experienceasset)
        if length(n_d)>(vfoptions.l_d2+vfoptions.l_dsemiz)
            n_d1=n_d(1:end-vfoptions.l_d2-vfoptions.l_dsemiz);
        else
            n_d1=0;
        end
        n_d2=n_d(end-vfoptions.l_d2-vfoptions.l_dsemiz+1:end-vfoptions.l_dsemiz); % n_d2 is the decision variable that influences the experience asset
        n_d3=n_d(end-vfoptions.l_dsemiz+1:end); % n_d3 is the decision variable that influences the transition probabilities of the semi-exogenous state
        d1_grid=d_grid(1:sum(n_d1));
        d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
        d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:end);
        % Split endogenous assets into the standard ones and the experience asset
        if length(n_a)<=vfoptions.l_a2
            n_a1=0;
        else
            n_a1=n_a(1:end-vfoptions.l_a2);
        end
        n_a2=n_a(end-vfoptions.l_a2+1:end); % last l_a2 (=vfoptions.experienceasset) dims are the experience asset
        a1_grid=a_grid(1:sum(n_a1));
        a2_grid=a_grid(sum(n_a1)+1:end);

    else % no semiz
        % Split decision variables into the standard ones and the one relevant to the experience asset
        if length(n_d)>vfoptions.l_d2
            n_d1=n_d(1:end-vfoptions.l_d2);
        else
            n_d1=0;
        end
        n_d2=n_d(end-vfoptions.l_d2+1:end); % n_d2 is the decision variable that influences next period vale of the experience asset
        d1_grid=d_grid(1:sum(n_d1));
        d2_grid=d_grid(sum(n_d1)+1:end);
        % Split endogenous assets into the standard ones and the experience asset
        if length(n_a)<=vfoptions.l_a2
            n_a1=0;
        else
            n_a1=n_a(1:end-vfoptions.l_a2);
        end
        n_a2=n_a(end-vfoptions.l_a2+1:end); % last l_a2 (=vfoptions.experienceasset) dims are the experience asset
        a1_grid=a_grid(1:sum(n_a1));
        a2_grid=a_grid(sum(n_a1)+1:end);
    end
    % Store the split so the caller can unpack it at the point of use
    vfoptions.n_d1=n_d1;
    vfoptions.n_d2=n_d2;
    vfoptions.n_a1=n_a1;
    vfoptions.n_a2=n_a2;
    vfoptions.d1_grid=d1_grid;
    vfoptions.d2_grid=d2_grid;
    vfoptions.a1_grid=a1_grid;
    vfoptions.a2_grid=a2_grid;
    if prod(vfoptions.n_semiz)>0
        % n_d3/d3_grid only exist when there is a semi-exogenous state
        vfoptions.n_d3=n_d3;
        vfoptions.d3_grid=d3_grid;
    end
end

%% Risky asset
if vfoptions.riskyasset==1
    % It is simply assumed that the risky asset is the last asset, and that all decisions influence it.

    % Split endogenous assets into the standard ones and the risky asset
    if isscalar(n_a)
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_a2=n_a(end); % n_a2 is the risky asset
    a1_grid=a_grid(1:sum(n_a1));
    a2_grid=a_grid(sum(n_a1)+1:end);

    % Check that aprimeFn is inputted
    if ~isfield(vfoptions,'aprimeFn')
        error('You have vfoptions.riskyasset=1, but have not setup vfoptions.aprimeFn')
    end
    % Check that the u shocks are inputted
    if ~isfield(vfoptions,'n_u')
        error('You have vfoptions.riskyasset=1, but have not setup vfoptions.n_u')
    end
    if ~isfield(vfoptions,'u_grid')
        error('You have vfoptions.riskyasset=1, but have not setup vfoptions.u_grid')
    end
    if ~isfield(vfoptions,'pi_u') % && ~isfield(vfoptions,'pi_u_J')
        error('You have vfoptions.riskyasset=1, but have not setup vfoptions.pi_u')
    end
    if ~isfield(vfoptions,'refine_d')
        warning('Using vfoptions.riskyasset=1 without setting vfoptions.refine_d is outdated behaviour, it is strongly recommended you set vfoptions.refine_d')
    end
    % Store the split so the caller can unpack it at the point of use
    vfoptions.n_a1=n_a1;
    vfoptions.n_a2=n_a2;
    vfoptions.a1_grid=a1_grid;
    vfoptions.a2_grid=a2_grid;
end

%% Residual asset
% (the divideandconquer/gridinterplayer checks are solver-option validation, and stay with the dispatch)
if vfoptions.residualasset==1
    % Split endogenous assets into the standard ones and the residual asset
    if isscalar(n_a)
        n_a1=0;
    else
        n_a1=n_a(1:end-1);
    end
    n_r=n_a(end); % n_a2 is the residual asset
    a1_grid=a_grid(1:sum(n_a1));
    r_grid=a_grid(sum(n_a1)+1:end);
    % Store the split so the caller can unpack it at the point of use
    vfoptions.n_a1=n_a1;
    vfoptions.n_r=n_r;
    vfoptions.a1_grid=a1_grid;
    vfoptions.r_grid=r_grid;
end

end
