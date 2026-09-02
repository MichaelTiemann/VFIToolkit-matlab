function [z_gridvals_J,pi_z_J,options]=ExogShockSetup_FHorz_AmbiguityAversion(n_z,z_grid,pi_z,N_j,Parameters,options)
% Ambiguity aversion: multiple priors over the exogenous shock transition probabilities.
% Design decision: ambiguity is over pi ONLY -- every prior shares the model shock grid. (A user
% who wants 'grid ambiguity' can build it by hand as a union grid with zero-padded pi's.)
%
% This subfn sorts out all the pi: each prior's pi is run through the standard
% ExogShockSetup_FHorz pipeline (so every accepted pi input shape, the age-broadcasting, and the
% timing/trim conventions apply to the priors identically -- no second implementation of those
% rules), and the grids (and the regular pi_z/pi_e) are sent back through a call of
% ExogShockSetup_FHorz to be handled as normal.
%
% Ambiguity inputs (options is vfoptions):
%   options.n_ambiguity: number of priors; scalar, or age-dependent [1,N_j] vector
%   options.ambiguity_pi_z:   [N_z,N_z,max(n_ambiguity)] (each prior age-independent), OR
%   options.ambiguity_pi_z_J: [N_z,N_z,N_j,max(n_ambiguity)] or [N_z,N_z,N_j-1,max(n_ambiguity)]
%                             (same accepted shapes as pi_z, plus the trailing prior dimension)
%   options.ambiguity_pi_e:   [N_e,max(n_ambiguity)], OR
%   options.ambiguity_pi_e_J: [N_e,N_j,max(n_ambiguity)] or [N_e,N_j+1,max(n_ambiguity)]
%                             (same accepted shapes as pi_e, plus the trailing prior dimension)
% If both the flat and _J forms are given, the flat form is used.
%
% Outputs (beyond the usual z_gridvals_J, pi_z_J, options.e_gridvals_J, options.pi_e_J):
%   options.n_ambiguity:      normalized to [1,N_j]
%   options.ambiguity_pi_z_J: [N_z,N_z,N_j-1,max(n_ambiguity)] (N_j slices when options.V_Jplus1 is used)
%   options.ambiguity_pi_e_J: [N_e,N_j,max(n_ambiguity)] (N_j+1 columns when options.V_Jplus1 is used)
%
% The regular pi_z/options.pi_e are processed as normal -- the agent distribution etc. use them.
% They should be one of the priors the agent entertains, so a warning() is thrown if they are not
% equal to any of the priors (warning, not error: a researcher may deliberately want a true
% process outside the prior set).

N_z=prod(n_z);
if ~isfield(options,'n_e')
    N_e=0;
else
    N_e=prod(options.n_e);
end

%% Validate the ambiguity inputs
if isfield(options,'riskyasset')
    riskyasset=options.riskyasset;
else
    riskyasset=0;
end
if N_z==0 && N_e==0 && riskyasset==0
    error('Cannot use Ambiguity Aversion without any shocks (what is the point?); you have n_z=0 and no e variables')
end
if isfield(options,'ExogShockFn') || isfield(options,'EiidShockFn')
    error('Cannot combine Ambiguity Aversion with ExogShockFn/EiidShockFn (which prior would their pi output be?); declare the priors via vfoptions.ambiguity_pi_z/ambiguity_pi_e instead')
end
if ~isfield(options,'n_ambiguity')
    error('When using Ambiguity Aversion you must declare vfoptions.n_ambiguity (number of multiple priors)')
end
if isscalar(options.n_ambiguity)
    options.n_ambiguity=options.n_ambiguity*ones(1,N_j);
elseif all(size(options.n_ambiguity)==[1,N_j]) || all(size(options.n_ambiguity)==[N_j,1])
    options.n_ambiguity=reshape(options.n_ambiguity,[1,N_j]);
else
    error('When using Ambiguity Aversion, vfoptions.n_ambiguity must be either a scalar or an age-dependent vector of size [1,N_j]')
end
if any(options.n_ambiguity<1)
    error('When using Ambiguity Aversion, vfoptions.n_ambiguity must be at least 1 in every period (you must declare the number of multiple priors)')
end
maxnamb=max(options.n_ambiguity);
if riskyasset==1
    % u is treated as AMBIGUITY, not risk: the agent does not know the risky return distribution,
    % so the multiple priors over pi_u are mandatory. The regular pi_u is only the true process
    % (used for the agent distribution etc.), exactly parallel to pi_z/pi_e.
    if ~isfield(options,'ambiguity_pi_u')
        error('When using Ambiguity Aversion with riskyasset you must declare vfoptions.ambiguity_pi_u (the multiple priors over the risky return distribution; u is ambiguity, not risk)')
    end
    N_u=prod(options.n_u);
    if size(options.ambiguity_pi_u,1)~=N_u || size(options.ambiguity_pi_u,2)~=maxnamb || ~ismatrix(options.ambiguity_pi_u)
        error('vfoptions.ambiguity_pi_u must be of size [N_u,max(n_ambiguity)] (one age-independent pi_u per prior); got [%s]',num2str(size(options.ambiguity_pi_u)))
    end
    options.ambiguity_pi_u=gpuArray(options.ambiguity_pi_u);
    % Warn if the regular pi_u is not one of the u-priors (it is what the agent distribution uses)
    ufound=0;
    for amb_c=1:maxnamb
        if isequal(gather(options.pi_u(:)),gather(options.ambiguity_pi_u(:,amb_c)))
            ufound=1;
        end
    end
    if ufound==0
        warning('AmbiguityAversion: the regular pi_u input (which the agent distribution, etc., will use) is not equal to any of the ambiguity_pi_u priors')
    end
end
if N_z>0
    if ~isfield(options,'ambiguity_pi_z') && ~isfield(options,'ambiguity_pi_z_J')
        error('When using Ambiguity Aversion with a z variable you must declare vfoptions.ambiguity_pi_z or vfoptions.ambiguity_pi_z_J (the multiple priors)')
    end
    if isfield(options,'ambiguity_pi_z')
        if size(options.ambiguity_pi_z,1)~=N_z || size(options.ambiguity_pi_z,2)~=N_z || size(options.ambiguity_pi_z,3)~=maxnamb || ndims(options.ambiguity_pi_z)>3
            error('vfoptions.ambiguity_pi_z must be of size [N_z,N_z,max(n_ambiguity)] (one age-independent pi per prior); got [%s]',num2str(size(options.ambiguity_pi_z)))
        end
    else
        if size(options.ambiguity_pi_z_J,1)~=N_z || size(options.ambiguity_pi_z_J,2)~=N_z || size(options.ambiguity_pi_z_J,4)~=maxnamb || ndims(options.ambiguity_pi_z_J)>4
            error('vfoptions.ambiguity_pi_z_J must be of size [N_z,N_z,N_j,max(n_ambiguity)] (or with N_j-1 slices; one age-dependent pi per prior); got [%s]',num2str(size(options.ambiguity_pi_z_J)))
        end
    end
end
if N_e>0
    if ~isfield(options,'ambiguity_pi_e') && ~isfield(options,'ambiguity_pi_e_J')
        error('When using Ambiguity Aversion with an e variable you must declare vfoptions.ambiguity_pi_e or vfoptions.ambiguity_pi_e_J (the multiple priors)')
    end
    if isfield(options,'ambiguity_pi_e')
        if size(options.ambiguity_pi_e,1)~=N_e || size(options.ambiguity_pi_e,2)~=maxnamb || ndims(options.ambiguity_pi_e)>2
            error('vfoptions.ambiguity_pi_e must be of size [N_e,max(n_ambiguity)] (one age-independent pi per prior); got [%s]',num2str(size(options.ambiguity_pi_e)))
        end
    else
        if size(options.ambiguity_pi_e_J,1)~=N_e || size(options.ambiguity_pi_e_J,3)~=maxnamb || ndims(options.ambiguity_pi_e_J)>3
            error('vfoptions.ambiguity_pi_e_J must be of size [N_e,N_j,max(n_ambiguity)] (or with N_j+1 columns when using V_Jplus1; one age-dependent pi per prior); got [%s]',num2str(size(options.ambiguity_pi_e_J)))
        end
    end
end

%% The grids (and the regular pi_z/pi_e) are handled as normal
% The recursive call must not re-enter the ambiguity branch, hence exoticpreferences is blanked
optionstemp=options;
optionstemp.exoticpreferences='None';
[z_gridvals_J,pi_z_J,optionstemp]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,optionstemp,3);
options.e_gridvals_J=optionstemp.e_gridvals_J;
options.pi_e_J=optionstemp.pi_e_J;

%% Each prior's pi goes through the same pipeline (one gridpiboth=2 call per prior: pi only)
% Note: the gridpiboth=2 path gathers its pi outputs (the agent distribution lives on cpu), so
% the stacked priors are moved back to gpu here (the value fn wants them there).
ambiguity_pi_z_J=[]; % [N_z,N_z,N_j-1 (or N_j),maxnamb] once filled
ambiguity_pi_e_J=[]; % [N_e,N_j (or N_j+1),maxnamb] once filled
for amb_c=1:maxnamb
    optionstemp=options;
    optionstemp.exoticpreferences='None';
    % This prior's pi_z (slice of the flat or _J form)
    if N_z>0
        if isfield(options,'ambiguity_pi_z')
            pi_z_amb_c=options.ambiguity_pi_z(:,:,amb_c);
        else
            pi_z_amb_c=options.ambiguity_pi_z_J(:,:,:,amb_c);
        end
    else
        pi_z_amb_c=pi_z; % no z variable; the z part of the call is inert
    end
    % This prior's pi_e (slice of the flat or _J form)
    if N_e>0
        if isfield(options,'ambiguity_pi_e')
            optionstemp.pi_e=options.ambiguity_pi_e(:,amb_c);
        else
            optionstemp.pi_e=options.ambiguity_pi_e_J(:,:,amb_c);
        end
    end
    [~,pi_z_J_amb_c,optionstemp]=ExogShockSetup_FHorz(n_z,z_grid,pi_z_amb_c,N_j,Parameters,optionstemp,2);
    if N_z>0
        if amb_c==1
            ambiguity_pi_z_J=zeros(N_z,N_z,size(pi_z_J_amb_c,3),maxnamb,'gpuArray');
        end
        ambiguity_pi_z_J(:,:,:,amb_c)=gpuArray(pi_z_J_amb_c);
    end
    if N_e>0
        if amb_c==1
            ambiguity_pi_e_J=zeros(N_e,size(optionstemp.pi_e_J,2),maxnamb,'gpuArray');
        end
        ambiguity_pi_e_J(:,:,amb_c)=gpuArray(optionstemp.pi_e_J);
    end
end
options.ambiguity_pi_z_J=ambiguity_pi_z_J;
options.ambiguity_pi_e_J=ambiguity_pi_e_J;

%% Warn if the regular pi_z/pi_e is not one of the priors
% The regular pi_z/pi_e are what the agent distribution etc. will use, so they should be one of
% the priors the agent entertains.
if N_z>0
    zfound=0;
    for amb_c=1:maxnamb
        if isequal(gather(pi_z_J),gather(ambiguity_pi_z_J(:,:,1:size(pi_z_J,3),amb_c)))
            zfound=1;
        end
    end
    if zfound==0
        warning('AmbiguityAversion: the regular pi_z input (which the agent distribution, etc., will use) is not equal to any of the ambiguity_pi_z priors')
    end
end
if N_e>0
    efound=0;
    for amb_c=1:maxnamb
        if isequal(gather(options.pi_e_J),gather(ambiguity_pi_e_J(:,1:size(options.pi_e_J,2),amb_c)))
            efound=1;
        end
    end
    if efound==0
        warning('AmbiguityAversion: the regular pi_e input (which the agent distribution, etc., will use) is not equal to any of the ambiguity_pi_e priors')
    end
end

end
