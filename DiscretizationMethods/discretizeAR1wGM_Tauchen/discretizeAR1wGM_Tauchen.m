function [z_grid,pi_z]=discretizeAR1wGM_Tauchen(mew,rho,mixprobs_i,mu_i,sigma_i,znum,Tauchen_q, tauchenoptions)
% Create states vector, z_grid, and transition matrix, pi_z, for the discrete markov process approximation
%    of AR(1) process with gaussian mixture innovations:
%       z'=mew+rho*z+e, e~F
%          where F=sum_{i=1}^nmix mixprobs_i*N(mu_i,sigma_i^2) is a gaussian mixture
%    by Tauchen method
%
% We use "nmix" to denote the number of normal distributions being mixed in the gaussian mixture innovations
%
% Inputs
%   mew            - constant term coefficient (the INTERCEPT of the AR(1), not the mean of z)
%   rho            - autocorrelation coefficient
%   mixprobs_i     - (nmix-by-1) mixture probabilities of the gaussian mixture innovations (must sum to 1)
%   mu_i           - (nmix-by-1) means of the gaussian mixture innovations
%   sigma_i        - (nmix-by-1) standard deviations of the gaussian mixture innovations
%   znum           - number of states in discretization of z (must be an odd number)
%   Tauchen_q      - (Hyperparameter) Defines max/min grid points as E(z)+-Tauchen_q*sigmaz (I suggest 2 or 3)
% Optional Inputs (tauchenoptions)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   z_grid         - column vector containing the znum states of the discrete approximation of z
%   pi_z           - transition matrix of the discrete approximation of z;
%                    pi_z(i,j) is the probability of transitioning from state i to state j
%
% Helpful info:
%   E(e)=sum_i mixprobs_i*mu_i
%   Var(e)=sum_i mixprobs_i*(mu_i^2+sigma_i^2) - E(e)^2
%   So E(z)=(mew+E(e))/(1-rho), and Var(z)=Var(e)/(1-rho^2); sigmaz=sqrt(Var(e))/sqrt(1-rho^2)
%
% Note: the grid is centred on E(z), which is the mean of z including the mean of the gaussian
%   mixture. Both discretizeAR1wGM_FarmerToda and discretizeLifeCycleAR1wGM_KFTT instead centre on
%   mew/(1-rho), which is E(z) only when the gaussian mixture is mean zero. The offset between the
%   two conventions is E(e)/(1-rho), which persistence makes large, so they are not interchangeable
%   for a mixture with a non-zero mean. Which one is intended is an open question rather than an
%   oversight - discretizeLifeCycleAR1wGM_KFTT carries a setmixturemutoenforcezeromean option,
%   which suggests the design intent was that mixtures be mean zero, in which case the two agree.
%   P7 of the DiscretizationMethodTests test bank measures the gap on a deliberately non-mean-zero
%   calibration; P3 measures it on a mean-zero one, where it is identically zero and invisible.
%%%%%%%%%%%%%%%
% Original paper (for the Tauchen method itself, which is for gaussian innovations):
% Tauchen (1986) - "Finite state Markov-chain approximations to univariate and vector autoregressions"
% The extension to gaussian mixture innovations is immediate: conditional on z, the next period
% value z' is a gaussian mixture with the same weights and standard deviations but with every
% component mean shifted by the conditional mean mew+rho*z. So the probability of landing in a bin
% is just the mixture-weighted sum of the normal cdf differences over that bin, one per component.

%%

if exist('tauchenoptions','var')==0
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    tauchenoptions.parallel=1+(gpuDeviceCount>0);
    tauchenoptions.verbose=1;
else
    %Check tauchenoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(tauchenoptions,'parallel')
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(tauchenoptions,'verbose')
        tauchenoptions.verbose=1;
    end
end

% The recommendation above is printed once per call, and the test bank that exercises these
% commands calls them hundreds of times, so it is gated. Set tauchenoptions.verbose=0 to silence it.
if tauchenoptions.verbose==1
    fprintf('COMMENT: The Tauchen method is likely inferior to the Farmer-Toda method for discretizing AR(1) processes with gaussian mixture innovations. \n')
    fprintf('         It is suggested you consider using discretizeAR1wGM_FarmerToda instead. \n')
end

%% Check inputs are correctly formatted
if size(mixprobs_i,1)<size(mixprobs_i,2)
    mixprobs_i=mixprobs_i'; % convert to column vector
end
if size(mu_i,1)<size(mu_i,2)
    mu_i=mu_i'; % convert to column vector
end
if size(sigma_i,1)<size(sigma_i,2)
    sigma_i=sigma_i'; % convert to column vector
end
nmix=length(mixprobs_i);
if length(mu_i)~=nmix || length(sigma_i)~=nmix
    error('mixprobs_i, mu_i and sigma_i must all be the same length (the number of components in the gaussian mixture)')
end
if any(mixprobs_i<0)
    error('mixture probabilities (mixprobs_i) must be non-negative')
end
if abs(sum(mixprobs_i)-1)>10^(-12)
    error('mixture probabilities (mixprobs_i) must sum to one')
end
if any(sigma_i<=0)
    error('standard deviations of the gaussian mixture components (sigma_i) must be strictly positive')
end
if abs(rho)>=1
    error('Tauchen error, autocorrelation is >=1 in absolute value. You cannot discretize an AR(1) with an autocorrelation coefficient of >=1')
end

%% Moments of the gaussian mixture innovations, and hence of z
mew_e=sum(mixprobs_i.*mu_i); % mean of the gaussian mixture
sigmasq_e=sum(mixprobs_i.*(mu_i.^2+sigma_i.^2))-mew_e^2; % variance of the gaussian mixture

if znum==1
    z_grid=(mew+mew_e)/(1-rho); %expected value of z
    pi_z=1;
    if tauchenoptions.parallel==2
        z_grid=gpuArray(z_grid);
        pi_z=gpuArray(pi_z);
    end
    return
end

if tauchenoptions.parallel==0 || tauchenoptions.parallel==1
    zstar=(mew+mew_e)/(1-rho); %expected value of z
    sigmaz=sqrt(sigmasq_e)/sqrt(1-rho^2); %stddev of z
    z_grid=zstar*ones(znum,1) + linspace(-Tauchen_q*sigmaz,Tauchen_q*sigmaz,znum)';
    omega=z_grid(2)-z_grid(1); %Note that all the points are equidistant by construction.
    upper=z_grid+omega/2;
    lower=z_grid-omega/2;

    zi=z_grid*ones(1,znum);
    upperj=ones(znum,1)*upper';
    lowerj=ones(znum,1)*lower';

    % Conditional on z, the next period value is a gaussian mixture with every component mean
    % shifted by the conditional mean, so just sum the per-component cdf differences.
    P_part1=zeros(znum,znum);
    P_part2=zeros(znum,znum);
    for i_c=1:nmix
        P_part1=P_part1+mixprobs_i(i_c)*normcdf(upperj-rho*zi-mu_i(i_c),mew,sigma_i(i_c));
        P_part2=P_part2+mixprobs_i(i_c)*normcdf(lowerj-rho*zi-mu_i(i_c),mew,sigma_i(i_c));
    end

    pi_z=P_part1-P_part2;
    pi_z(:,1)=P_part1(:,1);
    pi_z(:,znum)=1-P_part2(:,znum);

elseif tauchenoptions.parallel==2 %Parallelize on GPU
    zstar=(mew+mew_e)/(1-rho); %expected value of z
    sigmaz=sqrt(sigmasq_e)/sqrt(1-rho^2); %stddev of z
    z_grid=gpuArray(zstar*ones(znum,1) + linspace(-Tauchen_q*sigmaz,Tauchen_q*sigmaz,znum)');
    omega=z_grid(2)-z_grid(1); %Note that all the points are equidistant by construction.
    upper=z_grid+omega/2;
    lower=z_grid-omega/2;

    %Note: normcdf is not yet a supported function for use on the gpu in Matlab
    %However erf is supported, and we can easily construct our own normcdf
    %from erf (see http://en.wikipedia.org/wiki/Normal_distribution for the
    %formula for normcdf as function of erf)

    P_part1=zeros(znum,znum,'gpuArray');
    P_part2=zeros(znum,znum,'gpuArray');
    for i_c=1:nmix
        erfinput=arrayfun(@(zi,zj,rho,mew,mui,sigmai) ((zj-rho*zi)-mew-mui)/sqrt(2*sigmai^2), z_grid,upper', rho,mew,mu_i(i_c),sigma_i(i_c));
        P_part1=P_part1+mixprobs_i(i_c)*0.5*(1+erf(erfinput));

        erfinput=arrayfun(@(zi,zj,rho,mew,mui,sigmai) ((zj-rho*zi)-mew-mui)/sqrt(2*sigmai^2), z_grid,lower', rho,mew,mu_i(i_c),sigma_i(i_c));
        P_part2=P_part2+mixprobs_i(i_c)*0.5*(1+erf(erfinput));
    end

    pi_z=P_part1-P_part2;
    pi_z(:,1)=P_part1(:,1);
    pi_z(:,znum)=1-P_part2(:,znum);

end

end
