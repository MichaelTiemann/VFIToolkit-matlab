function [z_grid,pi_z] = discretizeAR1_TauchenHussey(mew,rho,sigma,znum,tauchenhusseyoptions)
% Create states vector, z_grid, and transition matrix, P, for the discrete markov process approximation
%    of AR(1) process z'=mew+rho*z+e, e~N(0,sigma^2), by Tauchen-Hussey method
%
% Input:
%   N         scalar, number of nodes for Z
%   mew       scalar, unconditional mean of process
%   rho       scalar
%   sigma     scalar, std. dev. of epsilons
% Optional inputs (tauchenhusseyoptions)
%   baseSigma scalar, std. dev. used to calculate Gaussian quadrature weights and nodes,
%                 i.e. to build the grid. I recommend that you use baseSigma = w*sigma +
%                 (1-w)*sigmaZ where sigmaZ = sigma/sqrt(1-rho^2), and w = 0.5 + rho/4.
%                 This is the default. It is the third variant of Floden (2008), pg 517:
%                 "sigmahat = w*sigma_epsilon + (1-w)*sigma_z where w = 1/2 + rho/4".
%                 Tauchen & Hussey recommend baseSigma = sigma, and also mention baseSigma = sigmaZ;
%                 those are the first two variants in Floden (2008), and his Table 1 reports all three.
%
% Output:
%   z_grid    N*1 vector, nodes for Z
%   pi_z      N*N matrix, transition probabilities
%
% Martin Floden, Stockholm School of Economics
% January 2007 (updated August 2007)
% This version was lightly modified by Robert Kirkby
%%%%%%%%
% Original paper:
% Tauchen and Hussey (1991) - Quadrature-Based Methods for Obtaining Approximate Solutions to Nonlinear Asset Pricing Models
%    Econometrica 59(2), 371-396
% Source of the default baseSigma (and of the accuracy comparison that motivates the COMMENT below):
% Floden (2008) - A note on the accuracy of Markov-chain approximations to highly persistent AR(1) processes
%    Economics Letters 99(3), 516-520. doi:10.1016/j.econlet.2007.09.040
%
% Note on conventions: Floden writes the process as z'=(1-rho)*mu+rho*z+e, so his mu is the
% unconditional mean of z. Here mew is the intercept, so mu=mew/(1-rho)=zstar below.

%% Set default for baseSigma following Floden's suggestion (see above)
if ~exist('tauchenhusseyoptions','var')
    w=0.5+rho/4;
    sigmaZ=sigma/sqrt(1-rho^2);
    baseSigma=w*sigma+(1-w)*sigmaZ; % Floden (2008); note the second term is (1-w)*sigmaZ, NOT (1-w)*sigma*sigmaZ
else
    if ~isfield(tauchenhusseyoptions,'baseSigma')
        w=0.5+rho/4;
        sigmaZ=sigma/sqrt(1-rho^2);
        baseSigma=w*sigma+(1-w)*sigmaZ; % Floden (2008); note the second term is (1-w)*sigmaZ, NOT (1-w)*sigma*sigmaZ
    else
        baseSigma=tauchenhusseyoptions.baseSigma;
    end
end

% verbose default, and the recommendation, which is printed once per call. The test bank that
% exercises these commands calls them hundreds of times, so it is gated. Set
% tauchenhusseyoptions.verbose=0 to silence it.
if ~exist('tauchenhusseyoptions','var')
    tauchenhusseyoptions.verbose=1;
elseif ~isfield(tauchenhusseyoptions,'verbose')
    tauchenhusseyoptions.verbose=1;
end
if tauchenhusseyoptions.verbose==1
    fprintf('COMMENT: The Tauchen-Hussey method is inferior to the Farmer-Toda method for discretizing AR(1) processes. \n')
    fprintf('         It is strongly recommended you use Farmer-Toda instead. \n')
end

% z_grid=zeros(znum,1);
pi_z = zeros(znum,znum);

zstar=mew/(1-rho); % expected value of z (note: mew is the intercept of the AR(1), not the mean of z)

[z_grid,w] = gaussnorm(znum,zstar,baseSigma^2);   % See note 1 below


for i = 1:znum
    for j = 1:znum
        EZprime    = mew + rho*z_grid(i);
        pi_z(i,j) = w(j) * norm_pdf(z_grid(j),EZprime,sigma^2) / norm_pdf(z_grid(j),zstar,baseSigma^2);
    end
end

for i = 1:znum
    pi_z(i,:) = pi_z(i,:) / sum(pi_z(i,:),2);
end

a = 1;

function c = norm_pdf(x,mu,s2)
    c = 1/sqrt(2*pi*s2) * exp(-(x-mu)^2/2/s2);

function [x,w] = gaussnorm(n,mu,s2)
% Find Gaussian nodes and weights for the normal distribution
% n  = # nodes
% mu = mean
% s2 = variance
[x0,w0] = gausshermite(n);
x = x0*sqrt(2*s2) + mu;
w = w0 / sqrt(pi);


function [x,w] = gausshermite(n)
% Gauss Hermite nodes and weights following "Numerical Recipes for C"

MAXIT = 100; % was 10, which is not enough for large n: at n=101, seven of the 51 computed nodes
             % fail to converge within 10 Newton iterations (and used to do so silently, see below)
EPS   = 3e-14;
PIM4  = 0.7511255444649425;

x = zeros(n,1);
w = zeros(n,1);

m = floor(n+1)/2;
for i=1:m
    if i == 1
        z = sqrt((2*n+1)-1.85575*(2*n+1)^(-0.16667));
    elseif i == 2
        z = z - 1.14*(n^0.426)/z;
    elseif i == 3
        z = 1.86*z - 0.86*x(1);
    elseif i == 4
        z = 1.91*z - 0.91*x(2);
    else
        z = 2*z - x(i-2);
    end

    converged = 0;
    for iter = 1:MAXIT
        p1 = PIM4;
        p2 = 0;
        for j=1:n
            p3 = p2;
            p2 = p1;
            p1 = z*sqrt(2/j)*p2 - sqrt((j-1)/j)*p3;
        end
        pp = sqrt(2*n)*p2;
        z1 = z;
        z = z1 - p1/pp;
        if abs(z-z1) <= EPS
            converged = 1;
            break
        end
    end
    if converged == 0
        % Note: this used to be 'if iter>MAXIT', which can never be true. After a for loop that
        % runs to completion MATLAB leaves iter equal to MAXIT, not MAXIT+1, so the guard never
        % fired and a non-converged node was returned silently. At n=101 that produced a z_grid
        % that was neither monotone nor symmetric, with no indication anything had gone wrong.
        error('discretizeAR1_TauchenHussey: gausshermite failed to converge for node %i of %i within %i Newton iterations',i,n,MAXIT)
    end
    x(i)     = z;
    x(n+1-i) = -z;
    w(i)     = 2/pp/pp;
    w(n+1-i) = w(i);
end
x(:) = x(end:-1:1);
