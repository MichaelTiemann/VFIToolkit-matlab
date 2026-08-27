function [z_grid_J, pi_z_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_FellaGallipoliPan(rho,sigma,znum,J,fellagallipolipanoptions)
% Please cite: Fella, Gallipoli & Pan (2019) "Markov-chain approximations for life-cycle models"
%
% Fella-Gallipoli-Pan discretization method for a 'life-cycle non-stationary AR(1) process'.
% This is an extension of the Rouwenhorst method to 'age-dependent parameters'
%
%  Exteneded-Rouwenhurst method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = rho(j)*z(j-1)+ epsilon(j),   epsilon(j)~iid N(0,sigma(j))
%       with initial condition z(0) = 0 (equivalently z(1)=epsilon(1))
%
% Inputs:
%   rho 	     - Jx1 vector of serial correlation coefficients
%   sigma        - Jx1 vector of standard deviations of innovations
%   znum         - Number of grid points (scalar, is the same for all ages)
%   J            - Number of 'ages' (finite number of periods)
% Optional inputs (fellagallipolipanoptions)
%   parallel:    - set equal to 2 to use GPU, 0 to use CPU
%   nSigmas      - the grid used will be +-nSigmas*(standard deviation of z at that age)
%                  Default sqrt(znum-1), and you should not change it: the Rouwenhorst
%                  construction only reproduces the variance exactly at that one width.
% Output:
%   z_grid_J     - an znum-by-J matrix, each column stores the Markov state space for period j
%   pi_z_J       - znum-by-znum-by-(J-1) matrix of J-1 (znum-by-znum) transition matrices.
%                  Transition probabilities are arranged by row.
%                  pi_z_J(:,:,j) is transition matrix from age j to j+1 (Modified from FGP where it is j-1 to j)
%                  There are only J-1 of them, as there is no period J+1 to transition to.
%   jequaloneDistz - znum-by-1 vector, the distribution of z in period 1
%   otheroutputs - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.sigma_z     - the standard deviation of z at each age (used to determine grid)
%
% This code is by Fella, Gallipoli & Pan.
% Lightly modified by Robert Kirkby.
% !========================================================================%
% Original paper:
% Fella, Gallipoli & Pan (2019) "Markov-chain approximations for life-cycle models"
%
% Two changes from FGP2019:
%    i) Allow z0 to be a normal distribution, rather than forcing z0=0;
%    using fellagallipolipanoptions.initialj0sigmaz
%    ii) Here P_J(:,:,j) is transition from j to j+1; in FGP2019 it was from j-1 to j.
%
% FGP use MIT license, which must be included with the code, you can find it at the bottom of this
% script. (VFI Toolkit is GPL3 license, hence having to reproduce.)

sigma_z = zeros(1,J);
% z_grid = zeros(znum,J);
pi_z_J = zeros(znum,znum,J-1); % pi_z_J(:,:,jj) is the transition from period jj to period jj+1, so there are only J-1 of them

% NOTE: sqrt(znum-1) is NOT a hyperparameter here, it is what the Rouwenhorst construction
% requires. The extended-Rouwenhorst transition below is built from
%    p = (sigma_z(j+1)+rho(j+1)*sigma_z(j))/(2*sigma_z(j+1))
% on an evenly spaced grid, and that pairing reproduces sigma_z(j) exactly only when the grid
% half-width is exactly sqrt(znum-1)*sigma_z(j). Any other width and the variance is simply
% wrong. This used to read min(sqrt(znum-1),4), so the cap bound for znum>17 and the variance
% stopped being exact: measured at 2.3e-01 at znum=31 and 3.3e-01 at znum=51, against a true
% variance of the same order, where below the cap it is 2e-16. Contrast discretizeAR1_Tauchen
% and discretizeAR1_FarmerToda, where the width IS a free hyperparameter and a cap costs
% accuracy rather than correctness.

%% Set options
if ~exist('fellagallipolipanoptions','var')
    fellagallipolipanoptions.parallel=1+(gpuDeviceCount>0);
    fellagallipolipanoptions.verbose=1;
    fellagallipolipanoptions.nSigmas=sqrt(znum-1);
else
    if ~isfield(fellagallipolipanoptions,'parallel')
        fellagallipolipanoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(fellagallipolipanoptions,'verbose')
        fellagallipolipanoptions.verbose=1;
    end
    if ~isfield(fellagallipolipanoptions,'nSigmas')
        fellagallipolipanoptions.nSigmas=sqrt(znum-1);
    end
end

% The recommendation above is printed once per call, and the test bank that exercises these
% commands calls them hundreds of times, so it is gated. Set fellagallipolipanoptions.verbose=0 to silence it.
if fellagallipolipanoptions.verbose==1
    fprintf('COMMENT: The Fella-Gallipoli-Pan extended Rouwenhorst method is typically inferior to the KFTT method for discretizing life-cycle AR(1) processes. \n')
    fprintf('         It is strongly recommended you use KFTT instead. \n')
end

%% Check inputs
if znum < 2
    error('The state space has to have dimension znum>1. Exiting.')
end

if J < 2
    error('The time horizon has to have dimension J>1. Exiting.')
end

%% Step 1: construct the state space z_grid_J for each period j.
% Evenly-spaced znum-state space over [-kirkbyoptions.nSigmas*sigma_z(j),kirkbyoptions.nSigmas*sigma_z(j)].

% 1.a Compute unconditional variances of z(j)
if isfield(fellagallipolipanoptions,'initialj0sigmaz')
    sigma_z(1) = sqrt(rho(1)^2*fellagallipolipanoptions.initialj0sigmaz^2+sigma(1)^2);
else
    sigma_z(1) = sigma(1);
end

for jj = 2:J
    sigma_z(jj) = sqrt(rho(jj)^2*sigma_z(jj-1)^2+sigma(jj)^2);
end

% 1.b Construct state space
h = 2*fellagallipolipanoptions.nSigmas*sigma_z/(znum-1); % grid step (2* as is nSigmas either side of zero)
z_grid_J = repmat(h,znum,1);
z_grid_J(1,:)=-fellagallipolipanoptions.nSigmas*sigma_z;
z_grid_J = cumsum(z_grid_J,1);

%% Step 2: Compute the transition matrices trans(:,:,t) from period (t-1) to period t
% The transition matrix for period t is defined by parameter p(t).
% p(t) = 0.5*(1+rho*sigma(t-1)/sigma(t))

% Note: P(:,:,1) is the transition matrix from z(0)=0 to any gridpoint of z_grid(1) in period 1.
% Any of its rows is the (unconditional) distribution in period 1.

% Note: rhmat() is the 'Rouwenhorst matrix' subfunction

% Period 1: p(1)=0.5 as y(1) is white noise, and any row of that matrix is the period 1 distribution
pi_z_0to1 = rhmat(1/2,znum);
jequaloneDistz=pi_z_0to1(1,:)';
clear pi_z_0to1

% pi_z_J(:,:,jj) is the transition from period jj into period jj+1 (Modified from FGP where it is j-1 to j),
% and is determined by the period jj+1 parameters, hence the jj+1 indexes
for jj = 1:J-1
    p = (sigma_z(jj+1)+rho(jj+1)*sigma_z(jj))/(2*sigma_z(jj+1));
    pi_z_J(:,:,jj) = rhmat(p,znum);
end

%% I AM BEING LAZY AND JUST MOVING RESULT TO GPU RATHER THAN CREATING IT THERE IN THE FIRST PLACE
if fellagallipolipanoptions.parallel==2
    z_grid_J=gpuArray(z_grid_J);
    pi_z_J=gpuArray(pi_z_J);
    jequaloneDistz=gpuArray(jequaloneDistz);
end


%% Subfunction rhmat()
function [Pmat] = rhmat(p,N)
    % Computes Rouwenhorst matrix as a function of p and N
    Pmat = zeros(N,N);
    % Step 2(a): get the transition matrix P1 for the N=2 case
    if N == 2
        Pmat = [p, 1-p; 1-p, p];
    else
        P1 = [p, 1-p; 1-p, p];
        % Step 2(b): if the number of states N>2, apply the Rouwenhorst
        % recursion to obtain the transition matrix trans
        for ii = 2:N-1
            P2 = p *     [P1,zeros(size(P1,1),1); zeros(1,size(P1,2)),0 ] + ...
                (1-p) * [zeros(size(P1,1),1),P1; 0,zeros(1,size(P1,2)) ] + ...
                (1-p) * [zeros(1,size(P1,2)),0 ; P1,zeros(size(P1,1),1)] + ...
                p *     [0,zeros(1,size(P1,2)) ; zeros(size(P1,1),1),P1];

            P2(2:ii,:) = 0.5*P2(2:ii,:);

            if ii==N-1
                Pmat = P2;
            else
                P1 = P2;
            end
        end % of for
    end % if N == 2
end % of rhmat function

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.sigma_z=sigma_z; % Standard deviation of z (for each period)

end


%%
% The MIT License (MIT)
%
% Copyright (c) 2019 Giulio Fella, Giovanni Gallipoli and Jutong Pan
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
%
%     If you use this library or parts of it in your work, we request that you cite the package. A suggested citation is
%
%     "nmarkov-matlab: Markov-chain approximations for non-stationary AR(1) processes (Matlab version)" https://github.com/gfell/nsmarkov-matlab based on the paper "Markov-Chain Approximations for Life-Cycle Models"
%     by Giulio Fella, Giovanni Gallipoli and Jutong Pan, Review of Economic Dynamics 34, 2019 (https://doi.org/10.1016/j.red.2019.03.013).
%
%     The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
