function [StandardDeviationVec, CorrMatrix] = cov2corr_homemade(CovarMatrix)
% Convert covariance matrix to vector of standard deviations and correlation matrix
%
%   [StandardDeviationVec, CorrMatrix] = cov2corr_homemade(CovarMatrix)
%
% This is a drop-in replacement for the Financial Toolbox function cov2corr(),
% so the output shapes and the treatment of zero-variance processes deliberately
% match that function (note that the standard deviations come out as a row).
%
% Input:
%   CovarMatrix   : covariance matrix (n x n)
%
% Outputs:
%   StandardDeviationVec : vector of standard deviations (1 x n)
%   CorrMatrix   : correlation matrix (n x n)

    % Basic dimension check
    [n, m] = size(CovarMatrix);
    if n ~= m
        error('Covariance matrix must be square.');
    end

    % Standard deviations from diagonal of covariance matrix (row vector)
    StandardDeviationVec = sqrt(diag(CovarMatrix))';

    % Start from the identity, so that a degenerate (zero variance) process is
    % left with a correlation of one with itself and zero with everything else
    CorrMatrix = eye(n);

    % Construct the correlations, but only for the non-degenerate processes
    IndPos = StandardDeviationVec>0;
    CorrMatrix(IndPos,IndPos) = CovarMatrix(IndPos,IndPos)./(StandardDeviationVec(IndPos)'*StandardDeviationVec(IndPos));

    % Force exact ones along the main diagonal
    CorrMatrix(sub2ind([n n], 1:n, 1:n)) = 1;

end
