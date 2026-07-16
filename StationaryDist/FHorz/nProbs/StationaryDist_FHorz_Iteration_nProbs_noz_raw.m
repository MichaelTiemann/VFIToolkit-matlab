function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_j,Parameters)
% 'nProbs' refers to four probabilities.
% Policy_aprime has an additional dimension of length 4 which is the four points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these four

% Policy_aprime and PolicyProbs are currently [N_a,N_probs,N_j]
N_a1=prod(n_a1);
N_a2=prod(n_a2);
if N_a2==0
    N_a=N_a1;
elseif N_a1==0
    N_a=N_a2;
else
    N_a=N_a1*N_a2;
end
Policy_aprime=gather(Policy_aprime);
needs_rounding=(PolicyProbs<=1e-6 | PolicyProbs>=1-1e-6);
PolicyProbs(needs_rounding)=round(PolicyProbs(needs_rounding));
PolicyProbs=gather(PolicyProbs);

%% Use Tan improvement

StationaryDist=zeros(N_a,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % sparse() creates a matrix of zeros
StationaryDist_lower_jj=StationaryDist_jj;
StationaryDist_upper_jj=StationaryDist_jj;
epsilon=2e-5; % A suitably small value; SD probabilities add to 1

% Precompute
II1=(1:1:N_a)';
II2=repmat((1:1:N_a)',1,N_probs); % Note the N_probs-copies

for jj=1:(N_j-1)
    % First, get Gamma
    Gammatranspose=sparse(Policy_aprime(:,:,jj),II2,PolicyProbs(:,:,jj),N_a,N_a);  % Note: sparse() will accumulate at repeated indices
    Gammatranspose_lower=sparse(Policy_aprime(:,1,jj),II1,PolicyProbs(:,1,jj),N_a,N_a);
    Gammatranspose_upper=sparse(Policy_aprime(:,2,jj),II1,PolicyProbs(:,2,jj),N_a,N_a);

    % No z, so just a single step
    StationaryDist_lower_jj=Gammatranspose_lower*StationaryDist_jj;
    StationaryDist_upper_jj=Gammatranspose_upper*StationaryDist_jj;
    StationaryDist_jj=Gammatranspose*StationaryDist_jj; % =StationaryDist_lower_jj+StationaryDist_upper_jj;

if false
    % Clean up Gaussian diffusion from Gamma step
    nnz_gamma=nnz(StationaryDist_jj);
    while nnz_gamma>8
        [epsilons,e_idx] = mink(nonzeros(StationaryDist_jj),nnz_gamma-4);
        e_idx=e_idx(epsilons<epsilon);
        epsilons=epsilons(epsilons<epsilon);
        if nnz(epsilons)==0
            break
        end
        nonzero_idx=find(StationaryDist_jj);
        % zero out likely error artifacts
        StationaryDist_jj(nonzero_idx(e_idx))=0;
        keep_nonzero=true(size(nonzero_idx));
        keep_nonzero(e_idx)=false;
        % Redistribute values zeroed out equally among remaining nonzero terms
        % By subtracting the largest zeroed epsilon, we return some
        % weight from the edges to the center of the distribution
        newdist_jj=StationaryDist_jj(nonzero_idx(keep_nonzero))-epsilons(end);
        StationaryDist_jj(nonzero_idx(keep_nonzero))=epsilon*newdist_jj./sum(newdist_jj);
        nnz_gamma=nnz(StationaryDist_jj);
    end
end

    [row_lower_idx,col_lower_idx,lower_values]=find(StationaryDist_lower_jj);
    assert(all(diff(col_lower_idx)==0))
    if isempty(row_lower_idx) || row_lower_idx(end)-row_lower_idx(1)+1>length(row_lower_idx)
        StationaryDist(:,jj+1)=gather(full(StationaryDist_jj));
        continue
    end
    multipliers_lower=(0:row_lower_idx(end)-row_lower_idx(1))'+1;
    if nnz(StationaryDist_upper_jj)>1
        % Attempt to consolidate upper into lower
        [row_upper_idx,~,upper_values]=find(StationaryDist_upper_jj);
        if row_upper_idx(end)-row_upper_idx(1)+1>length(row_upper_idx)
            % Empty rows => bail out
            StationaryDist(:,jj+1)=gather(full(StationaryDist_jj));
        end
        multipliers_upper=(row_upper_idx(1)-row_lower_idx(1):(row_upper_idx(end)-row_lower_idx(1)))'+1;
        sum_upper=sum(upper_values.*multipliers_upper);
        if sum_upper/multipliers_lower(end)+lower_values(end)<=1
            % We can fit all the upper values into slots allocated to lower
            lower_values(end)=lower_values(end)+sum_upper/multipliers_lower(end);
            % But in so doing, we may have probabilities that sum>1, so fix
            lower_values=linsolve([multipliers_lower';ones(1,length(lower_values))],[sum(lower_values.*multipliers_lower); 1]);
            needs_rounding=(lower_values<=1e-6 | lower_values>=1-1e-6);
            lower_values(needs_rounding)=round(lower_values(needs_rounding));
            StationaryDist_jj=sparse(row_lower_idx,1,lower_values,N_a,1);
        elseif row_upper_idx(1)>row_lower_idx(end) && length(unique(upper_values))>1
            if sum(upper_values.*multipliers_upper)/multipliers_upper(1)<=1
                upper_values=sum_upper/multipliers_upper(1);
                StationaryDist_jj=sparse([row_lower_idx;row_upper_idx(1)],1,[lower_values;upper_values(1)],N_a,1);
            else % Attempt to consolidate upper into itself
                for ii=1:length(upper_values)-1
                    if sum(upper_values(ii:end).*multipliers_upper(ii:end))/multipliers_upper(ii)<=1
                        upper_values(ii)=sum(upper_values(ii:end).*multipliers_upper(ii:end))/multipliers_upper(ii);
                        StationaryDist_jj=sparse([row_lower_idx;row_upper_idx(1:ii)],1,[lower_values;upper_values(1:ii)],N_a,1);
                        break
                    end
                end
                % StationaryDist_jj=StationaryDist_lower_jj+StationaryDist_upper_jj
            end
            % StationaryDist_jj=StationaryDist_lower_jj+StationaryDist_upper_jj
        end
    elseif length(unique(lower_values))>1
        % Attempt to consolidate lower into itself, if basis is valid
        for ii=1:length(lower_values)-1
            if sum(lower_values(ii:end).*multipliers_lower(ii:end))/multipliers_lower(1)<=1
                lower_values(ii)=sum(lower_values(ii:end).*multipliers_lower(ii:end))/multipliers_lower(1);
                StationaryDist_jj=sparse(row_lower_idx(1:ii),1,lower_values(1:ii),N_a,1);
                break
            end
        end
        % StationaryDist_jj=StationaryDist_lower_jj;
    end

    StationaryDist(:,jj+1)=gather(full(StationaryDist_jj));
end



% Reweight the different ages based on 'AgeWeightParamNames'. (it is assumed there is only one Age Weight Parameter (name))
try
    AgeWeights=Parameters.(AgeWeightParamNames{1});
catch
    error('Unable to find the AgeWeightParamNames in the parameter structure')
end
% I assume AgeWeights is a row vector
if size(AgeWeights,2)==1 % If it seems to be a column vector, then transpose it
    AgeWeights=AgeWeights';
end

StationaryDist=StationaryDist.*AgeWeights;

end
