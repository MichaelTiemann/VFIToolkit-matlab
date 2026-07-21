function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_z,N_j,pi_z_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these N_probs.

precision=underlyingType(jequaloneDistKron);
cast2precision=str2func(precision);
total_zeros_created=0;
epsilon=1e-7;
epsilon_round=7;
jj_at_max_a2=Inf;

% Policy_aprime and PolicyProbs are currently [N_a,N_z,N_probs,N_j]
N_a1=prod(n_a1);
N_a2=prod(n_a2);
if N_a2==0
    N_a=N_a1;
elseif N_a1==0
    N_a=N_a2;
else
    N_a=N_a1*N_a2;
end
Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index following the z dimension [Tan improvement, z stays where it is]
Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z,N_probs,N_j])); % sparse() requires inputs to be 2-D
needs_rounding=(PolicyProbs<epsilon | PolicyProbs>1-epsilon);
PolicyProbs(needs_rounding)=round(PolicyProbs(needs_rounding));
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,N_probs,N_j])); % sparse() requires inputs to be 2-D

%% Use Tan improvement

StationaryDist=zeros(N_a*N_z,N_j,precision,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % use sparse matrix

% Precompute
II1=(1:1:N_a*N_z)';
II2=repmat((1:1:N_a*N_z)',1,N_probs); %  Index for this period (a,z), note the N_probs-copies

for jj=1:(N_j-1)
    age_zeros_created=total_zeros_created;

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprimez(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices
    Gammatranspose_lower=sparse(Policy_aprimez(:,1,jj),II1,PolicyProbs(:,1,jj),N_a*N_z,N_a*N_z);
    Gammatranspose_upper=sparse(Policy_aprimez(:,2,jj),II1,PolicyProbs(:,2,jj),N_a*N_z,N_a*N_z);

    % First step of Tan improvement
    needs_rounding=full(StationaryDist_jj<epsilon | StationaryDist_jj>1-epsilon);
    needs_rounding(StationaryDist_jj==0)=0;
    needs_rounding(StationaryDist_jj==1)=0;
    StationaryDist_jj(needs_rounding)=round(StationaryDist_jj(needs_rounding));
    StationaryDist_lower_jj=reshape(Gammatranspose_lower*StationaryDist_jj,[N_a,N_z]);
    StationaryDist_upper_jj=reshape(Gammatranspose_upper*StationaryDist_jj,[N_a,N_z]);
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]);

    % Second step of Tan improvement
    pi_z=sparse(gather(pi_z_J(:,:,jj)));
    StationaryDist_jj=StationaryDist_jj*pi_z;
    StationaryDist_lower_jj=StationaryDist_lower_jj*pi_z;
    StationaryDist_upper_jj=StationaryDist_upper_jj*pi_z;

    for z_c=1:N_z
        StationaryDist_rowz_jj=reshape(StationaryDist_jj(:,z_c),[N_a1,N_a2]);
        probability_z=full(sum(StationaryDist_rowz_jj,'all'));
        if probability_z==0 || all(arrayfun(@(r) nnz(StationaryDist_rowz_jj(r,:)), 1:size(StationaryDist_rowz_jj,1))<3)
            % Sometimes nobody chooses the path less taken
            continue
        end
        if jj<jj_at_max_a2 && any(StationaryDist_rowz_jj(:,N_a2)~=0)
            jj_at_max_a2=jj;
        end
        StationaryDist_lowerz_jj=reshape(StationaryDist_lower_jj(:,z_c),[N_a1,N_a2]);
        StationaryDist_upperz_jj=reshape(StationaryDist_upper_jj(:,z_c),[N_a1,N_a2]);

        [rows,~]=find(StationaryDist_rowz_jj~=0);
        for row=unique(rows')
            % Process agents' ExpAssets row by row (i.e., each N_a1 asset mixture)
            row_prob_sum=full(sum(StationaryDist_rowz_jj(row,:),2));

            [~,ea_lower_idx,ea_lower_vals]=find(StationaryDist_lowerz_jj(row,:));
            if isempty(ea_lower_idx)
                % Swap and try again; the empty half-distribution will not prevent us from getting the job done
                ea_upper_idx=ea_lower_idx; % empty!!
                [~,ea_lower_idx,ea_lower_vals]=find(StationaryDist_upperz_jj(row,:));
                if length(ea_lower_idx)==2
                    % Below we take care of singletons, which might need to evaporate
                    continue
                end
            else
                [~,ea_upper_idx,ea_upper_vals]=find(StationaryDist_upperz_jj(row,:));
            end
            noise_vals=(ea_lower_vals/row_prob_sum)<1e-5;
            noise_vals(find(noise_vals==0,1,'first'):end)=0;
            if false && any(noise_vals)
                temp_cols=ea_lower_idx(noise_vals);
                ea_lower_idx=ea_lower_idx(~noise_vals);
                ea_lower_vals=ea_lower_vals(~noise_vals);
                StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=0;
            end

            if isempty(ea_upper_idx)
                if isscalar(ea_lower_idx) && row_prob_sum<epsilon && (ea_lower_idx==1 || ea_lower_idx==N_a2)
                    % Allow this infinitesimal to evaporate from grid
                    total_zeros_created=total_zeros_created+nnz(ea_lower_vals);
                    temp=sparse(row,ea_lower_idx,0,N_a1,N_a2);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,ea_lower_idx,z_c),z_c)=temp(row,ea_lower_idx);
                    continue
                elseif length(ea_lower_idx)<3
                    continue
                end
            else
                noise_vals=(ea_upper_vals/row_prob_sum)<1e-5;
                noise_vals(1:find(noise_vals==0,1,'last'))=0;
                if false && any(noise_vals)
                    temp_cols=ea_upper_idx(noise_vals);
                    ea_upper_idx=ea_upper_idx(~noise_vals);
                    ea_upper_vals=ea_upper_vals(~noise_vals);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=0;
                end

                if isempty(ea_lower_idx) || isempty(ea_upper_idx) || ea_upper_idx(end)-ea_lower_idx(1)<2
                    continue
                end
            end

            % We have two strategies for dealing with gaps.  The first is
            % to see whether the gap disappears when we look at the whole
            % picture.  It can happen that lower/upper columns are
            % disjoint like this: [2 4] [3 5] which gives 4-in-a-row.
            % If we see we have [2 3 4 5] we have to reconstruct
            % lower/upper columns somehow.  In this case, it is possible
            % for a singleton upper to have a lower index than a lower,
            % so we take care to fix that.

            % The second strategy is to fill in a single hole if that
            % allows us to connect lower and upper columns.  In the above
            % case we get [2 3* 4] [3 4* 5] where * means inserted zero.
            % This is simpler because lower/upper are already split.
            [~,ea_allz_idx,allz_vals_jj]=find(StationaryDist_rowz_jj(row,:));
            p=find(diff(ea_allz_idx)>1); % p columns are start and end of consecutive elements
            runs=[ea_allz_idx(1),ea_allz_idx(p+1);ea_allz_idx(p),ea_allz_idx(end)];

            if size(runs,2)>1
                single_gaps=ea_allz_idx(p+1)-ea_allz_idx(p)==2;
                if any(single_gaps)
                    % A zero logically bubbled in...so make space for it for now
                    s=p(single_gaps);
                    z = zeros(1,length(ea_allz_idx)+length(s));  %initialise a new vector of the appropriate size
                    z(s+(1:length(s))) = ea_allz_idx(s)+1; % set locations in 's' to s+1, which will have the value zero
                    z(z==0) = ea_allz_idx; %insert the original values in ea_allz_idx into the new vector at their new positions.
                    ea_allz_idx=z;
                    z = nan(1,length(allz_vals_jj)+length(s));  %initialise a new vector of the appropriate size
                    z(s+(1:length(s))) = 0; % set value locations in 'p' to zero
                    z(isnan(z)) = allz_vals_jj; %insert the original values in allz_vals_jj into the new vector at their new positions.
                    allz_vals_jj=z;
                    ea_gaps=find(diff(ea_allz_idx)>1);
                    p=find(diff(ea_allz_idx)>1); % p columns are start and end of consecutive elements
                    runs=[ea_allz_idx(1),ea_allz_idx(p+1);ea_allz_idx(p),ea_allz_idx(end)];
                    gap_idx=ea_allz_idx(s)';
                    single_gaps=sum(gap_idx>=runs(1,:) & gap_idx<=runs(2,:),1);
                else
                    single_gaps=zeros(1,size(runs,2));
                end
                for ridx=1:size(runs,2)
                    if runs(2,ridx)-runs(1,ridx)<2
                        % Remove traces of any short sequences
                        temp=ea_lower_idx<runs(1,ridx) | ea_lower_idx>runs(2,ridx);
                        ea_lower_idx=ea_lower_idx(temp);
                        ea_lower_vals=ea_lower_vals(temp);
                        temp=ea_upper_idx<runs(1,ridx) | ea_upper_idx>runs(2,ridx);
                        ea_upper_idx=ea_upper_idx(temp);
                        ea_upper_vals=ea_upper_vals(temp);
                    else
                        % Fill in zeros for everything we track
                        ea_lower_end=find(ea_lower_idx>=runs(1,ridx) & ea_lower_idx<=runs(2,ridx),1,'last');
                        ea_upper_1=find(ea_upper_idx>=runs(1,ridx) & ea_upper_idx<=runs(2,ridx),1,'first');
                        if ea_lower_idx(ea_lower_end)~=ea_upper_idx(ea_upper_1) || single_gaps(ridx)
                            % Merging disjoint/overlapping lower and upper
                            ea_lower_1=find(ea_lower_idx>=runs(1,ridx),1,'first');
                            if ea_lower_idx(ea_lower_1)>ea_upper_idx(ea_upper_1)
                                % add zeros to lower so both start at the same index
                                new_zeros=ea_lower_idx(ea_lower_1)-ea_upper_idx(ea_upper_1);
                                ea_lower_vals=[zeros(1,new_zeros),ea_lower_vals];
                                ea_lower_end=ea_lower_end+new_zeros;
                                ea_lower_idx=[ea_upper_idx(ea_upper_1)+(0:new_zeros-1),ea_lower_idx];
                            end
                            if ea_lower_end>ea_lower_1
                                % Non-singleton, so maybe insert zeros
                                new_vals=zeros(1,ea_lower_idx(ea_lower_end)-ea_lower_idx(ea_lower_1)+1); % pick up the new zeros
                                temp=ea_lower_idx>=runs(1,ridx) & ea_lower_idx<=runs(2,ridx);
                                new_vals(ea_lower_idx(temp)-ea_lower_idx(ea_lower_1)+1)=ea_lower_vals(temp);
                                ea_lower_vals=[ea_lower_vals(1:ea_lower_1-1), new_vals, ea_lower_vals(ea_lower_end+1:end)];
                                ea_lower_idx=[ea_lower_idx(1:ea_lower_1-1), ea_lower_idx(ea_lower_1):ea_lower_idx(ea_lower_end), ea_lower_idx(ea_lower_end+1:end)];
                            end
                            ea_upper_end=find(ea_upper_idx<=runs(2,ridx),1,'last');
                            if ea_upper_end>ea_upper_1
                                % Non-singleton, so maybe insert zeros
                                new_vals=zeros(1,ea_upper_idx(ea_upper_end)-ea_upper_idx(ea_upper_1)+1); % pick up the new zeros
                                temp=ea_upper_idx>=runs(1,ridx) & ea_upper_idx<=runs(2,ridx);
                                new_vals(ea_upper_idx(temp)-ea_upper_idx(ea_upper_1)+1)=ea_upper_vals(temp);
                                ea_upper_vals=[ea_upper_vals(1:ea_upper_1-1), new_vals, ea_upper_vals(ea_upper_end+1:end)];
                                ea_upper_idx=[ea_upper_idx(1:ea_upper_1-1), ea_upper_idx(ea_upper_1):ea_upper_idx(ea_upper_end), ea_upper_idx(ea_upper_end+1:end)];
                            end
                        else
                            continue
                        end
                    end
                end
                runs=runs(:,runs(2,:)-runs(1,:)>1);
                if isempty(runs)
                    % We have disqualified all merging opportunities
                    continue
                elseif size(runs,2)==1
                    ea_gaps=[];
                end
            else
                if ea_lower_idx(end)-ea_lower_idx(1)>=length(ea_lower_idx)
                    % We have no gaps in the big picture, but lower gaps
                    p=find(diff(ea_lower_idx)>1); % p columns are start and end of consecutive elements
                    runs=[ea_lower_idx(1),ea_lower_idx(p+1);ea_lower_idx(p),ea_lower_idx(end)];
                    z = zeros(1,length(ea_lower_idx)+length(p));  %initialise a new vector of the appropriate size
                    z(p+(1:length(p))) = ea_lower_idx(p)+1; % set locations in 'p' to p+1, which will have the value zero
                    z(z==0) = ea_lower_idx; %insert the original values in ea_lower_idx into the new vector at their new positions.
                    ea_lower_idx=z;
                    z = nan(1,length(ea_lower_vals)+length(p));  %initialise a new vector of the appropriate size
                    z(p+(1:length(p))) = 0; % set value locations in 'p' to zero
                    z(isnan(z)) = ea_lower_vals; %insert the original values in ea_lower_vals into the new vector at their new positions.
                    ea_lower_vals=z;
                end
                if ea_upper_idx(end)-ea_upper_idx(1)>=length(ea_upper_idx)
                    % We have no gaps in the big picture, but upper gaps
                    p=find(diff(ea_upper_idx)>1); % p columns are start and end of consecutive elements
                    runs=[ea_upper_idx(1),ea_upper_idx(p+1);ea_upper_idx(p),ea_upper_idx(end)];
                    z = zeros(1,length(ea_upper_idx)+length(p));  %initialise a new vector of the appropriate size
                    z(p+(1:length(p))) = ea_upper_idx(p)+1; % set locations in 'p' to p+1, which will have the value zero
                    z(z==0) = ea_upper_idx; %insert the original values in ea_upper_idx into the new vector at their new positions.
                    ea_upper_idx=z;
                    z = nan(1,length(ea_upper_vals)+length(p));  %initialise a new vector of the appropriate size
                    z(p+(1:length(p))) = 0; % set value locations in 'p' to zero
                    z(isnan(z)) = ea_upper_vals; %insert the original values in ea_upper_vals into the new vector at their new positions.
                    ea_upper_vals=z;
                end
                ea_gaps=[];
            end

            [ea_lower_idx,sort_idx]=sort(ea_lower_idx);
            ea_lower_vals=ea_lower_vals(sort_idx);
            if isempty(ea_gaps)
                lower_gaps=ea_gaps;
            else
                lower_gaps=find(diff(ea_lower_idx)>1);
            end

            group_lower_idx=[0,lower_gaps,length(ea_lower_idx)];
            for ll=1:length(group_lower_idx)-1
                ea_group_lower_idx=ea_lower_idx(group_lower_idx(ll)+1:group_lower_idx(ll+1));
                lower_vals=ea_lower_vals(group_lower_idx(ll)+1:group_lower_idx(ll+1));
    
                multiplier_lower=ea_group_lower_idx-ea_group_lower_idx(1)'+1;
    
                if isempty(ea_upper_idx)
                    if nnz(lower_vals)>2
                        % Attempt to consolidate lower into itself
                        assert(false);
                        for ii=1:length(lower_vals)-1
                            if sum(lower_vals(ii:end).*multiplier_lower(ii:end))/multiplier_lower(1)<=row_prob_sum
                                lower_vals(ii)=sum(lower_vals(ii:end).*multiplier_lower(ii:end))/multiplier_lower(1);
                                temp=sparse(row,ea_group_lower_idx(1:ii),lower_vals(1:ii),N_a1,N_a2);
                                temp_cols=ea_group_lower_idx(1):ea_group_lower_idx(end);
                                StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
                                break
                            end
                        end
                    else
                        continue
                    end
                end

                % Attempt to consolidate upper and lower
                [ea_upper_idx,sort_idx]=sort(ea_upper_idx);
                ea_upper_vals=ea_upper_vals(sort_idx);
                if isempty(ea_gaps)
                    upper_gaps=ea_gaps;
                else
                    upper_gaps=find(diff(ea_upper_idx)>1);
                end

                group_upper_idx=[0,upper_gaps,length(ea_upper_idx)];
                assert(length(group_lower_idx)==length(group_upper_idx))
                uu=ll; % we keep these two in sync
                ea_group_upper_idx=ea_upper_idx(group_upper_idx(uu)+1:group_upper_idx(uu+1));
                upper_vals=ea_upper_vals(group_upper_idx(uu)+1:group_upper_idx(uu+1));

                multiplier_upper=ea_group_upper_idx-ea_group_lower_idx(1)+1;
    
                sum_lower=sum(lower_vals.*multiplier_lower);
                sum_upper=sum(upper_vals.*multiplier_upper);
                starting_zeros=sum(lower_vals==0)+sum(upper_vals==0);
    
                if length(unique(multiplier_lower))>1 && (sum_upper+sum_lower)/multiplier_lower(end)<=row_prob_sum
                    % We can fit all the upper values into slots allocated to lower with a basis to work with
                    lower_vals(end)=lower_vals(end)+sum_upper/multiplier_lower(end);
                    % But in so doing, we may have probabilities that sum>1, so fix
                    zero_created=false;
                    if length(lower_vals)>2
                        next_candidate=length(lower_vals);
                        zero_candidate=zeros(1,next_candidate);
                        zero_candidate(next_candidate)=1;
                        while nnz(lower_vals)>1
                            % Aggressively try to zero out largest indices
                            new_vals=linsolve([multiplier_lower;ones(1,length(lower_vals));zero_candidate],[sum(lower_vals.*multiplier_lower); row_prob_sum; 0])';
                            new_vals=round(new_vals,epsilon_round);
                            if all(new_vals==lower_vals) || any(new_vals<0)
                                break
                            end
                            lower_vals=new_vals;
                            zero_created=true;
                            next_candidate=find(zero_candidate==0,1,'last');
                            zero_candidate(next_candidate)=1;
                        end
                        if zero_created
                            zero_candidate(next_candidate)=0;
                        end
                        next_candidate=1;
                        zero_candidate(next_candidate)=1;
                        while nnz(lower_vals)>1
                            % Try to zero out least index
                            new_vals=linsolve([multiplier_lower;ones(1,length(lower_vals));zero_candidate],[sum(lower_vals.*multiplier_lower); row_prob_sum; 0])';
                            new_vals=round(new_vals,epsilon_round);
                            if all(new_vals==lower_vals) || any(new_vals<0) || any(isnan(new_vals))
                                break
                            end
                            lower_vals=new_vals;
                            zero_created=true;
                            next_candidate=find(zero_candidate==0,1,'first');
                            zero_candidate(next_candidate)=1;
                        end
                    end
                    if ~zero_created
                        % Just re-balance the indices (possibly creating a zero in the middle we cannot move to either end of lower_vals
                        new_vals=linsolve([multiplier_lower;ones(1,length(lower_vals))],[sum(lower_vals.*multiplier_lower); row_prob_sum])';
                        new_vals=round(new_vals,epsilon_round);
                        if any(new_vals<0)
                            break
                        end
                        lower_vals=new_vals;
                    end
                    temp=sparse(row,ea_group_lower_idx,lower_vals,N_a1,N_a2);
                    temp_cols=ea_group_lower_idx(1):ea_group_upper_idx(end);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
                    total_zeros_created=total_zeros_created+sum(lower_vals==0)+length(upper_vals)-starting_zeros;
                    continue
                elseif length(unique(multiplier_upper))>1 && sum_lower/multiplier_upper(end-1)+sum(upper_vals)<=row_prob_sum
                    % We can fit all the lower values into slots allocated to upper with a basis to work with
                    upper_vals(end-1)=upper_vals(end-1)+sum_lower/multiplier_upper(end-1);
                    % But in so doing, we may have probabilities that sum>1, so fix
                    zero_created=false;
                    if length(upper_vals)>2
                        next_candidate=length(upper_vals);
                        zero_candidate=zeros(1,next_candidate);
                        zero_candidate(next_candidate)=1;
                        while nnz(upper_vals)>1
                            % Aggressively try to zero out largest indices
                            new_vals=linsolve([multiplier_upper;ones(1,length(upper_vals));zero_candidate],[sum(upper_vals.*multiplier_upper); row_prob_sum; 0])';
                            new_vals=round(new_vals,epsilon_round);
                            if all(new_vals==upper_vals) || any(new_vals<0) || any(isnan(new_vals))
                                break
                            end
                            upper_vals=new_vals;
                            zero_created=true;
                            next_candidate=find(zero_candidate==0,1,'last');
                            zero_candidate(next_candidate)=1;
                        end
                        if zero_created
                            zero_candidate(next_candidate)=0;
                        end
                        next_candidate=1;
                        zero_candidate(next_candidate)=1;
                        while nnz(upper_vals)>1
                            % Try to zero out least index
                            new_vals=linsolve([multiplier_upper;ones(1,length(upper_vals));zero_candidate],[sum(upper_vals.*multiplier_upper); row_prob_sum; 0])';
                            new_vals=round(new_vals,epsilon_round);
                            if all(new_vals==upper_vals) || any(new_vals<0)
                                break
                            end
                            upper_vals=new_vals;
                            zero_created=true;
                            next_candidate=find(zero_candidate==0,1,'first');
                            zero_candidate(next_candidate)=1;
                        end
                    end
                    if ~zero_created
                        % Just re-balance the indices (possibly creating a zero in the middle we cannot move to either end of lower_vals
                        new_vals=linsolve([multiplier_upper;ones(1,length(upper_vals))],[sum(upper_vals.*multiplier_upper); row_prob_sum])';
                        new_vals=round(new_vals,epsilon_round);
                        if any(new_vals<0)
                            break
                        end
                        upper_vals=new_vals;
                    end
                    temp=sparse(row,ea_group_upper_idx,upper_vals,N_a1,N_a2);
                    temp_cols=ea_group_lower_idx(1):ea_group_upper_idx(end);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
                    total_zeros_created=total_zeros_created+sum(upper_vals==0)+length(lower_vals)-starting_zeros;
                    continue
                elseif length(unique(multiplier_upper))>2
                    % Attempt to consolidate upper into itself
                    assert(false);
                end
            end
        end
    end

    StationaryDist_jj=reshape(StationaryDist_jj,[N_a*N_z,1]);
    assert(all(StationaryDist_jj>=0));
    StationaryDist(:,jj+1)=gpuArray(full(StationaryDist_jj));

    fprintf("Age %3d: zeros created = %d \n", jj, total_zeros_created-age_zeros_created);
end



% Reweight the different ages based on 'AgeWeightParamNames'. (it is assumed there is only one Age Weight Parameter (name))
try
    AgeWeights=cast2precision(Parameters.(AgeWeightParamNames{1}));
catch
    error('Unable to find the AgeWeightParamNames in the parameter structure')
end
% I assume AgeWeights is a row vector
if size(AgeWeights,2)==1 % If it seems to be a column vector, then transpose it
    AgeWeights=AgeWeights';
end

StationaryDist=StationaryDist.*AgeWeights;

fprintf("With epsilon = %.2e, total zeros created = %d \n", epsilon, total_zeros_created);
if isfinite(jj_at_max_a2)
    fprintf("Max ExpAsset value first observed at age %3d \n", jj_at_max_a2);
else
    temp=reshape(StationaryDist,[N_a1,N_a2,N_z,N_j]);
    [a1,a2,z_c,age_j]=ind2sub(size(temp),find(temp~=0));
    max_a2=max(a2);
    jj_at_max_a2=age_j(find(a2==max_a2));
    fprintf("Max ExpAsset index reached = %3d (of %3d) at age %3d \n", max_a2, N_a2, jj_at_max_a2);
end

end
