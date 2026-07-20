function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_z,N_j,pi_z_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these N_probs.

precision=underlyingType(jequaloneDistKron);
cast2precision=str2func(precision);
total_zeros_created=0;
epsilon=1e-7;
epsilon_round=7;

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
StationaryDist_lower_jj=StationaryDist_jj;
StationaryDist_upper_jj=StationaryDist_jj;

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
        probability_z=full(sum(StationaryDist_jj(:,z_c),1));
        if probability_z==0 || nnz(StationaryDist_jj(:,z_c))<3
            % Sometimes nobody chooses the path less taken
            continue
        end
        StationaryDist_rowz_jj=reshape(StationaryDist_jj(:,z_c),[N_a1,N_a2]);
        StationaryDist_lowerz_jj=reshape(StationaryDist_lower_jj(:,z_c),[N_a1,N_a2]);
        StationaryDist_upperz_jj=reshape(StationaryDist_upper_jj(:,z_c),[N_a1,N_a2]);

        [rows,~]=find(StationaryDist_rowz_jj~=0);
        for row=unique(rows')
            % Process agents' ExpAssets row by row (i.e., each N_a1 asset mixture)
            probability_row=full(sum(StationaryDist_rowz_jj(row,:),2));

            [~,col_lowerz_idx_jj,lowerz_values_jj]=find(StationaryDist_lowerz_jj(row,:));
            if isempty(col_lowerz_idx_jj)
                % Swap and try again; the empty half-distribution will not prevent us from getting the job done
                col_upperz_idx_jj=col_lowerz_idx_jj; % empty!!
                [~,col_lowerz_idx_jj,lowerz_values_jj]=find(StationaryDist_upperz_jj(row,:));
                if length(col_lowerz_idx_jj)==2
                    continue
                end
            else
                [~,col_upperz_idx_jj,upperz_values_jj]=find(StationaryDist_upperz_jj(row,:));
            end
            noise_vals=(lowerz_values_jj/probability_row)<1e-5;
            noise_vals(find(noise_vals==0,1,'first'):end)=0;
            if false && any(noise_vals)
                temp_cols=col_lowerz_idx_jj(noise_vals);
                col_lowerz_idx_jj=col_lowerz_idx_jj(~noise_vals);
                lowerz_values_jj=lowerz_values_jj(~noise_vals);
                StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=0;
            end

            if isempty(col_upperz_idx_jj)
                if isscalar(col_lowerz_idx_jj) && probability_row<epsilon && (col_lowerz_idx_jj==1 || col_lowerz_idx_jj==N_a2)
                    % Allow this infinitesimal to evaporate from grid
                    total_zeros_created=total_zeros_created+nnz(lowerz_values_jj);
                    temp=sparse(row,col_lowerz_idx_jj,0,N_a1,N_a2);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,col_lowerz_idx_jj,z_c),z_c)=temp(row,col_lowerz_idx_jj);
                    continue
                elseif length(col_lowerz_idx_jj)<3
                    continue
                end
            else
                noise_vals=(upperz_values_jj/probability_row)<1e-5;
                noise_vals(1:find(noise_vals==0,1,'last'))=0;
                if false && any(noise_vals)
                    temp_cols=col_upperz_idx_jj(noise_vals);
                    col_upperz_idx_jj=col_upperz_idx_jj(~noise_vals);
                    upperz_values_jj=upperz_values_jj(~noise_vals);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=0;
                end

                if isempty(col_lowerz_idx_jj) || isempty(col_upperz_idx_jj) || col_upperz_idx_jj(end)-col_lowerz_idx_jj(1)<2
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
            [~,col_allz_idx,allz_values_jj]=find(StationaryDist_rowz_jj(row,:));
            p=find(diff(col_allz_idx)>1); % p columns are start and end of consecutive elements
            ind=[col_allz_idx(1),col_allz_idx(p+1);col_allz_idx(p),col_allz_idx(end)];

            if size(ind,2)>1
                single_gaps=col_allz_idx(p+1)-col_allz_idx(p)==2;
                if any(single_gaps)
                    % A zero logically bubbled in...so make space for it for now
                    s=p(single_gaps);
                    z = zeros(1,length(col_allz_idx)+length(s));  %initialise a new vector of the appropriate size
                    z(s+(1:length(s))) = col_allz_idx(s)+1; % set locations in 's' to s+1, which will have the value zero
                    z(z==0) = col_allz_idx; %insert the original values in col_allz_idx into the new vector at their new positions.
                    col_allz_idx=z;
                    z = nan(1,length(allz_values_jj)+length(s));  %initialise a new vector of the appropriate size
                    z(s+(1:length(s))) = 0; % set value locations in 'p' to zero
                    z(isnan(z)) = allz_values_jj; %insert the original values in allz_values_jj into the new vector at their new positions.
                    allz_values_jj=z;
                    col_gaps=find(diff(col_allz_idx)>1);
                    p=find(diff(col_allz_idx)>1); % p columns are start and end of consecutive elements
                    ind=[col_allz_idx(1),col_allz_idx(p+1);col_allz_idx(p),col_allz_idx(end)];
                    gap_idx=col_allz_idx(s)';
                    single_gaps=sum(gap_idx>=ind(1,:) & gap_idx<=ind(2,:),1);
                else
                    single_gaps=zeros(1,size(ind,2));
                end
                for ind_idx=1:size(ind,2)
                    if ind(2,ind_idx)-ind(1,ind_idx)<2
                        % Remove traces of any short sequences
                        temp=col_lowerz_idx_jj<ind(1,ind_idx) | col_lowerz_idx_jj>ind(2,ind_idx);
                        col_lowerz_idx_jj=col_lowerz_idx_jj(temp);
                        lowerz_values_jj=lowerz_values_jj(temp);
                        temp=col_upperz_idx_jj<ind(1,ind_idx) | col_upperz_idx_jj>ind(2,ind_idx);
                        col_upperz_idx_jj=col_upperz_idx_jj(temp);
                        upperz_values_jj=upperz_values_jj(temp);
                    else
                        % Fill in zeros for everything we track
                        col_lowerz_end=find(col_lowerz_idx_jj>=ind(1,ind_idx) & col_lowerz_idx_jj<=ind(2,ind_idx),1,'last');
                        col_upperz_1=find(col_upperz_idx_jj>=ind(1,ind_idx) & col_upperz_idx_jj<=ind(2,ind_idx),1,'first');
                        if col_lowerz_idx_jj(col_lowerz_end)~=col_upperz_idx_jj(col_upperz_1) || single_gaps(ind_idx)
                            % Merging disjoint/overlapping lower and upper
                            col_lowerz_1=find(col_lowerz_idx_jj>=ind(1,ind_idx),1,'first');
                            if col_lowerz_idx_jj(col_lowerz_1)>col_upperz_idx_jj(col_upperz_1)
                                % add zeros to lower so both start at the same index
                                new_zeros=col_lowerz_idx_jj(col_lowerz_1)-col_upperz_idx_jj(col_upperz_1);
                                lowerz_values_jj=[zeros(1,new_zeros),lowerz_values_jj];
                                col_lowerz_end=col_lowerz_end+new_zeros;
                                col_lowerz_idx_jj=[col_upperz_idx_jj(col_upperz_1)+(0:new_zeros-1),col_lowerz_idx_jj];
                            end
                            if col_lowerz_end>col_lowerz_1
                                % Non-singleton, so maybe insert zeros
                                new_values=zeros(1,col_lowerz_idx_jj(col_lowerz_end)-col_lowerz_idx_jj(col_lowerz_1)+1); % pick up the new zeros
                                temp=col_lowerz_idx_jj>=ind(1,ind_idx) & col_lowerz_idx_jj<=ind(2,ind_idx);
                                new_values(col_lowerz_idx_jj(temp)-col_lowerz_idx_jj(col_lowerz_1)+1)=lowerz_values_jj(temp);
                                lowerz_values_jj=[lowerz_values_jj(1:col_lowerz_1-1), new_values, lowerz_values_jj(col_lowerz_end+1:end)];
                                col_lowerz_idx_jj=[col_lowerz_idx_jj(1:col_lowerz_1-1), col_lowerz_idx_jj(col_lowerz_1):col_lowerz_idx_jj(col_lowerz_end), col_lowerz_idx_jj(col_lowerz_end+1:end)];
                            end
                            col_upperz_end=find(col_upperz_idx_jj<=ind(2,ind_idx),1,'last');
                            if col_upperz_end>col_upperz_1
                                % Non-singleton, so maybe insert zeros
                                new_values=zeros(1,col_upperz_idx_jj(col_upperz_end)-col_upperz_idx_jj(col_upperz_1)+1); % pick up the new zeros
                                temp=col_upperz_idx_jj>=ind(1,ind_idx) & col_upperz_idx_jj<=ind(2,ind_idx);
                                new_values(col_upperz_idx_jj(temp)-col_upperz_idx_jj(col_upperz_1)+1)=upperz_values_jj(temp);
                                upperz_values_jj=[upperz_values_jj(1:col_upperz_1-1), new_values, upperz_values_jj(col_upperz_end+1:end)];
                                col_upperz_idx_jj=[col_upperz_idx_jj(1:col_upperz_1-1), col_upperz_idx_jj(col_upperz_1):col_upperz_idx_jj(col_upperz_end), col_upperz_idx_jj(col_upperz_end+1:end)];
                            end
                        else
                            continue
                        end
                    end
                end
                ind=ind(:,ind(2,:)-ind(1,:)>1);
                if isempty(ind)
                    % We have disqualified all merging opportunities
                    continue
                elseif size(ind,2)==1
                    col_gaps=[];
                end
            else
                if col_lowerz_idx_jj(end)-col_lowerz_idx_jj(1)>=length(col_lowerz_idx_jj)
                    % We have no gaps in the big picture, but lower gaps
                    p=find(diff(col_lowerz_idx_jj)>1); % p columns are start and end of consecutive elements
                    ind=[col_lowerz_idx_jj(1),col_lowerz_idx_jj(p+1);col_lowerz_idx_jj(p),col_lowerz_idx_jj(end)];
                    z = zeros(1,length(col_lowerz_idx_jj)+length(p));  %initialise a new vector of the appropriate size
                    z(p+(1:length(p))) = col_lowerz_idx_jj(p)+1; % set locations in 'p' to p+1, which will have the value zero
                    z(z==0) = col_lowerz_idx_jj; %insert the original values in col_lowerz_idx_jj into the new vector at their new positions.
                    col_lowerz_idx_jj=z;
                    z = nan(1,length(lowerz_values_jj)+length(p));  %initialise a new vector of the appropriate size
                    z(p+(1:length(p))) = 0; % set value locations in 'p' to zero
                    z(isnan(z)) = lowerz_values_jj; %insert the original values in lowerz_values_jj into the new vector at their new positions.
                    lowerz_values_jj=z;
                end
                if col_upperz_idx_jj(end)-col_upperz_idx_jj(1)>=length(col_upperz_idx_jj)
                    % We have no gaps in the big picture, but upper gaps
                    p=find(diff(col_upperz_idx_jj)>1); % p columns are start and end of consecutive elements
                    ind=[col_upperz_idx_jj(1),col_upperz_idx_jj(p+1);col_upperz_idx_jj(p),col_upperz_idx_jj(end)];
                    z = zeros(1,length(col_upperz_idx_jj)+length(p));  %initialise a new vector of the appropriate size
                    z(p+(1:length(p))) = col_upperz_idx_jj(p)+1; % set locations in 'p' to p+1, which will have the value zero
                    z(z==0) = col_upperz_idx_jj; %insert the original values in col_upperz_idx_jj into the new vector at their new positions.
                    col_upperz_idx_jj=z;
                    z = nan(1,length(upperz_values_jj)+length(p));  %initialise a new vector of the appropriate size
                    z(p+(1:length(p))) = 0; % set value locations in 'p' to zero
                    z(isnan(z)) = upperz_values_jj; %insert the original values in upperz_values_jj into the new vector at their new positions.
                    upperz_values_jj=z;
                end
                col_gaps=[];
            end

            [col_lowerz_idx_jj,sort_idx]=sort(col_lowerz_idx_jj);
            lowerz_values_jj=lowerz_values_jj(sort_idx);
            if isempty(col_gaps)
                lowerz_gaps=col_gaps;
            else
                lowerz_gaps=find(diff(col_lowerz_idx_jj)>1);
            end

            lower_group_idx=[0,lowerz_gaps,length(col_lowerz_idx_jj)];
            for ll=1:length(lower_group_idx)-1
                col_lowerz_idx=col_lowerz_idx_jj(lower_group_idx(ll)+1:lower_group_idx(ll+1));
                lowerz_values=lowerz_values_jj(lower_group_idx(ll)+1:lower_group_idx(ll+1));
    
                multiplierz_lower=col_lowerz_idx-col_lowerz_idx(1)'+1;
    
                if isempty(col_upperz_idx_jj)
                    if nnz(lowerz_values)>2
                        % Attempt to consolidate lower into itself
                        assert(false);
                        for ii=1:length(lowerz_values)-1
                            if sum(lowerz_values(ii:end).*multiplierz_lower(ii:end))/multiplierz_lower(1)<=probability_row
                                lowerz_values(ii)=sum(lowerz_values(ii:end).*multiplierz_lower(ii:end))/multiplierz_lower(1);
                                temp=sparse(row,col_lowerz_idx(1:ii),lowerz_values(1:ii),N_a1,N_a2);
                                temp_cols=col_lowerz_idx(1):col_lowerz_idx(end);
                                StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
                                break
                            end
                        end
                    else
                        continue
                    end
                end

                % Attempt to consolidate upper and lower
                [col_upperz_idx_jj,sort_idx]=sort(col_upperz_idx_jj);
                upperz_values_jj=upperz_values_jj(sort_idx);
                if isempty(col_gaps)
                    upperz_gaps=col_gaps;
                else
                    upperz_gaps=find(diff(col_upperz_idx_jj)>1);
                end

                upper_group_idx=[0,upperz_gaps,length(col_upperz_idx_jj)];
                assert(length(lower_group_idx)==length(upper_group_idx))
                uu=ll; % we keep these two in sync
                col_upperz_idx=col_upperz_idx_jj(upper_group_idx(uu)+1:upper_group_idx(uu+1));
                upperz_values=upperz_values_jj(upper_group_idx(uu)+1:upper_group_idx(uu+1));

                multiplierz_upper=col_upperz_idx-col_lowerz_idx(1)+1;
    
                sum_lowerz=sum(lowerz_values.*multiplierz_lower);
                sum_upperz=sum(upperz_values.*multiplierz_upper);
                starting_zeros=sum(lowerz_values==0)+sum(upperz_values==0);
    
                if length(unique(multiplierz_lower))>1 && (sum_upperz+sum_lowerz)/multiplierz_lower(end)<=probability_row
                    % We can fit all the upper values into slots allocated to lower with a basis to work with
                    lowerz_values(end)=lowerz_values(end)+sum_upperz/multiplierz_lower(end);
                    % But in so doing, we may have probabilities that sum>1, so fix
                    zero_created=false;
                    if length(lowerz_values)>2
                        next_candidate=length(lowerz_values);
                        zero_candidate=zeros(1,next_candidate);
                        zero_candidate(next_candidate)=1;
                        while nnz(lowerz_values)>1
                            % Aggressively try to zero out largest indices
                            new_values=linsolve([multiplierz_lower;ones(1,length(lowerz_values));zero_candidate],[sum(lowerz_values.*multiplierz_lower); probability_row; 0])';
                            new_values=round(new_values,epsilon_round);
                            if all(new_values==lowerz_values) || any(new_values<0)
                                break
                            end
                            lowerz_values=new_values;
                            zero_created=true;
                            next_candidate=find(zero_candidate==0,1,'last');
                            zero_candidate(next_candidate)=1;
                        end
                        if zero_created
                            zero_candidate(next_candidate)=0;
                        end
                        next_candidate=1;
                        zero_candidate(next_candidate)=1;
                        while nnz(lowerz_values)>1
                            % Try to zero out least index
                            new_values=linsolve([multiplierz_lower;ones(1,length(lowerz_values));zero_candidate],[sum(lowerz_values.*multiplierz_lower); probability_row; 0])';
                            new_values=round(new_values,epsilon_round);
                            if all(new_values==lowerz_values) || any(new_values<0) || any(isnan(new_values))
                                break
                            end
                            lowerz_values=new_values;
                            zero_created=true;
                            next_candidate=find(zero_candidate==0,1,'first');
                            zero_candidate(next_candidate)=1;
                        end
                    end
                    if ~zero_created
                        % Just re-balance the indices (possibly creating a zero in the middle we cannot move to either end of lowerz_values
                        new_values=linsolve([multiplierz_lower;ones(1,length(lowerz_values))],[sum(lowerz_values.*multiplierz_lower); probability_row])';
                        new_values=round(new_values,epsilon_round);
                        if any(new_values<0)
                            break
                        end
                        lowerz_values=new_values;
                    end
                    temp=sparse(row,col_lowerz_idx,lowerz_values,N_a1,N_a2);
                    temp_cols=col_lowerz_idx(1):col_upperz_idx(end);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
                    total_zeros_created=total_zeros_created+sum(lowerz_values==0)+length(upperz_values)-starting_zeros;
                    continue
                elseif length(unique(multiplierz_upper))>1 && sum_lowerz/multiplierz_upper(end-1)+sum(upperz_values)<=probability_row
                    % We can fit all the lower values into slots allocated to upper with a basis to work with
                    upperz_values(end-1)=upperz_values(end-1)+sum_lowerz/multiplierz_upper(end-1);
                    % But in so doing, we may have probabilities that sum>1, so fix
                    zero_created=false;
                    if length(upperz_values)>2
                        next_candidate=length(upperz_values);
                        zero_candidate=zeros(1,next_candidate);
                        zero_candidate(next_candidate)=1;
                        while nnz(upperz_values)>1
                            % Aggressively try to zero out largest indices
                            new_values=linsolve([multiplierz_upper;ones(1,length(upperz_values));zero_candidate],[sum(upperz_values.*multiplierz_upper); probability_row; 0])';
                            new_values=round(new_values,epsilon_round);
                            if all(new_values==upperz_values) || any(new_values<0) || any(isnan(new_values))
                                break
                            end
                            upperz_values=new_values;
                            zero_created=true;
                            next_candidate=find(zero_candidate==0,1,'last');
                            zero_candidate(next_candidate)=1;
                        end
                        if zero_created
                            zero_candidate(next_candidate)=0;
                        end
                        next_candidate=1;
                        zero_candidate(next_candidate)=1;
                        while nnz(upperz_values)>1
                            % Try to zero out least index
                            new_values=linsolve([multiplierz_upper;ones(1,length(upperz_values));zero_candidate],[sum(upperz_values.*multiplierz_upper); probability_row; 0])';
                            new_values=round(new_values,epsilon_round);
                            if all(new_values==upperz_values) || any(new_values<0)
                                break
                            end
                            upperz_values=new_values;
                            zero_created=true;
                            next_candidate=find(zero_candidate==0,1,'first');
                            zero_candidate(next_candidate)=1;
                        end
                    end
                    if ~zero_created
                        % Just re-balance the indices (possibly creating a zero in the middle we cannot move to either end of lowerz_values
                        new_values=linsolve([multiplierz_upper;ones(1,length(upperz_values))],[sum(upperz_values.*multiplierz_upper); probability_row])';
                        new_values=round(new_values,epsilon_round);
                        if any(new_values<0)
                            break
                        end
                        upperz_values=new_values;
                    end
                    temp=sparse(row,col_upperz_idx,upperz_values,N_a1,N_a2);
                    temp_cols=col_lowerz_idx(1):col_upperz_idx(end);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
                    total_zeros_created=total_zeros_created+sum(upperz_values==0)+length(lowerz_values)-starting_zeros;
                    continue
                elseif length(unique(multiplierz_upper))>2
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

end
