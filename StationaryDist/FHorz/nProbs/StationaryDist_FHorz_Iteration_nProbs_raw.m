function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_z,N_j,pi_z_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these N_probs.

precision=underlyingType(jequaloneDistKron);
cast2precision=str2func(precision);

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
needs_rounding=(PolicyProbs<1e-7 | PolicyProbs>1-1e-7);
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

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprimez(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices
    Gammatranspose_lower=sparse(Policy_aprimez(:,1,jj),II1,PolicyProbs(:,1,jj),N_a*N_z,N_a*N_z);
    Gammatranspose_upper=sparse(Policy_aprimez(:,2,jj),II1,PolicyProbs(:,2,jj),N_a*N_z,N_a*N_z);

    % First step of Tan improvement
    needs_rounding=full(StationaryDist_jj<1e-7 | StationaryDist_jj>1-1e-7);
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
            if any(noise_vals)
                temp_cols=col_lowerz_idx_jj(noise_vals);
                col_lowerz_idx_jj=col_lowerz_idx_jj(~noise_vals);
                lowerz_values_jj=lowerz_values_jj(~noise_vals);
                temp=sparse(row,col_lowerz_idx_jj,lowerz_values_jj,N_a1,N_a2);
                StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
            end

            if isempty(col_upperz_idx_jj)
                if isscalar(col_lowerz_idx_jj) && probability_row<1e-5 && (col_lowerz_idx_jj==1 || col_lowerz_idx_jj==N_a2)
                    % Allow this infinitesimal to evaporate from grid
                    temp=sparse(row,col_lowerz_idx_jj,0,N_a1,N_a2);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,col_lowerz_idx_jj,z_c),z_c)=temp(row,col_lowerz_idx_jj);
                    continue
                elseif length(col_lowerz_idx_jj)<3
                    continue
                end
            else
                noise_vals=(upperz_values_jj/probability_row)<1e-5;
                noise_vals(1:find(noise_vals==0,1,'last'))=0;
                if any(noise_vals)
                    temp_cols=col_upperz_idx_jj(noise_vals);
                    col_upperz_idx_jj=col_upperz_idx_jj(~noise_vals);
                    upperz_values_jj=upperz_values_jj(~noise_vals);
                    temp=sparse(row,col_upperz_idx_jj,upperz_values_jj,N_a1,N_a2);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
                end

                if col_upperz_idx_jj(end)-col_lowerz_idx_jj(1)<2
                    continue
                end
            end

            [col_lowerz_idx_jj,sort_idx]=sort(col_lowerz_idx_jj);
            lowerz_values_jj=lowerz_values_jj(sort_idx);

            col_gaps=find(diff(col_lowerz_idx_jj)>1);
            if ~isempty(col_gaps)
                continue
            end
            lower_group_idx=[0,col_gaps,length(col_lowerz_idx_jj)];
            for ll=1:length(lower_group_idx)-1
                col_lowerz_idx=col_lowerz_idx_jj(lower_group_idx(ll)+1:lower_group_idx(ll+1));
                lowerz_values=lowerz_values_jj(lower_group_idx(ll)+1:lower_group_idx(ll+1));
    
                if col_lowerz_idx(end)-col_lowerz_idx(1)+1>length(col_lowerz_idx)
                    % A zero logically bubbled in...so make space for it for now
                    full_col_idx=col_lowerz_idx(1):col_lowerz_idx(end);
                    [tf,~]=ismember(full_col_idx,col_lowerz_idx);
                    insert_pos=find(tf==0);
                    good_cols=setdiff(1:length(full_col_idx),insert_pos);
                    new_values=zeros(length(full_col_idx),1);
                    new_values(good_cols)=lowerz_values;
                    lowerz_values=new_values;
                    col_lowerz_idx=full_col_idx;
                end
    
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

                col_gaps=find(diff(col_upperz_idx_jj)>1);
                if ~isempty(col_gaps)
                    continue
                end
                upper_group_idx=[0,col_gaps,length(col_upperz_idx_jj)];
                assert(length(lower_group_idx)==length(upper_group_idx))
                uu=ll; % we keep these two in sync
                col_upperz_idx=col_upperz_idx_jj(upper_group_idx(uu)+1:upper_group_idx(uu+1));
                upperz_values=upperz_values_jj(upper_group_idx(uu)+1:upper_group_idx(uu+1));

                if col_upperz_idx(end)-col_upperz_idx(1)+1>length(col_upperz_idx)
                    % A zero logically bubbled in...so make space for it for now
                    full_col_idx=col_upperz_idx(1):col_upperz_idx(end);
                    [tf,~]=ismember(full_col_idx,col_upperz_idx);
                    insert_pos=find(tf==0);
                    good_cols=setdiff(1:length(full_col_idx),insert_pos);
                    new_values=zeros(length(full_col_idx),1);
                    new_values(good_cols)=upperz_values;
                    upperz_values=new_values;
                    col_upperz_idx=full_col_idx;
                end
    
                multiplierz_upper=col_upperz_idx-col_lowerz_idx(1)+1;
    
                sum_lowerz=sum(lowerz_values.*multiplierz_lower);
                sum_upperz=sum(upperz_values.*multiplierz_upper);
    
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
                            new_values=round(new_values,6);
                            if all(new_values==lowerz_values)
                                break
                            elseif all(new_values>=0)
                                lowerz_values=new_values;
                                zero_created=true;
                                next_candidate=find(zero_candidate==0,1,'last');
                                zero_candidate(next_candidate)=1;
                            else
                                break
                            end
                        end
                        if zero_created
                            zero_candidate(next_candidate)=0;
                        end
                        next_candidate=1;
                        zero_candidate(next_candidate)=1;
                        while nnz(lowerz_values)>1
                            % Try to zero out least index
                            new_values=linsolve([multiplierz_lower;ones(1,length(lowerz_values));zero_candidate],[sum(lowerz_values.*multiplierz_lower); probability_row;0])';
                            new_values=round(new_values,6);
                            if all(new_values==lowerz_values)
                                break
                            elseif all(new_values>=0)
                                lowerz_values=new_values;
                                zero_created=true;
                                next_candidate=find(zero_candidate==0,1,'first');
                                zero_candidate(next_candidate)=1;
                            else
                                break
                            end
                        end
                    end
                    if ~zero_created
                        % Just re-balance the indices (possibly creating a zero in the middle we cannot move to either end of lowerz_values
                        new_values=linsolve([multiplierz_lower;ones(1,length(lowerz_values))],[sum(lowerz_values.*multiplierz_lower); probability_row])';
                        if abs(sum(new_values)-probability_row)>1e-6
                            continue
                        end
                        new_values=round(new_values,6);
                        if all(new_values>=0)
                            lowerz_values=new_values;
                        else
                            continue
                        end
                    end
                    temp=sparse(row,col_lowerz_idx,lowerz_values,N_a1,N_a2);
                    temp_cols=col_lowerz_idx(1):col_upperz_idx(end);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
                    continue
                elseif length(unique(multiplierz_upper))>1 && sum_lowerz/multiplierz_lower(end)+sum(upperz_values)<=probability_row
                    % We can fit all the lower values into slots allocated to upper with a basis to work with
                    upperz_values(1)=upperz_values(1)+sum_lowerz/multiplierz_upper(1);
                    % But in so doing, we may have probabilities that sum>1, so fix
                    zero_created=false;
                    if length(upperz_values)>2
                        next_candidate=length(upperz_values);
                        zero_candidate=zeros(1,next_candidate);
                        zero_candidate(next_candidate)=1;
                        while nnz(upperz_values)>1
                            % Aggressively try to zero out largest indices
                            new_values=linsolve([multiplierz_upper;ones(1,length(upperz_values));zero_candidate],[sum(upperz_values.*multiplierz_upper); probability_row;0])';
                            new_values=round(new_values,6);
                            if all(new_values==upperz_values)
                                break
                            elseif all(new_values>=0)
                                upperz_values=new_values;
                                zero_created=true;
                                next_candidate=find(zero_candidate==0,1,'last');
                                zero_candidate(next_candidate)=1;
                            else
                                break
                            end
                        end
                        if zero_created
                            zero_candidate(next_candidate)=0;
                        end
                        next_candidate=1;
                        zero_candidate(next_candidate)=1;
                        while nnz(upperz_values)>1
                            % Try to zero out least index
                            new_values=linsolve([multiplierz_upper;ones(1,length(upperz_values));zero_candidate],[sum(upperz_values.*multiplierz_upper); probability_row;0])';
                            new_values=round(new_values,6);
                            if all(new_values==upperz_values)
                                break
                            elseif all(new_values>=0)
                                upperz_values=new_values;
                                zero_created=true;
                                next_candidate=find(zero_candidate==0,1,'first');
                                zero_candidate(next_candidate)=1;
                            else
                                break
                            end
                        end
                    end
                    if ~zero_created
                        % Just re-balance the indices (possibly creating a zero in the middle we cannot move to either end of lowerz_values
                        new_values=linsolve([multiplierz_upper;ones(1,length(upperz_values))],[sum(upperz_values.*multiplierz_upper); probability_row])';
                        if abs(sum(new_values)-probability_row)>1e-6
                            continue
                        end
                        new_values=round(new_values,6);
                        if all(new_values>=0)
                            upperz_values=new_values;
                        else
                            continue
                        end
                    end
                    temp=sparse(row,col_upperz_idx,upperz_values,N_a1,N_a2);
                    temp_cols=col_lowerz_idx(1):col_upperz_idx(end);
                    StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c),z_c)=temp(row,temp_cols);
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

end
