function [V,Policy3]=ValueFnIter_FHorz_EpsteinZin_SemiExo_DC1_noz_raw(n_d1,n_d2,n_a,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7, ezc8)
% Divide-and-conquer version of ValueFnIter_FHorz_EpsteinZin_SemiExo_noz_raw.
% Grafts the Epstein-Zin transforms onto ValueFnIter_FHorz_SemiExo_DC1_noz_raw:
% V' is transformed by ^ezc5 once per age (d2-independent, pointwise), the
% d2-dependent expectation is taken, then temp4 (the post-certainty-equivalent
% continuation) is pointwise in (aprime,semiz), so it is computed once per d2
% over the full aprime grid and indexed exactly where the vNM code indexes
% EV_d2; the return transform and the final ^ezc7 wrap each level's entireRHS
% before its max (a monotone transform, so the divide-and-conquer monotonicity
% logic is unaffected). The final max over d2 then compares fully-transformed
% values, so it is unaffected.

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=prod(n_d); % Needed for N_j when converting to form of Policy3
N_a=prod(n_a);
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d,d2,aprime seperately
Policy3=zeros(3,N_a,N_semiz,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];

d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]); % version to use when looping over d2

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1); % already includes -1
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-2); % already includes -1

% Preallocate
V_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
if vfoptions.EZoneminusbeta==1
    ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
elseif vfoptions.EZoneminusbeta==2
    ezc1=1-sj(N_j)*DiscountFactorParamsVec;
end

% If there is a warm-glow at end of the final period, evaluate the warmglowfn
if warmglow==1
    WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,N_j);
    WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
    WGmatrix=WGmatrixraw;
    WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(N_j);
    WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
    if ~isfield(vfoptions,'V_Jplus1')
        becareful=(WGmatrix==0);
        WGmatrix(isfinite(WGmatrix))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGmatrix(isfinite(WGmatrix)).^ezc8(N_j)).^ezc6(N_j));
        WGmatrix(becareful)=0;
    end
    % WGmatrix is a column over the full aprime grid; it is indexed by aprime below
else
    WGmatrix=zeros(N_a,1,'gpuArray');
end

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            % Modify the Return Function appropriately for Epstein-Zin Preferences
            becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
            ReturnMatrix_d2ii(becareful)=(ezc1*ReturnMatrix_d2ii(becareful).^ezc2(N_j)).^ezc7(N_j);
            ReturnMatrix_d2ii(ReturnMatrix_d2ii==0)=-Inf;
            entireRHS_ii=ReturnMatrix_d2ii+shiftdim(WGmatrix,-1); % warm-glow (zero if not using); (d,aprime,a,semiz)

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1); % d,aprime


            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrix(aprimeindexes),[N_d1*(maxgap(ii)+1),1,N_semiz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                    ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                    entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrix(loweredge),[N_d1,1,N_semiz]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
                end
            end
        end

    elseif vfoptions.lowmemory>=1 % loop semiz

        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,1);
                % Modify the Return Function appropriately for Epstein-Zin Preferences
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                ReturnMatrix_d2ii(becareful)=(ezc1*ReturnMatrix_d2ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                ReturnMatrix_d2ii(ReturnMatrix_d2ii==0)=-Inf;
                entireRHS_ii=ReturnMatrix_d2ii+shiftdim(WGmatrix,-1); % warm-glow (zero if not using); (d,aprime,a)

                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n]),[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex2,1); % d,aprime


                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                        entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrix(aprimeindexes),[N_d1*(maxgap(ii)+1),1]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        ReturnMatrix_ii(becareful)=(ezc1*ReturnMatrix_ii(becareful).^ezc2(N_j)).^ezc7(N_j);
                        ReturnMatrix_ii(ReturnMatrix_ii==0)=-Inf;
                        entireRHS_ii=ReturnMatrix_ii+reshape(WGmatrix(loweredge),[N_d1,1]);
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);    % First, switch V_Jplus1 into Kron form

    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=V_Jplus1;
    temp(isfinite(V_Jplus1))=(ezc4*V_Jplus1(isfinite(V_Jplus1))).^ezc5(N_j);
    temp(V_Jplus1==0)=0;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j); % reverse order

            EV_d2=temp.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,semiz)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrixbig(becareful).^ezc8(N_j)).^ezc6(N_j);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                temp4(EV_d2==0)=0;
            end

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
            becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
            temp2_ii(ReturnMatrix_d2ii==0)=-Inf;

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1); % (d,aprime,a,semiz)

            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_ii(entireRHS_ii==0)=-Inf;

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    aprimez=aprimeindexes+N_a*semizBind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    aprimez=loweredge+N_a*semizBind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
                end
            end
        end

    elseif vfoptions.lowmemory>=1 % loop semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j); % reverse order

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,N_j);

                % Calc the condl expectation term (except beta), which depends on semiz but not on control variables
                EV_d2z=temp.*pi_semiz(semiz_c,:);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % sum over semiz', leaving a singular second dimension

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
                temp4=EV_d2z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(N_j)*temp4(becareful).^ezc8(N_j)+(1-sj(N_j))*WGmatrix(becareful).^ezc8(N_j)).^ezc6(N_j);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(N_j)*temp4(isfinite(temp4)).^ezc8(N_j)).^ezc6(N_j);
                    temp4(EV_d2z==0)=0;
                end

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,1);
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(N_j);
                temp2_ii(ReturnMatrix_d2ii==0)=-Inf;

                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1); % (d,aprime,a)

                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                entireRHS_ii(entireRHS_ii==0)=-Inf;

                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n]),[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex2,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        aprimez=aprimeindexes; % single semiz
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                        entireRHS_ii(entireRHS_ii==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(N_j);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        aprimez=loweredge; % single semiz
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(N_j);
                        entireRHS_ii(entireRHS_ii==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,N_j)=V_jj;
    Policy3(2,:,:,N_j)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,N_j)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,N_j)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end


    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec; % Just in case it depends on age
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        % WGmatrix is a column over the full aprime grid; combined into temp4 below
    end

    EVpre=V(:,:,jj+1);
    % Part of Epstein-Zin is before taking expectation (d2-independent, so done once)
    temp=EVpre;
    temp(isfinite(EVpre))=(ezc4*EVpre(isfinite(EVpre))).^ezc5(jj);
    temp(EVpre==0)=0;

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,jj); % reverse order

            EV_d2=temp.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV_d2=sum(EV_d2,2); % sum over z', leaving a singular second dimension

            % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over (aprime,semiz)
            temp4=EV_d2;
            if warmglow==1
                WGmatrixbig=WGmatrix.*ones(1,1,N_semiz);
                becareful=logical(isfinite(temp4).*isfinite(WGmatrixbig)); % both are finite
                temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrixbig(becareful).^ezc8(jj)).^ezc6(jj);
                temp4((EV_d2==0)&(WGmatrixbig==0))=0; % Is actually zero
            else % not using warmglow
                temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                temp4(EV_d2==0)=0;
            end

            % n-Monotonicity
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
            becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
            temp2_ii=ReturnMatrix_d2ii;
            temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
            temp2_ii(ReturnMatrix_d2ii==0)=-Inf;

            entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1); % (d,aprime,a,semiz)

            temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
            entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
            entireRHS_ii(entireRHS_ii==0)=-Inf;

            % First, we want aprime conditional on (d,1,a,z)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n,N_semiz]),[],1);

            % Store
            V_ford2_jj(level1ii,:,d2_c)=shiftdim(Vtempii,1);
            Policy_ford2_jj(level1ii,:,d2_c)=shiftdim(maxindex2,1); % d,aprime

            % Second level based on monotonicity
            maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-n_semiz
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_semiz
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    aprimez=aprimeindexes+N_a*semizBind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1*(maxgap(ii)+1),1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
                else
                    loweredge=maxindex1(:,1,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                    becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                    temp2_ii=ReturnMatrix_ii;
                    temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                    temp2_ii(ReturnMatrix_ii==0)=-Inf;
                    aprimez=loweredge+N_a*semizBind;
                    entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,1,N_semiz]); % autoexpand level1iidiff(ii) in 2nd-dim
                    temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                    entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                    entireRHS_ii(entireRHS_ii==0)=-Inf;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V_ford2_jj(curraindex,:,d2_c)=shiftdim(Vtempii,1);
                    dind=(rem(maxindex-1,N_d1)+1);
                    allind=dind+N_d1*semizind; % loweredge is n_d-by-1-by-1-by-n_semiz
                    Policy_ford2_jj(curraindex,:,d2_c)=shiftdim(maxindex+N_d1*(loweredge(allind)-1)); % loweredge(given the d and z)
                end
            end
        end

    elseif vfoptions.lowmemory>=1 % loop semiz
        for d2_c=1:N_d2
            d12c_gridvals=d12_gridvals(:,:,d2_c);
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_semiz=pi_semiz_J(:,:,d2_c,jj); % reverse order

            for semiz_c=1:N_semiz
                semiz_val=semiz_gridvals_J(semiz_c,:,jj);

                % Calc the condl expectation term (except beta), which depends on semiz but not on control variables
                EV_d2z=temp.*pi_semiz(semiz_c,:);
                EV_d2z(isnan(EV_d2z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
                EV_d2z=sum(EV_d2z,2); % sum over semiz', leaving a singular second dimension

                % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise over aprime
                temp4=EV_d2z;
                if warmglow==1
                    becareful=logical(isfinite(temp4).*isfinite(WGmatrix)); % both are finite
                    temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGmatrix(becareful).^ezc8(jj)).^ezc6(jj);
                    temp4((EV_d2z==0)&(WGmatrix==0))=0; % Is actually zero
                else % not using warmglow
                    temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
                    temp4(EV_d2z==0)=0;
                end

                % n-Monotonicity
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid, a_grid(level1ii), semiz_val, ReturnFnParamsVec,1);
                becareful=logical(isfinite(ReturnMatrix_d2ii).*(ReturnMatrix_d2ii~=0)); % finite but not zero
                temp2_ii=ReturnMatrix_d2ii;
                temp2_ii(becareful)=ReturnMatrix_d2ii(becareful).^ezc2(jj);
                temp2_ii(ReturnMatrix_d2ii==0)=-Inf;

                entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*shiftdim(temp4,-1); % (d,aprime,a)

                temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                entireRHS_ii(entireRHS_ii==0)=-Inf;

                % First, we want aprime conditional on (d,1,a,z)
                [~,maxindex1]=max(entireRHS_ii,[],2);

                % Now, get and store the full (d,aprime)
                [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d1*N_a,vfoptions.level1n]),[],1);

                % Store
                V_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(Vtempii,1);
                Policy_ford2_jj(level1ii,semiz_c,d2_c)=shiftdim(maxindex2,1); % d,aprime

                % Second level based on monotonicity
                maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                        % loweredge is n_d-by-1
                        aprimeindexes=loweredge+(0:1:maxgap(ii));
                        % aprime possibilities are n_d-by-maxgap(ii)+1
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        aprimez=aprimeindexes; % single semiz
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1*(maxgap(ii)+1),1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                        entireRHS_ii(entireRHS_ii==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    else
                        loweredge=maxindex1(:,1,ii,:);
                        % Just use aprime(ii) for everything
                        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), semiz_val, ReturnFnParamsVec,2);
                        becareful=logical(isfinite(ReturnMatrix_ii).*(ReturnMatrix_ii~=0)); % finite but not zero
                        temp2_ii=ReturnMatrix_ii;
                        temp2_ii(becareful)=ReturnMatrix_ii(becareful).^ezc2(jj);
                        temp2_ii(ReturnMatrix_ii==0)=-Inf;
                        aprimez=loweredge; % single semiz
                        entireRHS_ii=ezc1*temp2_ii+ezc3*DiscountFactorParamsVec*reshape(temp4(aprimez),[N_d1,1]); % autoexpand level1iidiff(ii) in 2nd-dim
                        temp5=logical(isfinite(entireRHS_ii).*(entireRHS_ii~=0));
                        entireRHS_ii(temp5)=entireRHS_ii(temp5).^ezc7(jj);
                        entireRHS_ii(entireRHS_ii==0)=-Inf;
                        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                        V_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(Vtempii,1);
                        dind=(rem(maxindex-1,N_d1)+1);
                        allind=dind; % single semiz
                        Policy_ford2_jj(curraindex,semiz_c,d2_c)=shiftdim(maxindex+N_d1*(reshape(loweredge(allind),size(maxindex))-1)); % loweredge(given the d)
                    end
                end
            end
        end

    end
    % Now we just max over d2, and keep the policy that corresponded to that (including modify the policy to include the d2 decision)
    [V_jj,maxindex]=max(V_ford2_jj,[],3); % max over d2
    V(:,:,jj)=V_jj;
    Policy3(2,:,:,jj)=shiftdim(maxindex,-1); % d2 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d1aprime_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy3(1,:,:,jj)=shiftdim(rem(d1aprime_ind-1,N_d1)+1,-1);
    Policy3(3,:,:,jj)=shiftdim(ceil(d1aprime_ind/N_d1),-1);

end


end
