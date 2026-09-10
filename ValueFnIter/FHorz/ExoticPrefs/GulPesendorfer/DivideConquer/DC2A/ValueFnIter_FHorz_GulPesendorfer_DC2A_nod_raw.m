function [V, Policy]=ValueFnIter_FHorz_GulPesendorfer_DC2A_nod_raw(n_a,n_z,N_j, a_grid, z_gridvals_J,pi_z_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer with divide-and-conquer on a1 (a2prime fully scanned). The tempted objective
% u+v+beta*EV goes through the standard DC machinery (so the windows bracket the argmax of the
% TEMPTED objective); the most-tempting term is a max over the FULL joint (a1prime,a2prime) choice
% (never over a window), computed from full-column temptation matrices one a1-slab at a time,
% and subtracted from V after the max.
% divide-and-conquer in the first endo state
% lowmemory: =0 vectorize over z, =1 loop over z

N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); % joint (a1prime,a2prime) index at each (a,z,j) cell

%%
n_a1=n_a(1);
n_a2=n_a(2:end);
N_a1=n_a1;
N_a2=n_a2;
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
level1ii=round(linspace(1,N_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% precompute
a2ind=gpuArray(0:1:N_a2-1); % already includes -1
if vfoptions.lowmemory==0
    zind=shiftdim(gpuArray(0:1:N_z-1),-1); % already includes -1
    zBind=shiftdim(gpuArray(0:1:N_z-1),-3); % already includes -1
elseif vfoptions.lowmemory>=1
    special_n_z=ones(1,length(n_z),'gpuArray');
end

%% j=N_j
% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);
TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
        MostTempting1=max(reshape(TemptationMatrix_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1); % full joint (a1prime,a2prime) grid at the level-1 stations

        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii;

        % size(ReturnMatrix_ii) % (aprime, a,z)
        % [n_a,vfoptions.level1n,n_z]

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii-MostTempting1,1);
        Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            % Most-tempting term over the FULL joint (a1prime,a2prime) grid for these a (never just the window)
            TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid, a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
            MostTempting_ii=max(reshape(TemptationMatrix_full,[N_a1*N_a2,level1iidiff(ii)*N_a2,N_z]),[],1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii-MostTempting_ii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*zind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii;
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii-MostTempting_ii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*zind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_vals=z_gridvals_J(z_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsVec,1);
            TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, TemptationFnParamsVec,1);
            MostTempting1=max(reshape(TemptationMatrix_ii_z,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1); % full joint (a1prime,a2prime) grid at the level-1 stations

            entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z;

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V(curraindex,z_c,N_j)=shiftdim(Vtempii-MostTempting1,1);
            Policy(curraindex,z_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                % Most-tempting term over the FULL joint (a1prime,a2prime) grid for these a (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,1);
                MostTempting_ii=max(reshape(TemptationMatrix_full,[N_a1*N_a2,level1iidiff(ii)*N_a2]),[],1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                    TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii-MostTempting_ii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                    a2primeind=ceil(maxindex/(maxgap(ii)+1));
                    maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                    TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z;
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii-MostTempting_ii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    a1primeind=1;
                    a2primeind=maxindex;
                    maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                end
            end
        end
    end

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=reshape(vfoptions.V_Jplus1,[N_a,N_z]); % Using V_Jplus1

    EV=EV.*shiftdim(pi_z_J(:,:,N_j)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);  % autoexpand (a,z)

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
        MostTempting1=max(reshape(TemptationMatrix_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1); % full joint (a1prime,a2prime) grid at the level-1 stations

        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,:,N_j)=shiftdim(Vtempii-MostTempting1,1);
        Policy(curraindex,:,N_j)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            % Most-tempting term over the FULL joint (a1prime,a2prime) grid for these a (never just the window)
            TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid, a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), TemptationFnParamsVec,1);
            MostTempting_ii=max(reshape(TemptationMatrix_full,[N_a1*N_a2,level1iidiff(ii)*N_a2,N_z]),[],1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
                aprimez=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV(reshape(aprimez,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_z]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii-MostTempting_ii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*zind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,N_j), TemptationFnParamsVec,2);
                aprimez=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV(reshape(aprimez,[1*N_a2,level1iidiff(ii)*N_a2,N_z]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,N_j)=shiftdim(Vtempii-MostTempting_ii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*zind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_vals=z_gridvals_J(z_c,:,N_j);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);
            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsVec,1);
            TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, TemptationFnParamsVec,1);
            MostTempting1=max(reshape(TemptationMatrix_ii_z,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1); % full joint (a1prime,a2prime) grid at the level-1 stations

            entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z+DiscountedEV_z;

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V(curraindex,z_c,N_j)=shiftdim(Vtempii-MostTempting1,1);
            Policy(curraindex,z_c,N_j)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                % Most-tempting term over the FULL joint (a1prime,a2prime) grid for these a (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,1);
                MostTempting_ii=max(reshape(TemptationMatrix_full,[N_a1*N_a2,level1iidiff(ii)*N_a2]),[],1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                    TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,2);
                    aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z+reshape(DiscountedEV_z(aprime),[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii-MostTempting_ii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                    a2primeind=ceil(maxindex/(maxgap(ii)+1));
                    maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                    TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,2);
                    aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z+reshape(DiscountedEV_z(aprime),[1*N_a2,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,N_j)=shiftdim(Vtempii-MostTempting_ii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    a1primeind=1;
                    a2primeind=maxindex;
                    maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                    Policy(curraindex,z_c,N_j)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                end
            end
        end
    end

end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    TemptationFnParamsVec=CreateVectorFromParams(Parameters, TemptationFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=V(:,:,jj+1);

    % Use sparse for a few lines until sum over zprime
    EV=EV.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);  % autoexpand (a,z)

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);
        TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
        MostTempting1=max(reshape(TemptationMatrix_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1); % full joint (a1prime,a2prime) grid at the level-1 stations

        entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Now, get and store the full (d,aprime)
        [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
        % Store
        curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
        V(curraindex,:,jj)=shiftdim(Vtempii-MostTempting1,1);
        Policy(curraindex,:,jj)=shiftdim(maxindex2,1);

        % Attempt for improved version
        maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
        for ii=1:(vfoptions.level1n-1)
            curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
            % Most-tempting term over the FULL joint (a1prime,a2prime) grid for these a (never just the window)
            TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid, a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,1);
            MostTempting_ii=max(reshape(TemptationMatrix_full,[N_a1*N_a2,level1iidiff(ii)*N_a2,N_z]),[],1);
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,2);
                aprimez=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV(reshape(aprimez,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_z]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii-MostTempting_ii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                a2primeind=ceil(maxindex/(maxgap(ii)+1));
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*zind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,jj)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            else
                loweredge=maxindex1(1,:,ii,:,:);
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                TemptationMatrix_ii=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals_J(:,:,jj), TemptationFnParamsVec,2);
                aprimez=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind+N_a*zBind;
                entireRHS_ii=ReturnMatrix_ii+TemptationMatrix_ii+DiscountedEV(reshape(aprimez,[1*N_a2,level1iidiff(ii)*N_a2,N_z]));
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,jj)=shiftdim(Vtempii-MostTempting_ii,1);
                % maxindex needs to be reworked:
                %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                a1primeind=1;
                a2primeind=maxindex;
                maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii))+N_a2*N_a2*zind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
                Policy(curraindex,:,jj)=shiftdim(maxindexfix+loweredge(allind)-1,1);
            end
        end

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_vals=z_gridvals_J(z_c,:,jj);
            DiscountedEV_z=DiscountedEV(:,:,:,:,z_c);

            % n-Monotonicity
            ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, ReturnFnParamsVec,1);
            TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_vals, TemptationFnParamsVec,1);
            MostTempting1=max(reshape(TemptationMatrix_ii_z,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1); % full joint (a1prime,a2prime) grid at the level-1 stations
            % (a1prime, a2prime, a1, a2 | z)

            entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z+DiscountedEV_z;

            % Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Now, get and store the full (d,aprime)
            [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2]),[],1);
            % Store
            curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem(a2ind',vfoptions.level1n,1);
            V(curraindex,z_c,jj)=shiftdim(Vtempii-MostTempting1,1);
            Policy(curraindex,z_c,jj)=shiftdim(maxindex2,1);

            % Attempt for improved version
            maxgap=squeeze(max(max(maxindex1(1,:,2:end,:)-maxindex1(1,:,1:end-1,:),[],4),[],2));
            for ii=1:(vfoptions.level1n-1)
                curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem(a2ind',level1iidiff(ii),1);
                % Most-tempting term over the FULL joint (a1prime,a2prime) grid for these a (never just the window)
                TemptationMatrix_full=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid, a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,1);
                MostTempting_ii=max(reshape(TemptationMatrix_full,[N_a1*N_a2,level1iidiff(ii)*N_a2]),[],1);
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,:,ii,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-n_a2-by-1-by-n_a2
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                    TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,2);
                    aprime=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z+reshape(DiscountedEV_z(aprime),[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii-MostTempting_ii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
                    a2primeind=ceil(maxindex/(maxgap(ii)+1));
                    maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                    Policy(curraindex,z_c,jj)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                else
                    loweredge=maxindex1(1,:,ii,:);
                    % Just use aprime(ii) for everything
                    ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(ReturnFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, ReturnFnParamsVec,2);
                    TemptationMatrix_ii_z=CreateReturnFnMatrix_Disc_DC2A_nod(TemptationFn, special_n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_vals, TemptationFnParamsVec,2);
                    aprime=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*a2ind;
                    entireRHS_ii=ReturnMatrix_ii_z+TemptationMatrix_ii_z+reshape(DiscountedEV_z(aprime),[1*N_a2,level1iidiff(ii)*N_a2]);
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,jj)=shiftdim(Vtempii-MostTempting_ii,1);
                    % maxindex needs to be reworked:
                    %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
                    a1primeind=1;
                    a2primeind=maxindex;
                    maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
                    %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
                    allind=a2primeind+N_a2*repelem(a2ind,1,level1iidiff(ii)); % loweredge is 1-by-n_a2-by-1-by-n_a2
                    Policy(curraindex,z_c,jj)=shiftdim(maxindexfix+loweredge(allind)-1,1);
                end
            end
        end
    end

end

%%
Policy=shiftdim(Policy,-1);


end
