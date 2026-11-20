function [VKron, Policy]=ValueFnIter_InfHorz_raw(VKron, n_d,n_a,n_z, pi_z, beta, ReturnMatrix, Howards,Howards2, Tolerance, maxiter)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% Policy=zeros(N_a,N_z,'gpuArray');
% Ftemp=zeros(N_a,N_z,'gpuArray');

pi_z_T=pi_z';
has_piz_pos=(pi_z_T>0);
pi_z_howards=repelem(pi_z,N_a,1);

addindexforaz=gpuArray(shiftdim((0:1:N_a-1)*N_d*N_a+shiftdim((0:1:N_z-1)*N_d*N_a*N_a,-1),1));

if N_d>1
    RM_4D=reshape(ReturnMatrix,[N_d,N_a,N_a,N_z]);
end

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    % Calc the condl expectation term (except beta), which depends on z but not on control variables
    Vinf=(VKronold==-Inf);
    if any(Vinf(:))
        Vclean=VKronold;
        Vclean(Vinf)=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV=Vclean*pi_z_T; % sum over z', leaving a singular second dimension
        EV(Vinf*has_piz_pos>0)=-Inf;
    else
        EV=VKronold*pi_z_T; % sum over z', leaving a singular second dimension
    end

    if N_d>1
        entireRHS=reshape(RM_4D+reshape(beta*EV,[1,N_a,1,N_z]),[N_d*N_a,N_a,N_z]); % z and d dimensions of EV will autoexpand without repelem
    else
        entireRHS=ReturnMatrix+reshape(beta*EV,[N_a,1,N_z]); % z dimension of EV will autoexpand
    end

    % Calc the max and it's index
    [VKron,PolicyIndexes]=max(entireRHS,[],1);

    PolicyIndexes=shiftdim(PolicyIndexes,1);

    tempmaxindex=PolicyIndexes+addindexforaz; % aprime index, add the index for a and z
    Ftemp=ReturnMatrix(tempmaxindex); % keep return function of optimal policy for using in Howards

    VKron=shiftdim(VKron,1); % a by z

    % Update currdist
    VKrondist=VKron(:)-VKronold(:);
    VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        Policy_aprime=ceil(PolicyIndexes(:)/N_d);
        if ~any(~isfinite(Ftemp(:))) && isfinite(currdist)
            for Howards_counter=1:Howards
                EVKrontemp=reshape(sum(VKron(Policy_aprime,:).*pi_z_howards,2),[N_a,N_z]);
                VKron=Ftemp+beta*EVKrontemp;
            end
        else
            for Howards_counter=1:Howards
                EVKrontemp=VKron(Policy_aprime,:);

                EVKrontemp=EVKrontemp.*pi_z_howards;
                EVKrontemp(isnan(EVKrontemp))=0;
                EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);
                VKron=Ftemp+beta*EVKrontemp;
            end
        end
    end

    tempcounter=tempcounter+1;

end


Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

if currdist > Tolerance
    warning(['Value fn iteration has stopped due to reaching the maximum number of iterations ', ...
        '(not due to convergence); can be set by vfoptions.maxiter. ', ...
        'Last currdist = %.16g; tolerance = %.16g.'], ...
        currdist, Tolerance)
end


end
