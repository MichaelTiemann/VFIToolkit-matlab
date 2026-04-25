function [PolicyIndexesPath,N_probs,II1,II2]=TransitionPath_InfHorz_substeps_Step0_setup(l_d,l_aprime,N_a,N_z,T,transpathoptions,vfoptions,simoptions)
% Note: N_d is not relevant here
% If not using d, set l_d=0 as input

N_probs=1; % Not using N_probs
if vfoptions.gridinterplayer==1
    N_probs=N_probs*2;
end
if vfoptions.experienceasset==1
    N_probs=N_probs*2;
end

if N_z==0
    % Shapes:
    % V is [N_a,1]
    % AgentDist for basic is [N_a,1]
    if vfoptions.gridinterplayer==0
        PolicyIndexesPath=zeros(l_d+l_aprime,N_a,T-1,'gpuArray'); %Periods 1 to T-1
    elseif vfoptions.gridinterplayer==1
        PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,T-1,'gpuArray'); %Periods 1 to T-1
    end
    if vfoptions.gridinterplayer==0
        II1=1:1:N_a;
        II2=ones(N_a,1);
    elseif vfoptions.gridinterplayer==1
        II1=repelem((1:1:N_a)',1,N_probs);
        II2=[];
    end
else % N_z > 0
    % Shapes:
    % V is [N_a,N_z]
    % AgentDist for basic is [N_a*N_z,1]
    if vfoptions.gridinterplayer==0
        PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
    elseif vfoptions.gridinterplayer==1
        PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
    end
    if vfoptions.gridinterplayer==0
        II1=1:1:N_a*N_z;
        II2=ones(N_a*N_z,1);
    elseif vfoptions.gridinterplayer==1
        II1=repelem((1:1:N_a*N_z)',1,N_probs);
        II2=[];
    end
end


%% Clean up output
if vfoptions.gridinterplayer==0
    N_probs=[];
end


end
