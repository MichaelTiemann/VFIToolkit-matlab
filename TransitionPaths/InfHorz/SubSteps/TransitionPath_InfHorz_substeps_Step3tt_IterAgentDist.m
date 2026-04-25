function [AgentDistnext]=TransitionPath_InfHorz_substeps_Step3tt_IterAgentDist(AgentDist,PolicyPath_ForAgentDistIter,PolicyProbsPath,tt,N_a,N_z,N_probs,pi_z,II1,II2,transpathoptions,simoptions)

if N_z>0
    if transpathoptions.zpathtrivial==0
        pi_z=transpathoptions.pi_z_T(:,:,tt);
    end
else
    N_z=1;
end

if N_probs==0
    AgentDistnext=AgentDist_InfHorz_TPath_SingleStep(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a,N_z,pi_z);
else
    AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II2,PolicyProbsPath(:,tt),N_a,N_z,pi_z);
end






end
