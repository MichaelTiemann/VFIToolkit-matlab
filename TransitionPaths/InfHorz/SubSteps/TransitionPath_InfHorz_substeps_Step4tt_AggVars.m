function AggVars=TransitionPath_InfHorz_substeps_Step4tt_AggVars(AgentDist,PolicyValuesPath_tt,tt,FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,n_a,n_z,a_gridvals,z_gridvals,transpathoptions)
% Maybe should just take PolicyValuesPath_tt as input instead of PolicyValuesPath

% Note we are passing `AggVarNames` into a function expecting `FnsToEvaluateNames`.  The aggvar names are indeed identical to their function's names.
AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(AgentDist, PolicyValuesPath_tt, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, AggVarNames, n_a, n_z, a_gridvals, z_gridvals, 1)





end
