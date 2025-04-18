% 1) Train
%run("batched_train_multiple_reluGCN_PCA.m");

% 2) Reachability
%run("reach_all_models_Linf");

% 3) Verify
run("verify_AllReachSets");

% 4) Process results
run("process_results");

