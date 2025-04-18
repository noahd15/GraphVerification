setenv('AV_PROJECT_HOME', 'C:\Users\Noah\OneDrive - Vanderbilt\Spring 2025\CS 6315\Project\AV_Project')

% 1) Train
run("batched_CORA_Modified_new.m");

% 2) Reachability
run("reach_all_models_CORA");

% 3) Verify
run("verify_AllReachSets");

% 4) Process results
run("process_results");

