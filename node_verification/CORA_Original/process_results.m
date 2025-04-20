%% Create visualizations for computed L_inf results

% We are interested in:
% 1) How many complete molecules are completely robustly verified (all atoms in a moolecule)?
% 2) How many atoms are robustly verified?

%% Process results for each model independently

% seeds = [0,1,2,3,4]; % models
seeds = [0]; %[5,6,7,8,9]; % models
epsilon = [0.0005]; %[0.005; 0.01; 0.02; 0.05];
num_features = 16
eN = length(epsilon);

% Verify one model at a time
for m=1:length(seeds)

    % get model
    modelPath = "cora_node_gcn_"+string(seeds(m)+"_"+string(num_features));
    
    % initialize vars
    atoms = zeros(eN,4);     % # robust, #unknown, # not robust/misclassified, # atoms
    
    for k = 1:eN
        
        % Load data one at a time
        load("verification_results/mat_files/verified_nodes_"+modelPath+"_eps_"+string(epsilon(k))+"_" + string(num_features) + ".mat")
    
        N = length(targets);
        for i=1:N
            
            % get result data
            res = results{i};
            n = length(res);
            rb  = sum(res==1); % robust
            unk = sum(res==2); % unknown
            nrb = sum(res==0); % not robust
            

            % atoms
            atoms(k,1) = atoms(k,1) + rb;
            atoms(k,2) = atoms(k,2) + unk;
            atoms(k,3) = atoms(k,3) + nrb;
            atoms(k,4) = atoms(k,4) + n;
    
        end
            
    end

    % Save summary
    save("verification_results/mat_files/summary_results_Linf_"+modelPath+".mat", "atoms");

    model = load("models/"+modelPath+".mat");
    
    % Create table with these values
    fileID = fopen("verification_results/summay_results_Linf_"+modelPath+".txt",'w');
    fileID = fopen("verification_results/summay_results_Linf_" + modelPath + ".txt", 'w');
    fprintf(fileID, ...
        'Summary of robustness results of GNN model with %d features and accuracy = %.4f\n\n', ...
        num_features, model.testAcc);



    % Print ATOMS table dynamically
    fprintf(fileID, '               CORA\n');
    fprintf(fileID, 'Epsilon | Robust    Unknown   Not Rob.   Count \n');
    for k = 1:eN
        fprintf(fileID, ' %8.4f | %.3f    %.3f    %.3f    %d \n', epsilon(k), atoms(k,1)/atoms(k,4), atoms(k,2)/atoms(k,4), atoms(k,3)/atoms(k,4), atoms(k,4));
    end
    fclose(fileID);

end