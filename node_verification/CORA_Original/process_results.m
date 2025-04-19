%% Create visualizations for computed L_inf results

% We are interested in:
% 1) How many complete molecules are completely robustly verified (all atoms in a moolecule)?
% 2) How many atoms are robustly verified?

%% Process results for each model independently

% seeds = [0,1,2,3,4]; % models
seeds = [1]; % models
epsilon = [0.0005] %; 0.01; 0.02; 0.05];
eN = length(epsilon);

% Verify one model at a time
for m=1:length(seeds)

    % get model
    modelPath = "cora_node_gcn_"+string(seeds(m));
    
    % initialize vars
    samples = zeros(eN,4);

    for k = 1:eN
        % Load data one at a time
        load("verification_results/mat_files/verified_nodes_"+modelPath+"_eps_"+string(epsilon(k))+".mat"); 

        N = length(targets);
        for i=1:N
            % get result data
            res = results{i};
            rb  = sum(res==1); % robust
            unk = sum(res==2); % unknown
            nrb = sum(res==0); % not robust

            % atoms
            samples(k,1) = samples(k,1) + rb;
            samples(k,2) = samples(k,2) + unk;
            samples(k,3) = samples(k,3) + nrb;
            samples(k,4) = samples(k,4) + length(res);
        end
    end

    % Save summary
    save("verification_results/mat_files/summary_results_"+modelPath+".mat", "samples");

    model = load("models/"+modelPath+".mat");
    
    % Create table with these values
    fileID = fopen("verification_results/summary_results_"+modelPath+".txt",'w');
    fprintf(fileID, 'Summary of robustness results of CORA gnn model with accuracy = %.4f \n\n', model.testAcc);
    fprintf(fileID,'                 Compromised \n');
    fprintf(fileID, 'Epsilon | Robust  Unknown  Not Rob.  N \n');
    fprintf(fileID, '  0.005 | %.3f    %.3f   %.3f   %d \n', samples(1,1)/samples(1,4), samples(1,2)/samples(1,4), samples(1,3)/samples(1,4), samples(1,4));
    % fprintf(fileID, '   0.01 | %.3f    %.3f   %.3f   %d \n', samples(2,1)/samples(2,4), samples(2,2)/samples(2,4), samples(2,3)/samples(2,4), samples(2,4));
    % fprintf(fileID, '   0.02 | %.3f    %.3f   %.3f   %d \n', samples(3,1)/samples(3,4), samples(3,2)/samples(3,4), samples(3,3)/samples(3,4), samples(3,4));
    % fprintf(fileID, '   0.05 | %.3f    %.3f   %.3f   %d \n', samples(4,1)/samples(4,4), samples(4,2)/samples(4,4), samples(4,3)/samples(4,4), samples(4,4));
    fclose(fileID);

end