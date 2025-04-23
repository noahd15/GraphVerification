seeds = [0, 1, 2];                          % Model seeds
epsilon = [.00005, .0005, 0.005, .05, .5];           % Epsilon values
num_features = size(X_full, 2);             % Number of features
eN = length(epsilon);
numSeeds = length(seeds);

% Preallocate 3D matrix: [#epsilons x 4 metrics x #seeds]
allAtoms = zeros(eN, 4, numSeeds);
testAccs = zeros(numSeeds, 1);

% Open combined output file
combinedTxtFile = "verification_results/summary_all_Linf.txt";
fileID = fopen(combinedTxtFile, 'w');
fprintf(fileID, ...
    'Summary of robustness across all GNN models with dropout with %d features\n\n', num_features);
% Add an “Acc” column
fprintf(fileID, '%-6s %-10s %-10s %-10s %-10s %-10s %-6s\n', ...
    'Seed', 'Epsilon', 'Robust', 'Unknown', 'NotRob', 'Total', 'Acc');

for m = 1:numSeeds
    seed = seeds(m);
    modelPath = "karate_node_gcn_" + string(seed) + "_" + string(num_features);

    % Load model test accuracy once per seed
    mdl = load("models/" + modelPath + ".mat");
    testAcc = mdl.testAcc;
    testAccs(m) = testAcc;
    % modelPath = "cora_node_gcn_" + string(seed) + "_" + string(num_features);
    atoms = zeros(eN, 4);

    for k = 1:eN
        % Load verification results
        matFile = "verification_results/mat_files/verified_nodes_" + modelPath + ...
                  "_eps_" + string(epsilon(k)) + "_" + string(num_features) + ".mat";
        load(matFile, 'results', 'targets');

        N = numel(targets);
        for i = 1:N
            res = results{i};
            % Debug: print unique values in res
            disp(['Seed: ', num2str(seed), ', Epsilon: ', num2str(epsilon(k)), ', Node: ', num2str(i), ', Unique res: ', mat2str(unique(res))]);
            atoms(k,1) = atoms(k,1) + sum(res == 1);  % robust
            atoms(k,2) = atoms(k,2) + sum(res == 5);  % unknown
            atoms(k,3) = atoms(k,3) + sum(res == 0);  % not robust
            atoms(k,4) = atoms(k,4) + numel(res);     % total
        end
        
        % Print a row with accuracy at the end
        fprintf(fileID, '%-6d %8.5f   %.3f     %.3f     %.3f     %4d   %.3f\n', ...
            seed, epsilon(k), ...
            atoms(k,1)/atoms(k,4), ...
            atoms(k,2)/atoms(k,4), ...
            atoms(k,3)/atoms(k,4), ...
            atoms(k,4), ...
            testAcc);
    end

    allAtoms(:,:,m) = atoms;
end

fclose(fileID);

% Save summary
save("verification_results/mat_files/summary_all_Linf_dropout.mat", ...
     'allAtoms', 'epsilon', 'seeds', 'num_features', 'testAccs');

fprintf("Summary written to:\n- %s\n- %s\n", ...
    combinedTxtFile, "verification_results/mat_files/summary_all_Linf_dropout.mat");
