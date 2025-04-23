seeds = [0,1,2];                          % Model seeds
epsilon = [.00005, .0005, 0.005];           % Epsilon values
num_features = 16;
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
    modelPath = "drone_node_gcn_pca_" + string(seed);

    % Load model test accuracy once per seed
    mdl = load("models/" + modelPath + ".mat");
    testAcc = mdl.testAcc;
    testAccs(m) = testAcc;

    atoms = zeros(eN, 4);

    for k = 1:eN
        % Load verification results
        matFile = "verification_results/mat_files/verified_nodes_" + modelPath + ...
                  "_eps_" + string(epsilon(k))  + ".mat";
        load(matFile, 'results', 'targets');

        N = numel(targets);
        for i = 1:N
            res = results{i};
            atoms(k,1) = atoms(k,1) + sum(res == 1);  % robust
            atoms(k,2) = atoms(k,2) + sum(res == 2);  % unknown
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

% Ensure output directory exists
outDir = "verification_results/mat_files";
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Save summary
save("verification_results/mat_files/summary_all_Linf_dropout_16.mat", ...
     'allAtoms', 'epsilon', 'seeds', 'num_features', 'testAccs');

fprintf("Summary written to:\n- %s\n- %s\n", ...
    combinedTxtFile, "verification_results/mat_files/summary_all_Linf_dropout_"+num_features+".mat");
