%% Combined summary of robustness results across seeds and epsilons

seeds = [0, 1, 2];                          % Model seeds
epsilon = [.00005, .0005, .005];                  % Epsilon values
num_features = 16;
eN = length(epsilon);
numSeeds = length(seeds);

% Preallocate 3D matrix: [#epsilons x 4 metrics x #seeds]
% Metrics: [robust, unknown, not robust, total]
allAtoms = zeros(eN, 4, numSeeds);

% Also track testAcc per model
testAccs = zeros(numSeeds, 1);

% Open combined output file
combinedTxtFile = "verification_results/summary_all_Linf.txt";
fileID = fopen(combinedTxtFile, 'w');
fprintf(fileID, ...
    'Summary of robustness across all GNN models with dropout with %d features\n\n', num_features);
fprintf(fileID, '%-6s %-10s %-10s %-10s %-10s %-10s\n', ...
    'Seed', 'Epsilon', 'Robust', 'Unknown', 'NotRob', 'Total');

for m = 1:numSeeds
    seed = seeds(m);
    modelPath = "cora_node_gcn_" + string(seed) + "_" + string(num_features);

    atoms = zeros(eN, 4);

    for k = 1:eN
        % Load verification results
        matFile = "verification_results/mat_files/verified_nodes_" + modelPath + ...
                  "_eps_" + string(epsilon(k)) + "_" + string(num_features) + ".mat";
        load(matFile, 'results', 'targets');  % assumes both exist

        N = length(targets);
        for i = 1:N
            res = results{i};
            n = length(res);
            rb  = sum(res == 1); % robust
            unk = sum(res == 2); % unknown
            nrb = sum(res == 0); % not robust

            atoms(k, 1) = atoms(k, 1) + rb;
            atoms(k, 2) = atoms(k, 2) + unk;
            atoms(k, 3) = atoms(k, 3) + nrb;
            atoms(k, 4) = atoms(k, 4) + n;
        end

        % Print to combined file (just seed + epsilon)
        fprintf(fileID, '%-6d %8.4f   %.3f     %.3f     %.3f     %d\n', ...
            seed, epsilon(k), ...
            atoms(k, 1) / atoms(k, 4), ...
            atoms(k, 2) / atoms(k, 4), ...
            atoms(k, 3) / atoms(k, 4), ...
            atoms(k, 4));
    end

    % Save per-model atom counts
    allAtoms(:, :, m) = atoms;

    % Load model accuracy
    model = load("models/" + modelPath + ".mat");
    testAccs(m) = model.testAcc;
end

fclose(fileID);

% Save entire summary to a single .mat file
save("verification_results/mat_files/summary_all_Linf.mat", ...
     'allAtoms', 'epsilon', 'seeds', 'num_features', 'testAccs');

fprintf("Summary written to:\n- %s\n- %s\n", ...
    combinedTxtFile, "verification_results/mat_files/summary_all_Linf.mat");
