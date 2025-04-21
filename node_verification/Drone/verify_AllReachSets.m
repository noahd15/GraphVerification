%% Verify the robustness reach sets of all models
% Adjust epsilon and seeds as needed
epsilon = [0.00005, .0005, .005, .05] %; 0.01; 0.02; 0.05];
seeds = [0,1,2];

parfor m = 1:length(seeds)

    modelPath = "drone_node_gcn_pca_" + string(seeds(m));
    
    for k = 1:length(epsilon)
        fprintf('Verifying model %s with epsilon %.5f\n', modelPath, epsilon(k));
        % Load outputs (must match how you saved them)
        rdata = load("verification_results/mat_files/verified_nodes_" + modelPath + "_eps_" + string(epsilon(k)) + ".mat");

        % Check verification result
        results = cell(size(rdata.outputSets));
        for i = 1:length(rdata.outputSets)
            Y = rdata.outputSets{i};
            lbl = rdata.targets{i};
            results{i} = verifySample(Y, lbl);
        end
        
        % Save results
        parsave(modelPath, epsilon(k), results, rdata.outputSets, rdata.rT, rdata.targets);
        disp("DONE")
    end

end

function results = verifySample(X, target)
    % Generic sample-level verification
    Nsample = size(target,1);
    results = 3 * ones(Nsample,1);
    for i = 1:Nsample
        matIdx = zeros(1,Nsample);
        matIdx(i) = 1;
        if iscell(X)
            X = X{1};
        end
        Y = X.affineMap(matIdx, []);
        Y = Y.toStar; 
        sampleLabel = target(i,:);
        sampleHs = label2Hs(sampleLabel);
        res = verify_specification(Y, sampleHs);
        if res == 2
            res = checkViolated(Y, sampleLabel);
        end
        results(i) = res;
    end
end

function res = checkViolated(Set, label)
    target = getLabelIndex(label);
    [lb, ub] = Set.getRanges;
    maxTarget = ub(target);
    if any(lb > maxTarget)
        res = 0;  % falsified
    else
        res = 2;  % unknown
    end
end

function Hs = label2Hs(label)
    % First determine the actual dimension of your model output
    % For debugging, add this at the beginning of verifySample:
    % fprintf('Star dimension: %d\n', Y.dim);
    
    % Then set outSize to match your actual model output dimension
    outSize = 3; 
    target = getLabelIndex(label);

    % Create verification constraints matching your model's output dimension
    G = -eye(outSize);
    G(:, target) = 1;
    g = zeros(outSize,1);

    Hs = [];
    for i = 1:length(g)
        Hs = [Hs; HalfSpace(G(i,:), g(i))];
    end
end

function index = getLabelIndex(label)
    % If label is already a numeric index
    if isnumeric(label)
        index = label;
    else
        % Your existing switch statement for string/categorical labels
        switch label
            case 'Normal'
                index = 1;
            case 'Low Privilege'
                index = 2;
            case 'Compromised'
                index = 3;
            otherwise
                index = 1; % fallback
        end
    end
end

function parsave(modelPath, epsilon, results, outputSets, rT, targets)
    save("verification_results/mat_files/verified_nodes_" + modelPath + "_eps_" + string(epsilon) + ".mat", ...
         "results", "outputSets", "rT", "targets");
end

