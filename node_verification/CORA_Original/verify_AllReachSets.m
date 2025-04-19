%% Verify the robustness reach sets of all models

epsilon = [0.1] %[0.005; 0.01; 0.02; 0.05];
% seeds = [0,1,2,3,4];
seeds = [1] %[5,6,7,8,9]; % models

for m=1:length(seeds)
    % get model
    modelPath = sprintf("cora_node_gcn_%d", seeds(m));
    
    for k = 1:length(epsilon)
    
        % Load data one at a time
        fprintf("Worker %d: Loading data for model %d\n", m, seeds(m));

        rdata = load("verification_results/mat_files/verified_nodes_"+modelPath+"_eps_"+string(epsilon(k))+".mat");
        % Check for robustness value (one molecule, 1 atom at a time)
        results = {};
        for i = 1:length(rdata.outputSets)
            Y = rdata.outputSets{i};   
            label = rdata.targets{i};  
            results{i} = verifyAtom(Y, label);
        end

    
        % Save results
        parsave(modelPath, epsilon(k), results, rdata.outputSets, rdata.rT, rdata.targets);
        % save("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat", "results", "outputSets", "rT", "targets");
        
    end

end

function result = verifyAtom(X, target)
    % X is a 7D Star set (output scores for one node)
    atomHs = label2Hs(target);

    res = verify_specification(X, atomHs);

    if res == 2
        res = checkViolated(X, target);
    end

    result = res;
end

function res = checkViolated(Set, label)
    res = 5; % assume unknown (property is not unsat, try to sat)
    target = label;
    % Get bounds for every index
    [lb,ub] = Set.getRanges;
    maxTarget = ub(target);
    % max value of the target index smaller than any other lower bound?
    if any(lb > maxTarget)
        res = 0; % falsified
    end
end

function Hs = label2Hs(label)
    % Convert output target to halfspace for verification
    % @Hs: unsafe/not robust region defined as a HalfSpace

    outSize = 7; % num of classes
    % classes = ["H";"C";"N";"O";"S"];
    target = label;

    % Define HalfSpace Matrix and vector
    G = ones(outSize,1);
    G = diag(G);
    G(target, :) = [];
    G = -G;
    G(:, target) = 1;

    g = zeros(size(G,1),1);

    % Create HalfSapce to define robustness specification
    Hs = [];
    for i=1:length(g)
        Hs = [Hs; HalfSpace(G(i,:), g(i))];
    end

end

function parsave(modelPath, epsilon, results, outputSets, rT, targets)
    save("verification_results/mat_files/verified_nodes_"+modelPath+"_eps_"+string(epsilon)+".mat", "results", "outputSets", "rT", "targets");
end
