function reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest, num_features)
    % Setup
    adjacencyDataTest = reshape(adjacencyDataTest, [size(adjacencyDataTest, 1), size(adjacencyDataTest, 2), 1]);
    featureDataTest   = reshape(featureDataTest, [size(featureDataTest, 1), size(featureDataTest, 2), 1]);
    adjacencyDataTest = double(adjacencyDataTest);
    featureDataTest   = double(featureDataTest);

    load("models/"+modelPath+".mat");

    w1 = gather(parameters.mult1.Weights);
    w2 = gather(parameters.mult2.Weights);
    w3 = gather(parameters.mult3.Weights);

    N = size(featureDataTest, 3);
    reachMethod = 'approx-star';
    L = ReluLayer();

    for k = 1:length(epsilon)
        results    = {};
        outputSets = {};
        targets    = {};
        rT         = {};

        for i = 1:N
            [ATest, XTest, labelsTest] = preprocessData(adjacencyDataTest(:,:,i), featureDataTest(:,:,i), labelDataTest(i,:));
            Averify = normalizeAdjacency(ATest);
            XTest = dlarray(XTest);

            % Construct input set
            lb = extractdata(XTest - epsilon(k));
            ub = extractdata(XTest + epsilon(k));
            Xverify = ImageStar(lb, ub);

            % Reachability
            t = tic;
            Y_all = computeReachability({w1, w2, w3}, L, reachMethod, Xverify, Averify);
            elapsedTime = toc(t);

            numNodes = size(Y_all.V, 1);
            for j = 1:numNodes
                matIdx = zeros(1, numNodes); matIdx(j) = 1;
                Y_j = Y_all.affineMap(matIdx, []);

                outputSets{(i-1)*numNodes + j} = Y_j;
                targets{(i-1)*numNodes + j}    = labelDataTest(j);
                rT{(i-1)*numNodes + j}         = elapsedTime;

                % Verify immediately
                results{(i-1)*numNodes + j} = verifyAtom(Y_j, labelDataTest(j));
            end
        end

        % Save everything at once (smaller since verification done)
        save("verification_results/mat_files/verified_nodes_"+modelPath+"_eps_"+string(epsilon(k))+ "_" + string(num_features) + ".mat", ...
    "results", "rT", "targets", "-v7.3");
        disp("DONE: "+modelPath+", eps="+epsilon(k));
    end
end


function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData)
    [adjacency, features] = preprocessPredictors(adjacencyData, featureData);
    labels = labelData(:);
end

function [adjacency, features] = preprocessPredictors(adjacencyData, featureData)
    adjacency = sparse([]);
    features = [];

    for i = 1:size(adjacencyData, 3)
        numNodes = find(any(adjacencyData(:,:,i)), 1, "last");
        if isempty(numNodes) || numNodes==0
            continue
        end

        A = adjacencyData(1:numNodes, 1:numNodes, i);
        X = featureData(1:numNodes, :, i);

        adjacency = blkdiag(adjacency, A);

        % Concatenate feature rows
        features = [features; X];
    end
end

function ANorm = normalizeAdjacency(A)

    % Add self connections to adjacency matrix.
    A = A + speye(size(A));

    % Compute inverse square root of degree.
    degree = sum(A, 2);
    degreeInvSqrt = sparse(sqrt(1./degree));

    % Normalize adjacency matrix.
    ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);

end

function Y = computeReachability(weights, L, reachMethod, input, adjMat)
    Xverify = input;
    Averify = adjMat; 
    n = size(adjMat,1); 

    %%%%%%%%  LAYER 1  %%%%%%%%
    newV = Xverify.V; %272x16x1x4353
    newV = squeeze(Xverify.V); %272x16x4353  
    Averify_full = full(Averify);
    newV = tensorprod(Averify_full, newV, 2, 1); %272x16x4353 
    w = extractdata(weights{1}); %32x64 
    newV = tensorprod(newV, extractdata(weights{1}), 2, 1);
    newV = reshape(newV, [size(newV,1), size(newV,2), 1, size(newV,3)]); 
    newV = permute(newV, [1 4 3 2]); 
    
    X2 = ImageStar(newV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub); 
    % part 2 %
    X2b = L.reach(X2, reachMethod);
    repV = repmat(Xverify.V,[1,2,1,1]); 
    Xrep = ImageStar(repV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    X2b_ = X2b.MinkowskiSum(Xrep);

    %%%%%%%%  LAYER 2  %%%%%%%%

    % part 1
    newV = X2b_.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{2}),2,1);
    newV = permute(newV, [1 4 2 3]);
    X3 = ImageStar(newV, X2b_.C, X2b_.d, X2b_.pred_lb, X2b_.pred_ub);
    % part 2
    X3b = L.reach(X3, reachMethod);
    X3b_ = X3b.MinkowskiSum(X2b_);

    %%%%%%%%  LAYER 3  %%%%%%%%

    newV = X3b_.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{3}), 2, 1);
    newV = permute(newV, [1 4 2 3]);
    Y = ImageStar(newV, X3b_.C, X3b_.d, X3b_.pred_lb, X3b_.pred_ub);
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
