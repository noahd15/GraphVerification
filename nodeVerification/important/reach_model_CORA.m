function reach_model_CORA(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest)
    % Verification of a Graph Neural Network

    load("models/"+modelPath+".mat");

    w1 = gather(parameters.mult1.Weights);
    w2 = gather(parameters.mult2.Weights);
    w3 = gather(parameters.mult3.Weights);

    N = size(featureDataTest, 3);

    targets = {};
    outputSets = {};
    rT = {};

    for k = 1:length(epsilon)

        for i = 1:N

            [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest(:,:,i),featureDataTest(:,:,i),labelDataTest(i,:));

            XTest = dlarray(XTest);
            Averify = normalizeAdjacency(ATest);

            lb = extractdata(XTest-epsilon(k));
            ub = extractdata(XTest+epsilon(k));

            Xverify = ImageStar(lb,ub);

            t = tic;

            reachMethod = 'approx-star';
            L = ReluLayer();

            Y = computeReachability({w1,w2,w3}, L, reachMethod, Xverify, Averify);

            % store results
            outputSets{i} = Y;
            targets{i} = labelsTest;
            rT{i} = toc(t);
        end

        if ~exist('results', 'dir')
            mkdir('results');
        end
        
        save("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat", "outputSets", "targets", "rT", '-v7.3');
        disp("SAVED")

    end
end

function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData)
    [adjacency, features] = preprocessPredictors(adjacencyData, featureData);
    % For node classification, just return the label vector
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

        if mod(i, 500) == 0
            fprintf('Processing graph %d\n', i);
        end
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

    % part 1
    newV = Xverify.V;
    newV = squeeze(Xverify.V);
    Averify_full = full(Averify);
    newV = tensorprod(Averify_full, newV, 2, 1);
    w = extractdata(weights{1});
    fprintf('weights{1} size: %s\nnewV size: %s\n', mat2str(size(w)), mat2str(size(newV)));


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

function sym = labelSymbol(labelNumbers)
    sym = strings(size(labelNumbers));
    for k = 1:numel(labelNumbers)
        switch labelNumbers(k)
            case 1
                sym(k) = "Not Compromised";
            case 2
                sym(k) = "Compromised";
            case 3
                sym(k) = "Highly Compromised";
            otherwise
                error("Invalid label number: %g. Supported labels are 0,1,2,3.", labelNumbers(k));
        end
    end
end