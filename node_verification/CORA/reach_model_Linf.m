function reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest)
    projectRoot = getenv('AV_PROJECT_HOME');
    matDir = fullfile(projectRoot,'node_verification','CORA', ...
                      'verification_results','mat_files');
    if ~exist(matDir,'dir')
        mkdir(matDir);
    end
    % Verification of a Graph Neural Network
    adjacencyDataTest = double(adjacencyDataTest);
    featureDataTest   = double(featureDataTest);

    load("models/"+modelPath+".mat");

    w1 = gather(parameters.mult1.Weights);
    w2 = gather(parameters.mult2.Weights);
    w3 = gather(parameters.mult3.Weights);

    Averify = normalizeAdjacency(adjacencyDataTest);
    XTest = dlarray(featureDataTest);     
    
    N = size(featureDataTest,1);
    % L_inf size
    targets = {};
    outputSets = {};
    rT = {};

    for k = 1:length(epsilon)

        for i = 1:N
            disp("i: ")
            disp(i)
            labelsTest = labelDataTest(i);

            lb = extractdata(XTest);
            ub = extractdata(XTest);
            lb(i, :) = lb(i, :) - epsilon(k);
            ub(i, :) = ub(i, :) + epsilon(k);

            Xverify = ImageStar(lb,ub);
            
            t = tic;

            reachMethod = 'approx-star';
            L = ReluLayer();

            Ybig = computeReachability({w1,w2,w3}, L, reachMethod, Xverify, Averify);
            
            [lb_full, ub_full] = Ybig.getRanges();      
            C = round(numel(lb_full) / N);             
            idx = (i-1)*C + (1:C);                     
            node_lb = lb_full(idx);
            node_ub = ub_full(idx);
            
            % flatten into true vectors
            node_lb = node_lb(:);
            node_ub = node_ub(:);
            
            % build the per‑node star in R^C
            Ynode = Star(node_lb, node_ub);
            
            % store the sliced star (not the big one)
            outputSets{i} = Ynode;   
            targets{i}   = labelsTest;
            rT{i}        = toc(t);
        end

        epsStr = sprintf('%.4f',epsilon(k));
        fname  = fullfile(matDir, ...
                 sprintf("verified_nodes_%s_eps_%s.mat", modelPath, epsStr));
        save(fname, "outputSets","targets","rT","-v7");
        fprintf("SAVED → %s\n", fname);

        disp("SAVED")

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
    newV = Xverify.V; 
    newV = squeeze(Xverify.V); 
    Averify_full = full(Averify);
    newV = tensorprod(Averify_full, newV, 2, 1); 
    w = extractdata(weights{1}); 
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