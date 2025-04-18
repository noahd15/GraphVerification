function reach_model_Linf_CORA(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest)
    % Verification of a Graph Neural Network

    load("models/"+modelPath+".mat");

    w1 = gather(parameters.mult1.Weights);
    w2 = gather(parameters.mult2.Weights);
    w3 = gather(parameters.mult3.Weights);
    % fc_w = gather(parameters.fc.Weights);
    % fc_b = gather(parameters.fc.Bias);

    N = size(featureDataTest, 3);
    % L_inf size
    % epsilon = [0.005; 0.01; 0.02; 0.05];
    targets = {};
    outputSets = {};
    rT = {};

    for k = 1:length(epsilon)

        load("models/"+modelPath+".mat");

        % Build test graph once
        [ATest, XTest, labelsTest] = preprocessData( ...
            adjacencyDataTest, ...    % N_test×N_test
            featureDataTest,   ...    % N_test×featureDim
            labelDataTest      ...    % N_test×1
        );

        % Prepare set‑based input
        XTest   = dlarray(XTest);
        Averify = normalizeAdjacency(ATest);

        lb = extractdata(XTes t - epsilon(k));
        ub = extractdata(XTest + epsilon(k));
        Xverify = ImageStar(lb, ub);

        t = tic;

        reachMethod = 'approx-star';
        L = ReluLayer();

        Y = computeReachability({w1,w2,w3}, L, reachMethod, Xverify, Averify);

        % store results
        outputSets{i} = Y;
        targets{i} = labelsTest;
        rT{i} = toc(t);

        if ~exist('results', 'dir')
            mkdir('results');
        end
        
        save("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat", "outputSets", "targets", "rT", '-v7.3');
        disp("SAVED")

    end
end

function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData)
    % Handle the single Cora graph case (2‑D adjacency & features)
    %   adjacencyData: N × N
    %   featureData  : N × F
    %   labelData    : N × 1
    %
    % Returns:
    %   adjacency : sparse double N×N
    %   features  : N×F
    %   labels    : N×1

    % Convert to sparse double
    adjacency = sparse(double(adjacencyData));

    % Pass features through untouched
    features = featureData;

    % Make sure labels is a column
    labels = labelData(:);
end


function [adjacency, features] = preprocessPredictors(adjacencyData, featureData)
    % Start with an empty sparse‐double matrix
    adjacency = sparse([], [], [], 0, 0, 0);
    features  = [];

    for i = 1:size(adjacencyData, 3)
        % Find how many nodes are actually present in this slice
        numNodes = find(any(adjacencyData(:,:,i)), 1, "last");
        if isempty(numNodes) || numNodes==0
            continue
        end

        % Extract the single‐precision sparse and convert to sparse double
        A_single = adjacencyData(1:numNodes, 1:numNodes, i);
        A = sparse(double(A_single));  

        % Grab the corresponding feature rows
        X = featureData(1:numNodes, :, i);

        % Build the big block‑diagonal adjacency
        adjacency = blkdiag(adjacency, A);

        % Stack the features
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
    Averify = adjMat; %18 x 18
    n = size(adjMat,1); %18

    %%%%%%%%  LAYER 1  %%%%%%%%

    % part 1
    newV = Xverify.V; %18 x 16 x 1 x 289
    newV = squeeze(Xverify.V); % 18 x 16 x 289
    Averify_full = full(Averify);
    newV = tensorprod(Averify_full, newV, 2, 1); % 18 x 16 x 289
    w = extractdata(weights{1}); % 16x32
    newV = tensorprod(newV, extractdata(weights{1}), 2, 1); %18 x 289 x 32
    newV = reshape(newV, [size(newV,1), size(newV,2), 1, size(newV,3)]); % 18 x 289 x 1 x 32
    newV = permute(newV, [1 4 3 2]); % 18 x 32 x 1 x 289
    X2 = ImageStar(newV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub); % 18 x 32 x 1 x 289
    % part 2 %
    X2b = L.reach(X2, reachMethod); % 18 x 32 x 1 x 289
    repV = repmat(Xverify.V,[1,2,1,1]); %18 x 32 x 1 x 289
    Xrep = ImageStar(repV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    % X2b_ = X2b.MinkowskiSum(Xrep);
    % size(X2b_.V)

    %%%%%%%%  LAYER 2  %%%%%%%%

    % part 1
    newV = X2b.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{2}),2,1);
    newV = permute(newV, [1 4 2 3]);
    X3 = ImageStar(newV, X2b.C, X2b.d, X2b.pred_lb, X2b.pred_ub);
    % part 2
    X3b = L.reach(X3, reachMethod);
    % X3b_ = X3b.MinkowskiSum(X2b_);

    %%%%%%%%  LAYER 3  %%%%%%%%

    newV = X3b.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{3}), 2, 1);
    newV = permute(newV, [1 4 2 3]);
    Y = ImageStar(newV, X3b.C, X3b.d, X3b.pred_lb, X3b.pred_ub);

     %%%%%%%%  LAYER 4  %%%%%%%%
    % numNodes = size(Y_conv.V,1);
    % Y_nodes = cell(numNodes,1);

    % for nodeIdx = 1:numNodes
    %     Y_node = Y_conv.slice(1, nodeIdx);  
    %     Y_final_node = Y_node.affineMap(w1, w2);  
    %     Y_nodes{nodeIdx} = Y_final_node;
    % end
    % Y = Y_nodes;


end