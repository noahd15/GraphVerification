function reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest)
    % Verification of a Graph Neural Network


    adjacencyDataTest = reshape(adjacencyDataTest, [size(adjacencyDataTest, 1), size(adjacencyDataTest, 2), 1]);
    featureDataTest = reshape(featureDataTest, [size(featureDataTest, 1), size(featureDataTest, 2), 1]);

    fprintf('adjacencyDataTest size: %d x %d x %d\n', size(adjacencyDataTest, 1), size(adjacencyDataTest, 2), size(adjacencyDataTest, 3));
    fprintf('featureDataTest size: %d x %d x %d\n', size(featureDataTest, 1), size(featureDataTest, 2), size(featureDataTest, 3));


    load("models/"+modelPath+".mat");

    w1 = gather(parameters.mult1.Weights);
    w2 = gather(parameters.mult2.Weights);
    w3 = gather(parameters.mult3.Weights);

    N = size(featureDataTest, 3);
    % L_inf size
    % epsilon = [0.005; 0.01; 0.02; 0.05];
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
            % fprintf('Xverify size: %d x %d x %d x %d\n', size(Xverify.V,1), size(Xverify.V,2), size(Xverify.V,3), size(Xverify.V,4));
            x = Xverify.V;
            whos x;

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
        
        save("verification_results/mat_files/verified_nodes_"+modelPath+"_eps_"+string(epsilon(k))+".mat", "outputSets", "targets", "rT", '-v7.3');
        disp("SAVED")

    end
end

function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData)
    [adjacency, features] = preprocessPredictors(adjacencyData, featureData);
    labels = [];
    for i = 1:size(adjacencyData, 3)
        numNodes = find(any(adjacencyData(:,:,i)), 1, "last");
        if isempty(numNodes)
            numNodes = 0;
        end

        % Fix: Use only the first numNodes elements from the i-th row
        T = labelData(i, 1:min(numNodes, size(labelData,2)));
        labels = [labels; T(:)];
    end
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

    % adjacency = reshape(adjacency, [size(adjacency, 1), size(adjacency, 2), 1]);
    % features = reshape(features, [size(features, 1), size(features, 2), 1]);

    % fprintf('Feature dimensions: %d x %d x %d\n', size(features, 1), size(features, 2), size(features, 3));
    % fprintf('Adjacency dimensions: %d x %d x %d\n', size(adjacency, 1), size(adjacency, 2), size(adjacency, 3));
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
    % fprintf('newV size: %d x %d x %d\n', size(newV,1), size(newV,2), size(newV,3));
    newV = reshape(newV, [size(newV,1), size(newV,2), 1, size(newV,3)]); % 18 x 289 x 1 x 32
    newV = permute(newV, [1 4 3 2]); % 18 x 32 x 1 x 289
    X2 = ImageStar(newV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub); % 18 x 32 x 1 x 289

    % part 2 %
    X2b = L.reach(X2, reachMethod); % 18 x 32 x 1 x 289
    repV = repmat(Xverify.V,[1,2,1,1]); %18 x 32 x 1 x 289
    Xrep = ImageStar(repV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    % fprintf('Xrep size: %d x %d x %d x %d\n', size(Xrep.V,1), size(Xrep.V,2), size(Xrep.V,3), size(Xrep.V,4));
    X2b_ = X2b.MinkowskiSum(Xrep);
    % size(X2b_.V)

    %%%%%%%%  LAYER 2  %%%%%%%%

    % part 1
    newV = X2b_.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{2}),2,1);
    newV = permute(newV, [1 4 2 3]);
    X3 = ImageStar(newV, X2b.C, X2b.d, X2b.pred_lb, X2b.pred_ub);
    % part 2
    X3b = L.reach(X3, reachMethod);
    X3b_ = X3b.MinkowskiSum(X2b_);

    %%%%%%%%  LAYER 3  %%%%%%%%

    newV = X3b_.V;
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