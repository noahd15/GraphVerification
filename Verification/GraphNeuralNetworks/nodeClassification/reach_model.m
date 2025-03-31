% Save model parameters in a format compatible with reach_model_Linf.m
parameters = struct();
parameters.mult1 = struct('Weights', W1);
parameters.mult2 = struct('Weights', W2);
parameters.mult3 = struct('Weights', W3);

% Save to models directory
if ~exist('models', 'dir')
    mkdir('models');
end
save("./models/node_gnn_model.mat", "parameters", "muX", "sigsqX");

% Set parameters for verification
modelPath = "node_gnn_model.mat";
epsilon = [0.01, 0.05, 0.1]; % Different perturbation bounds to test

% Load test data to verify
load('data/dataset_matlab_node.mat');

% Extract a subset of test samples for verification
num_samples = 5; % Adjust as needed
test_indices = randperm(length(features), num_samples);

test_adjacency = edge_indices(test_indices);
test_features = features(test_indices);
test_labels = labels(test_indices);

% Convert to 3D arrays as expected by reach_model_Linf
adjacencyDataTest = zeros(18, 18, num_samples);
featureDataTest = zeros(18, size(features{1}, 2), num_samples);
labelDataTest = zeros(num_samples, 1);

for i = 1:num_samples
    A = test_adjacency{i};
    X = test_features{i};
    y = test_labels{i};
    
    % Ensure consistent dimensions (pad if needed)
    if size(A, 1) > 18
        A = A(1:18, 1:18);
    elseif size(A, 1) < 18
        A_new = zeros(18, 18);
        A_new(1:size(A,1), 1:size(A,2)) = A;
        A = A_new;
    end
    
    if size(X, 1) > 18
        X = X(1:18, :);
    elseif size(X, 1) < 18
        X_new = zeros(18, size(X, 2));
        X_new(1:size(X,1), :) = X;
        X = X_new;
    end
    
    % Store in the format expected by reach_model_Linf
    adjacencyDataTest(:,:,i) = A;
    featureDataTest(:,:,i) = X;
    labelDataTest(i) = y(1); % Use first node's label for graph-level label
end

% Run reachability analysis
reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest);

% After verification, analyze results
disp('Verification complete. Results saved in the results folder.');

function reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest)
    % Verification of a Graph Neural Network
    %% Load parameters of gcn
    load("./models/"+modelPath);
    
    w1 = gather(parameters.mult1.Weights);
    w2 = gather(parameters.mult2.Weights);
    w3 = gather(parameters.mult3.Weights);
    
    %% Start for loop for verification here, preprocess one graph at a time
    
    N = size(adjacencyDataTest, 3);
    
    % Store results
    targets = {};
    outputSets = {};
    rT = {};
    
    for k = 1:length(epsilon)
    
        for i = 1:N
            % Get graph data
            [ATest, XTest, labelsTest] = preprocessData(adjacencyDataTest(:,:,i), featureDataTest(:,:,i), labelDataTest(i,:));
            
            % normalize data
            XTest = (XTest - muX)./sqrt(sigsqX);
            XTest = dlarray(XTest);
                    
            % adjacency matrix represent connections, so keep it as is
            Averify = normalizeAdjacency(ATest);
            
            % Get input set: input values for each node is X
            lb = extractdata(XTest-epsilon(k));
            ub = extractdata(XTest+epsilon(k));
            Xverify = ImageStar(lb,ub);
            
            % Compute reachability
            t = tic;
            
            reachMethod = 'approx-star';
            L = ReluLayer(); % Create relu layer;
            
            Y = computeReachability({w1,w2,w3}, L, reachMethod, Xverify, Averify);

            % store results
            outputSets{i} = Y;
            targets{i} = labelsTest;
            rT{i} = toc(t);
        
        end
        
        % Save verification results
        save("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat", "outputSets", "targets", "rT");
    
    end
end

%% Helper functions

function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData)
    % Extract the non-zero part of the adjacency matrix and features
    numNodes = find(any(adjacencyData), 1, "last");
    if isempty(numNodes)
        numNodes = size(featureData, 1);
    end
    
    % Extract unpadded adjacency and features
    adjacency = sparse(adjacencyData(1:numNodes, 1:numNodes));
    features = featureData(1:numNodes, :);
    
    % Extract labels
    labels = labelData(1:numNodes);
    
    % Convert labels to categorical if needed
    if ~iscategorical(labels) && ~isempty(labels)
        uniqueLabels = unique(labels(labels > 0));
        labels = categorical(labels, uniqueLabels);
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
    % weights = weights of GNN ({w1, w2, w3}
    % L = Layer type (ReLU)
    % reachMethod = reachability method for all layers('approx-star is default)
    % input = pertubed input features (ImageStar)
    % adjMat = adjacency matric of corresonding input features
    % Y = computed output of GNN (ImageStar)

    Xverify = input;
    Averify = adjMat;
    n = size(adjMat,1);
    
    %%%%%%%%  LAYER 1  %%%%%%%%
    
    % Graph convolution + linear transformation
    newV = Xverify.V;
    newV = reshape(newV, [n n+1]);
    newV = Averify * newV;
    newV = tensorprod(newV, extractdata(weights{1}));
    newV = permute(newV, [1 4 3 2]);
    X1 = ImageStar(newV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    
    % ReLU activation
    X1_act = L.reach(X1, reachMethod);
    
    %%%%%%%%  LAYER 2  %%%%%%%%
    
    % Graph convolution + linear transformation
    newV = X1_act.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{2}),2,1);
    newV = permute(newV, [1 4 2 3]);
    X2 = ImageStar(newV, X1_act.C, X1_act.d, X1_act.pred_lb, X1_act.pred_ub);
    
    % ReLU activation
    X2_act = L.reach(X2, reachMethod);
    
    %%%%%%%%  LAYER 3  %%%%%%%%
    
    % Graph convolution + linear transformation (final layer)
    newV = X2_act.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{3}), 2, 1);
    newV = permute(newV, [1 4 2 3]);
    Y = ImageStar(newV, X2_act.C, X2_act.d, X2_act.pred_lb, X2_act.pred_ub);
end


