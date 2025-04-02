%% Verification of all GNN for node classification with Linf perturbation

% verify multiple models trained on different random seeds
% Same data for all
% 4 different size of Linf attacks

addpath(genpath('C:/Users/Noah/OneDrive - Vanderbilt/Spring 2025/CS 6315/Project/AV_Project/Verification/GraphNeuralNetworks/nodeClassification/functions/'));
addpath(genpath("C:/Users/Noah/OneDrive - Vanderbilt/Spring 2025/CS 6315/Project/AV_Project/Verification/GraphNeuralNetworks/nodeClassification/models/"));
addpath(genpath('C:/Users/Noah/nnv'))
%% Load data
dataset = load('../../../Data/node.mat');
rng(0); % ensure we can reproduce (data partition)

% Convert edge indices to adjacency matrices
adjacency_matrices = edges2Adjacency(dataset);

% Partition data
numObservations = length(dataset.edge_indices);
[idxTrain, idxVal, idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

% Get data from test partition
% Convert cell array to 3D matrix for the test set
test_adjacency_cells = adjacency_matrices(idxTest);
test_features = dataset.features(idxTest);
test_labels = dataset.labels(idxTest);

% Create 3D matrix from adjacency matrices
num_test = length(idxTest);
max_nodes = 0;
max_features = 0;

% Find maximum dimensions for node count and feature size
for i = 1:num_test
    max_nodes = max(max_nodes, size(test_adjacency_cells{i}, 1));
    if ~isempty(test_features{i})
        feature_size = size(test_features{i}, 2);
        max_features = max(max_features, feature_size);
    end
end

% Create standardized 3D matrices
adjacencyDataTest = zeros(max_nodes, max_nodes, num_test);
featureDataTest = zeros(max_nodes, max_features, num_test);
labelDataTest = zeros(num_test, max_nodes);

% Fill the matrices with data, padding as needed
for i = 1:num_test
    % Adjacency matrix
    adj_size = size(test_adjacency_cells{i}, 1);
    adjacencyDataTest(1:adj_size, 1:adj_size, i) = full(test_adjacency_cells{i});
    
    % Features
    if ~isempty(test_features{i})
        feat = test_features{i};
        feat_rows = size(feat, 1);
        feat_cols = size(feat, 2);
        featureDataTest(1:feat_rows, 1:feat_cols, i) = feat;
    end
    
    % Labels
    if ~isempty(test_labels{i})
        label = test_labels{i};
        if numel(label) <= max_nodes
            labelDataTest(i, 1:numel(label)) = label(:)';
        else
            labelDataTest(i, 1:max_nodes) = label(1:max_nodes)';
        end
    end
end

%% Verify models

% Study Variables
% seeds = [0,1,2,3,4]; % models
seeds = [5,6,7,8,9]; % models
epsilon = [0.005, 0.01, 0.02, 0.05]; % attack

% Verify one model at a time
% parfor k=1:length(seeds)
    % get model
    % modelPath = "gcn_"+string(seeds(k));
modelPath = "trained_node_model";

% Verify model
reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest);

% end

