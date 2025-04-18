projectRoot = getenv('AV_PROJECT_HOME');

addpath(genpath(fullfile(projectRoot, '/nodeVerification/functions/')));
addpath(genpath(fullfile(projectRoot, '/nodeVerification/models/')));
% addpath(genpath('/users/noahdahle/nnv'))
addpath(genpath(fullfile(projectRoot, '/nodeVerification/nnv/')));

dataFile = fullfile(projectRoot, 'data', 'cora_node.mat');
data = load(dataFile);

% Match label processing to batched_CORA_Modified.m
adjacencyData = data.edge_indices(:,:,1);           % [2708 x 2708]
featureData   = data.features(:,:,1);               % [2708 x featureDim]
labelData     = double(data.labels(:)) + 1;         % [2708 x 1], labels 1–7
numNodes      = size(featureData, 1);

% Recreate train/val/test splits as in batched_CORA_Modified.m
rng(2024);
indices = randperm(numNodes);
nTrain = round(0.8 * numNodes);
nVal   = round(0.1 * numNodes);
idxTrain      = indices(1:nTrain);
idxValidation = indices(nTrain+1 : nTrain+nVal);
idxTest       = indices(nTrain+nVal+1 : end);

% Select test data as rows
adjacencyDataTest = adjacencyData(idxTest, idxTest);      % [numTest x numTest]
featureDataTest   = featureData(idxTest, :);              % [numTest x featureDim]
labelDataTest     = labelData(idxTest);                   % [numTest x 1]

fprintf('Number of test samples: %d\n', length(idxTest));
fprintf('Feature dimension: %d\n', size(featureDataTest, 2));

%% Verify models

seeds = [1]; % models
epsilon = [0.005]; % attack

for k = 1:length(seeds)
    modelPath = "node_gcn_" + string(seeds(k));
    fprintf('Verifying model %s with epsilon %.4f\n', modelPath, epsilon);
    reach_model_CORA(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest);
end

