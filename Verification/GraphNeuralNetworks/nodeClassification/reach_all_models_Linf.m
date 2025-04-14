projectRoot = getenv('AV_PROJECT_HOME');

addpath(genpath(fullfile(projectRoot, '/Verification/GraphNeuralNetworks/nodeClassification/functions/')))
addpath(genpath(fullfile(projectRoot, '/Verification/GraphNeuralNetworks/nodeClassification/models/')))

% Load data
dataFile = fullfile(projectRoot, 'data', 'node.mat');
data = load(dataFile);rng(0); 

% Convert edge indices to adjacency matrices
adjacencyData = edges2Adjacency(data);
featureData = data.features;
labelData = double(permute(data.labels, [2 1]));

% Partition data
numObservations = size(adjacencyData, 3);
[idxTrain, idxVal, idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

adjacencyDataTest = adjacencyData(:,:,idxTest);
featureDataTest = featureData(:,:,idxTest);
atomDataTest = labelData(idxTest,:);

%% Verify models

% Study Variables
% seeds = [0,1,2,3,4]; % models
seeds = [1]; % models
epsilon = [0.005]% , 0.01, 0.02, 0.05]; % attack

% Verify one model at a time
parfor k = 1:length(seeds)
    % Construct the model path
    modelPath = "node_gcn_" + string(seeds(k));

    % Verify the model
    reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest);
end

