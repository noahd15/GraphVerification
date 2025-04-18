projectRoot = getenv('AV_PROJECT_HOME');

addpath(genpath(fullfile(projectRoot, '/nodeVerification/functions/')));
addpath(genpath(fullfile(projectRoot, '/nodeVerification/models/')));

% 2) Load Cora
dataFile  = fullfile(projectRoot, 'data', 'cora_node.mat');
data      = load(dataFile);
A_full    = data.edge_indices(:,:,1);      % 2708×2708
X_full    = data.features(:,:,1);          % 2708×featureDim
y_full    = double(data.labels(:)) + 1;    % 2708×1
[numNodes, featureDim] = size(X_full);

% 3) Train/Val/Test split (same as before)
rng(2024);
indices      = randperm(numNodes);
nTrain       = round(0.8 * numNodes);
nVal         = round(0.1 * numNodes);
idxTest      = indices(nTrain+nVal+1 : end);

%% Verify models

% Study Variables
% seeds = [0,1,2,3,4]; % models
seeds = [1]; % models
epsilon = [0.005]; % attack

% Verify one model at a time - using regular for loop instead of parfor to avoid file access issues
for k = 1:length(seeds)
    % Construct the model path
    modelPath = "cora_node_gcn_" + string(seeds(k));

    fprintf('Verifying model %s with epsilon %.4f\n', modelPath, epsilon);

    reach_model_Linf_CORA(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest);
end

