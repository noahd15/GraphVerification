projectRoot = getenv('AV_PROJECT_HOME');

addpath(genpath(fullfile(projectRoot, '/nodeVerification/functions/')));
addpath(genpath(fullfile(projectRoot, '/nodeVerification/models/')));

% Load Cora dataset
dataFile = fullfile(projectRoot, 'data', 'cora_node.mat');
data = load(dataFile);

% Full Cora graph
A_full = data.edge_indices(:,:,1);    % 2708×2708 sparse (possibly single)
X_full = data.features(:,:,1);        % 2708×featureDim
y_full = double(data.labels(:)) + 1;   % 2708×1, labels 1–7
[numNodes, featureDim] = size(X_full);
rng(2024);

% Train/Val/Test node splits
indices = randperm(numNodes);
nTrain = round(0.8 * numNodes);
nVal = round(0.1 * numNodes);
idxTrain = indices(1:nTrain);
idxValidation = indices(nTrain+1 : nTrain+nVal);
idxTest = indices(nTrain+nVal+1 : end);

% Prepare test data for verification
adjacencyDataTest = A_full(idxTest, idxTest);
featureDataTest = X_full(idxTest, :);
labelDataTest = y_full(idxTest);

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

    reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest);
end

