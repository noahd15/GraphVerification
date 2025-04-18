projectRoot = getenv('AV_PROJECT_HOME');

addpath(genpath(fullfile(projectRoot, '/node_verification/functions/')));
addpath(genpath(fullfile(projectRoot, '/node_verification/models/')));
addpath(genpath('/Users/Noah/nnv'));

dataFile = fullfile(projectRoot, 'data', 'reducedDatasetNode.mat');
data = load(dataFile);
rng(0);

adjacencyData = data.adjacencyData;
featureData = data.featureData_reduced;
labelData = data.labelData;

idxTest = data.idxTest;

adjacencyDataTest = adjacencyData(:,:,idxTest);
featureDataTest = featureData(:,:,idxTest);
labelDataTest = labelData(idxTest,:);

fprintf('Number of test samples: %d\n', length(idxTest));
fprintf('Feature dimension: %d\n', size(featureDataTest, 2));

%% Verify models

% Study Variables
% seeds = [0,1,2,3,4]; % models
seeds = [1]; % models
epsilon = [0.005]; % attack

% Verify one model at a time - using regular for loop instead of parfor to avoid file access issues
for k = 1:length(seeds)
    % Construct the model path
    modelPath = "drone_node_gcn_pca_" + string(seeds(k));

    fprintf('Verifying model %s with epsilon %.4f\n', modelPath, epsilon);

    reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest);
end

