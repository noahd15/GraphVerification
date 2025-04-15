projectRoot = getenv('AV_PROJECT_HOME');

addpath(genpath(fullfile(projectRoot, '/Verification/GraphNeuralNetworks/nodeClassification/functions/')))
addpath(genpath(fullfile(projectRoot, '/Verification/GraphNeuralNetworks/nodeClassification/models/')))

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
seeds = 1; % models
epsilon = 0.005; % attack

% Verify one model at a time - using regular for loop instead of parfor to avoid file access issues
for k = 1:length(seeds)
    % Construct the model path
    modelPath = "node_gcn_" + string(seeds(k));

    fprintf('Verifying model %s with epsilon %.4f\n', modelPath, epsilon);

    % Verify the model
    try
        reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest);
        fprintf('Successfully verified model %s\n', modelPath);
    catch ME
        fprintf('Error verifying model %s: %s\n', modelPath, ME.message);
    end
end

