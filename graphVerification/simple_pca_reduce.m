% This script performs PCA on the node features to reduce dimensionality.
% It creates a new feature matrix with reduced dimensions that can be used with the batched training script.

projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME environment variable is not set. Please set it to your project root directory.');
end

addpath(genpath(fullfile(projectRoot, '/nodeVerification/important/functions/')));
addpath(genpath(fullfile(projectRoot, '/nodeVerification/important/models/')));

%% Load original data
dataFile = fullfile(projectRoot, 'data', 'node.mat');
data = load(dataFile);

disp(data);


featureData = data.features;          % [numNodes x featureDim x numGraphs]
% disp(featureData);
labelData = double(permute(data.labels, [2 1]));
if size(labelData, 1) < size(labelData, 2)
    labelData = labelData';
end
labelData = labelData + 1;  % Adjust labels to start from 1 instead of 0

num_classes = max(labelData);


adjacencyData = edges2Adjacency(data);
numObservations = size(adjacencyData, num_classes);

%% Split data into train/validation/test sets
rng(2024);  % For reproducibility


[idxTrain, idxValidation, idxTest] = trainingPartitions(numObservations, [0.8 0.1 0.1]);
%% --- Perform PCA on the Training Data ---
% Get dimensions
[numNodes, numFeatures, numGraphs] = size(featureData);
desired_dim = 16;  % Target dimensionality after PCA

% Create a new feature array with reduced dimensions
featureData_reduced = zeros(numNodes, desired_dim, numGraphs);

% Collect all non-zero node features for PCA training
allFeatures = [];
for i = 1:length(idxTrain)
    graphIdx = idxTrain(i);
    % Get features for this graph
    graphFeatures = featureData(:, :, graphIdx);
    % Only include non-zero rows (actual nodes)
    nonZeroRows = any(graphFeatures, 2);
    allFeatures = [allFeatures; graphFeatures(nonZeroRows, :)];
end

% Perform PCA on the collected features
fprintf('Performing PCA to reduce features from %d to %d dimensions...\n', numFeatures, desired_dim);
[coeff, ~, ~, ~, explained, mu] = pca(allFeatures);

% Keep only the first desired_dim principal components

coeff_reduced = coeff(:, 1:desired_dim);

% Print variance explained
varExplained = sum(explained(1:desired_dim));
fprintf('Variance explained by %d components: %.2f%%\n', desired_dim, varExplained);

% Apply PCA transformation to all graphs
for i = 1:numGraphs
    % Get features for this graph
    graphFeatures = featureData(:, :, i);
    
    % Find non-zero rows (actual nodes)
    nonZeroRows = any(graphFeatures, 2);
    numActualNodes = sum(nonZeroRows);
    
    if numActualNodes > 0
        % Apply PCA transformation only to non-zero rows
        nodeFeatures = graphFeatures(nonZeroRows, :);
        reducedFeatures = (nodeFeatures - mu) * coeff_reduced;
        
        % Store the reduced features back in the 3D array
        featureData_reduced(nonZeroRows, :, i) = reducedFeatures;
    end
end

%% Save the Reduced Dataset
% Save in a format compatible with the batched training script
save(fullfile(projectRoot, 'data', 'reducedDatasetNode.mat'), 'coeff_reduced', 'mu', ...
    'featureData_reduced', 'adjacencyData', 'labelData', ...
    'idxTrain', 'idxValidation', 'idxTest', '-v7.3');

fprintf('Reduced dataset saved to: %s\n', fullfile(projectRoot, 'data', 'reducedDatasetNode.mat'));
