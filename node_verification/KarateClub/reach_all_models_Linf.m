    projectRoot = getenv('AV_PROJECT_HOME');
nnvRoot = getenv('NNV_ROOT');

nnvRoot = fullfile(nnvRoot);

if ~isfolder(nnvRoot)
    error('NNV folder not found: %s', nnvRoot)
end

% 3) Add NNV (and all subfolders) to your MATLAB path
addpath( genpath(nnvRoot) );

% savepath;

addpath(genpath(fullfile(projectRoot, '/node_verification/functions/')));
addpath(genpath(fullfile(projectRoot, '/node_verification/models/')));

% data = load(fullfile(projectRoot, 'data', 'cora_node.mat'));
data = load(fullfile(projectRoot, 'data', 'karate_node.mat'));
A_full     = data.edge_indices(:,:,1);    
X_full     = data.features(:,:,1);        
y_full     = double(data.labels(:)) + 1;  
numNodes   = size(X_full,1);

rng(2024);

disp(data);


[~, ~, idxTest] = trainingPartitions(numNodes, [0.4 0.35 0.25]);

[trainIdx, valIdx, testIdx] = trainingPartitions(numNodes, [0.4 0.35 0.25]);

sets = {'Train', 'Validation', 'Test'};
indices = {trainIdx, valIdx, testIdx};

for i = 1:length(sets)
    labels = y_full(indices{i});
    classes = unique(labels);
    counts = histc(labels, classes);
    fprintf('%s set class distribution:\n', sets{i});
    for j = 1:length(classes)
        fprintf('  Class %d: %d\n', classes(j), counts(j));
    end
end

adjacencyDataTest = A_full(idxTest, idxTest);
featureDataTest   = X_full(idxTest, :);
labelDataTest     = y_full(idxTest);


%% Verify models

% Study Variables
% seeds = [0,1,2,3,4]; % models
seeds = [0, 1, 2]; % models
num_features = size(featureDataTest, 2);
epsilons = [.00005, .0005, .005, .05]; % epsilons

% Verify one model at a time - using regular for loop instead of parfor to avoid file access issues
parfor e = 1:length(epsilons)
    for k = 1:length(seeds)
        % Construct the model path
        modelPath = "karate_node_gcn_" + string(seeds(k) + "_" + string(num_features));
    
        fprintf('Verifying model %s with epsilon %.5f\n', modelPath, epsilons(e));
        


        reach_model_Linf(modelPath, epsilons(e), adjacencyDataTest, featureDataTest, labelDataTest, num_features);

    end
end


