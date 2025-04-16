%% Non‐Batched GCN Training on Cora Data (Revised)
% clear; close all; clc;

%% Setup and Data Loading
% Ensure the project root environment variable is set
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME environment variable is not set. Please set it to your project root directory.');
end

% Load the Cora data file (adjust the path as needed)
dataFile = fullfile(projectRoot, 'data', 'cora_node.mat');
data = load(dataFile);
disp(data);

% Load graph information
adjacencyData = data.edge_indices;  % [numNodes x numNodes]
featureData   = data.features;        % [numNodes x featureDim]
% Convert labels to a row vector and adjust to start from 1
labelData = double(permute(data.labels, [2 1]));
labelData = labelData + 1;

%% Partition the Data
numNodes = size(featureData, 1);
rng(2024);  % For reproducibility
indices = randperm(numNodes);
nTrain = round(0.8 * numNodes);
nVal   = round(0.1 * numNodes);
idxTrain = indices(1:nTrain);
idxValidation = indices(nTrain+1:nTrain+nVal);
idxTest = indices(nTrain+nVal+1:end);

% Create node-induced subgraphs
adjTrain = adjacencyData(idxTrain, idxTrain);
adjVal   = adjacencyData(idxValidation, idxValidation);
adjTest  = adjacencyData(idxTest, idxTest);

featuresTrain = featureData(idxTrain, :);
featuresVal   = featureData(idxValidation, :);
featuresTest  = featureData(idxTest, :);

labelsTrain = labelData(idxTrain);
labelsVal   = labelData(idxValidation);
labelsTest  = labelData(idxTest);

fprintf('Shape of XTrain: %s\n', mat2str(size(featuresTrain)));

%% Preprocess Data (Caching included)
[ATrain_full, XTrain_full, labelsTrain_full] = preprocessData(adjTrain, featuresTrain, labelsTrain, 'preprocessedPredictors_train_PCA.mat');
[AValidation, XValidation, labelsValidation] = preprocessData(adjVal, featuresVal, labelsVal, 'preprocessedPredictors_val_PCA.mat');
[ATest, XTest, labelsTest] = preprocessData(adjTest, featuresTest, labelsTest, 'preprocessedPredictors_test_PCA.mat');

% Define classes from categorical training labels
classes = categories(labelsTrain_full);
numClasses = numel(classes);

% Use onehotencode on the row-vector labels along dimension 1.
TTrain_full = onehotencode(labelsTrain_full, 1, 'ClassNames', classes);  % [numClasses x nTrain]

%% Network Initialization
rng(1);  % For reproducibility
parameters = struct;
numHiddenFeatureMaps = 32;
numInputFeatures = size(featureData, 2);
fprintf('Input feature dimension: %d\n', numInputFeatures);

% Initialize weights with Glorot initialization
parameters.mult1.Weights = dlarray(initializeGlorot([numInputFeatures, numHiddenFeatureMaps], numHiddenFeatureMaps, numInputFeatures, "double"));
parameters.mult2.Weights = dlarray(initializeGlorot([numHiddenFeatureMaps, numHiddenFeatureMaps], numHiddenFeatureMaps, numHiddenFeatureMaps, "double"));
parameters.mult3.Weights = dlarray(initializeGlorot([numHiddenFeatureMaps, numClasses], numClasses, numHiddenFeatureMaps, "double"));

%% Training Setup (Non‐Batched)
numEpochs = 200;
learnRate = 0.001;
canUseGPU = false;  % Set true if you have a supported GPU

% Preallocate arrays for metrics
train_losses  = zeros(numEpochs, 1);
train_accs    = zeros(numEpochs, 1);
train_precision = zeros(numEpochs, 1);
train_recall  = zeros(numEpochs, 1);
train_f1      = zeros(numEpochs, 1);
val_losses    = zeros(numEpochs, 1);
val_accs      = zeros(numEpochs, 1);
val_precision = zeros(numEpochs, 1);
val_recall    = zeros(numEpochs, 1);
val_f1        = zeros(numEpochs, 1);

trailingAvg = [];
trailingAvgSq = [];
globalStep = 0;

% Convert full training features to dlarray.
XTrain_full_dl = dlarray(XTrain_full);
if canUseGPU
    XTrain_full_dl = gpuArray(XTrain_full_dl);
end

%% Training Loop (Full Graph)
tStart = tic;
for epoch = 1:numEpochs
    globalStep = globalStep + 1;
    
    % Calculate loss and gradients on the full training set
    [loss, gradients] = dlfeval(@modelLoss, parameters, XTrain_full_dl, ATrain_full, TTrain_full, []);
    % Update parameters using Adam optimizer
    [parameters, trailingAvg, trailingAvgSq] = adamupdate(parameters, gradients, trailingAvg, trailingAvgSq, globalStep, learnRate);
    
    train_losses(epoch) = double(loss);
    
    % Evaluate on the training set
    YTrain = model(parameters, XTrain_full_dl, ATrain_full);
    % Decode predictions along dimension 1 to match onehotencode
    YTrainClass = onehotdecode(YTrain, classes, 1);
    train_accs(epoch) = mean(YTrainClass == labelsTrain_full);
    [prec, rec, f1] = calculatePrecisionRecall(YTrainClass, labelsTrain_full);
    train_precision(epoch) = prec(end);
    train_recall(epoch) = rec(end);
    train_f1(epoch) = f1(end);
    
    % Evaluate on the validation set
    XVal_dl = dlarray(XValidation);
    if canUseGPU
        XVal_dl = gpuArray(XVal_dl);
    end
    YValidation = model(parameters, XVal_dl, AValidation);
    YValClass = onehotdecode(YValidation, classes, 1);
    val_accs(epoch) = mean(YValClass == labelsValidation);
    TValidation = onehotencode(labelsValidation, 1, 'ClassNames', classes);
    lossValidation = crossentropy(YValidation, TValidation, "DataFormat", "CB");
    val_losses(epoch) = double(lossValidation);
    [prec_val, rec_val, f1_val] = calculatePrecisionRecall(YValClass, labelsValidation);
    val_precision(epoch) = prec_val(end);
    val_recall(epoch) = rec_val(end);
    val_f1(epoch) = f1_val(end);
    
    fprintf('Epoch %d/%d: TRAIN Loss=%.4f, Acc=%.4f | VAL Loss=%.4f, Acc=%.4f, Time=%.2fs\n', ...
        epoch, numEpochs, train_losses(epoch), train_accs(epoch), val_losses(epoch), val_accs(epoch), toc(tStart));
end

%% Final Testing
XTest_dl = dlarray(XTest);
if canUseGPU
    XTest_dl = gpuArray(XTest_dl);
end
YTest = model(parameters, XTest_dl, ATest);
YTestClass = onehotdecode(YTest, classes, 1);
testAcc = mean(YTestClass == labelsTest);
[precision, recall, f1] = calculatePrecisionRecall(YTestClass, labelsTest);

fprintf('\n===== FINAL TEST RESULTS =====\n');
fprintf('Accuracy:  %.4f\nPrecision: %.4f\nRecall:    %.4f\nF1 Score:  %.4f\n', ...
    testAcc, precision(end), recall(end), f1(end));

%% Save the Model and Logs
save(fullfile(projectRoot, 'models', 'node_gcn_full.mat'), "testAcc", "parameters", "precision", "recall", "f1", ...
    "train_losses", "val_losses", "train_accs", "val_accs", "train_precision", "train_recall", "train_f1", ...
    "val_precision", "val_recall", "val_f1");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper Functions

function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData, cacheFileName)
    projectRoot = getenv('AV_PROJECT_HOME');
    cacheFile = fullfile(projectRoot, 'data', cacheFileName);
    if exist(cacheFile, 'file')
        loadedData = load(cacheFile);
        if isfield(loadedData, 'labels')
            labels = loadedData.labels;
        else
            labels = labelData;
        end
        adjacency = loadedData.adjacency;
        features = loadedData.features;
    else
        [adjacency, features] = preprocessPredictors(adjacencyData, featureData);
        labels = labelData;
        save(cacheFile, 'adjacency', 'features', 'labels', '-v7.3');
    end
    labelNumbers = unique(labels);
    labelNames = labelSymbol(labelNumbers);
    labels = categorical(labels, labelNumbers, labelNames);
end

function [adjacency, features] = preprocessPredictors(adjacencyData, featureData)
    % Convert the adjacency matrix to a sparse matrix.
    adjacency = sparse(adjacencyData);
    features = featureData;
end

function sym = labelSymbol(labelNumbers)
    if iscategorical(labelNumbers)
        labelNumbers = double(labelNumbers);
    end
    sym = strings(size(labelNumbers));
    for k = 1:numel(labelNumbers)
        switch labelNumbers(k)
            case 1, sym(k) = "Probabilistic_Methods";
            case 2, sym(k) = "Neural_Networks";
            case 3, sym(k) = "Rule_Learning";
            case 4, sym(k) = "Case_Based";
            case 5, sym(k) = "Reinforcement_Learning";
            case 6, sym(k) = "Theory";
            case 7, sym(k) = "Genetic_Algorithms";
            otherwise, error("Invalid label number: %g. Supported labels are 1 through 7.", labelNumbers(k));
        end
    end
end

function Y = model(parameters, X, A)
    % Normalize the adjacency matrix and convert it to a full double array.
    ANorm = normalizeAdjacency(A);
    % Convert the normalized adjacency to dlarray so that all operands are compatible.
    ANorm = dlarray(ANorm);
    
    % Perform three graph convolution layers.
    conv1 = ANorm * X * parameters.mult1.Weights;
    relu1 = relu(conv1);
    conv2 = ANorm * relu1 * parameters.mult2.Weights;
    relu2 = relu(conv2);
    conv3 = ANorm * relu2 * parameters.mult3.Weights;
    % Use data format "CB": first dimension = classes, second = samples.
    Y = softmax(conv3, "DataFormat", "CB");
end

function [loss, gradients] = modelLoss(parameters, X, A, T, ~)
    % Ensure network parameters are dlarray objects.
    if ~isa(parameters.mult1.Weights, 'dlarray')
        parameters.mult1.Weights = dlarray(parameters.mult1.Weights);
        parameters.mult2.Weights = dlarray(parameters.mult2.Weights);
        parameters.mult3.Weights = dlarray(parameters.mult3.Weights);
    end
    Y = model(parameters, X, A);
    loss = crossentropy(Y, T, "DataFormat", "CB");
    gradients = dlgradient(loss, parameters);
end

function ANorm = normalizeAdjacency(A)
    % Add self-loops to the adjacency matrix.
    A = A + speye(size(A));
    
    % Compute degree and inverse square root of degree.
    degree = sum(A, 2);
    degreeInvSqrt = sqrt(1 ./ degree);
    
    % Create a diagonal sparse matrix D.
    D = spdiags(degreeInvSqrt, 0, size(A,1), size(A,1));
    
    % Estimate the memory needed for a full dense m x m matrix.
    m = size(A,1);
    denseElemCount = m * m;
    bytesPerElem = 8;  % Each element (double precision) uses 8 bytes.
    estimatedBytes = denseElemCount * bytesPerElem;
    estimatedMB = estimatedBytes / (1024^2);
    
    % Print the estimated memory requirement.
    fprintf('Estimated memory to hold the dense normalized adjacency matrix: %.2f MB\n', estimatedMB);
    
    % Convert normalized adjacency to a full array.
    ANorm = full(D * A * D);
end


function weights = initializeGlorot(sz, fanOut, fanIn, dataType)
    stddev = sqrt(2 / (fanIn + fanOut));
    weights = stddev * randn(sz, dataType);
end

function [precision, recall, f1] = calculatePrecisionRecall(predictions, trueLabels)
    classes = categories(trueLabels);
    numClasses = numel(classes);
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1 = zeros(numClasses, 1);
    for i = 1:numClasses
        truePositives = sum(predictions == classes(i) & trueLabels == classes(i));
        falsePositives = sum(predictions == classes(i) & trueLabels ~= classes(i));
        falseNegatives = sum(predictions ~= classes(i) & trueLabels == classes(i));
        if (truePositives + falsePositives) > 0
            precision(i) = truePositives / (truePositives + falsePositives);
        else
            precision(i) = 0;
        end
        if (truePositives + falseNegatives) > 0
            recall(i) = truePositives / (truePositives + falseNegatives);
        else
            recall(i) = 0;
        end
        if (precision(i) + recall(i)) > 0
            f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        else
            f1(i) = 0;
        end
    end
    % Append macro-average as the last element.
    precision(end+1) = mean(precision(1:numClasses));
    recall(end+1) = mean(recall(1:numClasses));
    f1(end+1) = mean(f1(1:numClasses));
end
