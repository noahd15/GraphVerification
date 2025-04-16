%% Data Loading and Partitioning
canUseGPU = false;

% Find the project root
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME environment variable is not set. Please set it to your project root directory.');
end

% Load in reduced data from PCA
dataFile = fullfile(projectRoot, 'data', 'cora_node.mat');
data = load(dataFile);
disp(data);

% Assume the data is stored as [numNodes x ... x 1]
adjacencyData = data.edge_indices;  % [numNodes x numNodes x 1]
featureData = data.features;          % [numNodes x featureDim x 1]
labelData = double(permute(data.labels, [2 1]));
labelData = labelData + 1;  % Adjust labels to start from 1 instead of 0

% Use the maximum label as number of classes
num_classes = max(labelData);

% Since there is only one graph, partition by node rather than by graph.
numNodes = size(featureData, 1);
rng(2024);  % For reproducibility

% Partition nodes
indices = randperm(numNodes);
nTrain = round(0.8 * numNodes);
nVal = round(0.1 * numNodes);
idxTrain = indices(1:nTrain);
idxValidation = indices(nTrain+1:nTrain+nVal);
idxTest = indices(nTrain+nVal+1:end);

% Create node-induced subgraphs (subselect rows/cols for the single graph)
adjTrain = adjacencyData(idxTrain, idxTrain, 1);
adjVal   = adjacencyData(idxValidation, idxValidation, 1);
adjTest  = adjacencyData(idxTest, idxTest, 1);

featuresTrain = featureData(idxTrain, :, 1);
featuresVal   = featureData(idxValidation, :, 1);
featuresTest  = featureData(idxTest, :, 1);

labelsTrain = labelData(idxTrain);
labelsVal   = labelData(idxValidation);
labelsTest  = labelData(idxTest);

fprintf('Shape of XTrain: %s\n', mat2str(size(featuresTrain)));

% Preprocess data (updated for single-graph processing)
[ATrain_full, XTrain_full, labelsTrain_full] = preprocessData(adjTrain, featuresTrain, labelsTrain, 'preprocessedPredictors_train_PCA.mat');
[AValidation, XValidation, labelsValidation] = preprocessData(adjVal, featuresVal, labelsVal, 'preprocessedPredictors_val_PCA.mat');
[ATest, XTest, labelsTest] = preprocessData(adjTest, featuresTest, labelsTest, 'preprocessedPredictors_test_PCA.mat');

% Define class mapping and compute class weights based on the full training set
classes = categories(labelsTrain_full);   % Classes as categorical strings
numClasses = numel(classes);
classList = categories(labelsTrain_full);
counts = countcats(labelsTrain_full);
classWeights = 1 ./ counts;
classWeights = classWeights / sum(classWeights) * numel(classList);
classWeights = classWeights(:)';

%% Network Initialization
seeds = [1];
for i = 1:length(seeds)
    seed = seeds(i);
    rng(seed);

    % Initialize network parameters structure
    parameters = struct;
    numHiddenFeatureMaps = 32;
    % Get the correct input feature dimension (features are [numNodes x featureDim])
    numInputFeatures = size(featureData, 2);
    fprintf('Input feature dimension: %d\n', numInputFeatures);

    % Layer 1 - First Graph Convolution
    sz = [numInputFeatures, numHiddenFeatureMaps];
    parameters.mult1.Weights = dlarray(initializeGlorot(sz, numHiddenFeatureMaps, numInputFeatures, "double"));

    % Layer 2 - Second Graph Convolution
    sz = [numHiddenFeatureMaps, numHiddenFeatureMaps];
    parameters.mult2.Weights = dlarray(initializeGlorot(sz, numHiddenFeatureMaps, numHiddenFeatureMaps, "double"));

    % Layer 3 - Third Graph Convolution (outputs directly to numClasses)
    sz = [numHiddenFeatureMaps, numClasses];
    parameters.mult3.Weights = dlarray(initializeGlorot(sz, numClasses, numHiddenFeatureMaps, "double"));

    %% Training Setup
    numEpochs = 200;
    learnRate = 0.001;
    validationFrequency = 10;  % Used here only for diagnostic printing

    % For mini-batch processing on node level
    numTrainNodes = numel(idxTrain);
    batchSize = 1024;
    numBatches = ceil(numTrainNodes / batchSize);

    trailingAvg = [];
    trailingAvgSq = [];

    % Initialize arrays to store metrics
    train_losses = zeros(numEpochs, 1);
    train_accs = zeros(numEpochs, 1);
    train_precision = zeros(numEpochs, 1);
    train_recall = zeros(numEpochs, 1);
    train_f1 = zeros(numEpochs, 1);

    val_losses = zeros(numEpochs, 1);
    val_accs = zeros(numEpochs, 1);
    val_precision = zeros(numEpochs, 1);
    val_recall = zeros(numEpochs, 1);
    val_f1 = zeros(numEpochs, 1);

    t = tic;
    globalStep = 0;  % Count mini-batch steps

    %% Training Loop (Mini-Batches)
    for epoch = 1:numEpochs
        % Shuffle training node indices each epoch
        shuffledIndices = randperm(numTrainNodes);
        epochLoss = 0;  % Accumulate loss over mini-batches

        for batch = 1:numBatches
            globalStep = globalStep + 1;
            startIdx = (batch-1)*batchSize + 1;
            endIdx = min(batch*batchSize, numTrainNodes);
            batchIndices = shuffledIndices(startIdx:endIdx);

            % Create mini-batch for nodes
            [A_batch, X_batch, labels_batch] = createMiniBatch(adjTrain, featuresTrain, labelsTrain, batchIndices);

            % Convert features to dlarray
            X_batch = dlarray(X_batch);

            % Convert numeric labels to categorical using fixed mapping.
            labels_batch = categorical(labels_batch(:), 1:numClasses, labelSymbol(1:numClasses));
            T_batch = onehotencode(labels_batch, 2, 'ClassNames', classes);
            
            
            if canUseGPU
                X_batch = gpuArray(X_batch);
            end

            % Evaluate model loss and gradients
            disp('Calculating loss and gradients...');
            [loss, gradients] = dlfeval(@modelLoss, parameters, X_batch, A_batch, T_batch, classWeights);
            disp('Loss and gradients calculated.');
            
            % Update parameters using Adam optimizer
            [parameters, trailingAvg, trailingAvgSq] = adamupdate(parameters, gradients, trailingAvg, trailingAvgSq, globalStep, learnRate);

            % Accumulate loss
            epochLoss = epochLoss + double(loss);
        end

        % Average loss over mini-batches for this epoch
        train_losses(epoch) = epochLoss / numBatches;

        % Evaluate on the full preprocessed training set for metrics
        XTrain_full = dlarray(XTrain_full);
        TTrain_full = onehotencode(labelsTrain_full(:), 2, 'ClassNames', classes);

        YTrain = model(parameters, XTrain_full, ATrain_full);
        YTrainClass = onehotdecode(YTrain, classes, 2);
        train_accs(epoch) = mean(YTrainClass == labelsTrain_full);
        [prec, rec, f1] = calculatePrecisionRecall(YTrainClass, labelsTrain_full);
        train_precision(epoch) = prec(end);
        train_recall(epoch) = rec(end);
        train_f1(epoch) = f1(end);

        % Evaluate on validation set
        TValidation = onehotencode(labelsValidation, 2, 'ClassNames', classes);
        YValidation = model(parameters, XValidation, AValidation);
        YvalClass = onehotdecode(YValidation, classes, 2);
        val_accs(epoch) = mean(YvalClass == labelsValidation);
        lossValidation = crossentropy(YValidation, TValidation, "DataFormat","BC");
        val_losses(epoch) = double(lossValidation);
        [prec_val, rec_val, f1_val] = calculatePrecisionRecall(YvalClass, labelsValidation);
        val_precision(epoch) = prec_val(end);
        val_recall(epoch) = rec_val(end);
        val_f1(epoch) = f1_val(end);

        % Print diagnostics for the epoch
        fprintf('\nEpoch %d/%d:\n', epoch, numEpochs);
        fprintf('TRAIN: Loss=%.4f  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f\n', ...
            train_losses(epoch), train_accs(epoch), train_precision(epoch), train_recall(epoch), train_f1(epoch));
        fprintf('VAL:   Loss=%.4f  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f\n', ...
            val_losses(epoch), val_accs(epoch), val_precision(epoch), val_recall(epoch), val_f1(epoch));
        fprintf('Time elapsed: %.2f seconds\n', toc(t));
    end

    %% Final Testing (once after training)
    TTest = onehotencode(labelsTest, 2, 'ClassNames', classes);
    YTest = model(parameters, XTest, ATest);
    YTestClass = onehotdecode(YTest, classes, 2);
    testAcc = mean(YTestClass == labelsTest);
    [precision, recall, f1] = calculatePrecisionRecall(YTestClass, labelsTest);

    fprintf('\n===== FINAL TEST RESULTS =====\n');
    fprintf('OVERALL METRICS:\n  Accuracy:  %.4f\n  Precision: %.4f\n  Recall:    %.4f\n  F1 Score:  %.4f\n', ...
        testAcc, precision(end), recall(end), f1(end));

    % Print per-class metrics
    fprintf('\nPER-CLASS METRICS:\n');
    fprintf('%-20s %-12s %-12s %-12s\n', 'Class', 'Precision', 'Recall', 'F1 Score');
    fprintf('%-20s %-12s %-12s %-12s\n', '-----', '---------', '------', '--------');
    classNames = labelSymbol(1:numClasses);
    for j = 1:length(classNames)
        if j <= length(precision)
            fprintf('%-20s %-12.4f %-12.4f %-12.4f\n', classNames(j), precision(j), recall(j), f1(j));
        end
    end

    % Create confusion matrix with enhanced labels and titles
    figure('Position', [100, 100, 800, 600]);
    cm = confusionchart(labelsTest, YTestClass, 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
    title("GCN Confusion Matrix");
    xlabel('Predicted Class');
    ylabel('True Class');
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], 'String', 'Column normalized: Each cell shows what percentage of predictions for a class were correct', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    annotation('textbox', [0.15, 0.95, 0.7, 0.05], 'String', 'Row normalized: Each cell shows what percentage of actual instances of a class were correctly predicted', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    % Save confusion matrix to results and logs directories
    resultsDir = fullfile(projectRoot, 'results');
    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end
    logsDir = fullfile(projectRoot, 'logs');
    if ~exist(logsDir, 'dir')
        mkdir(logsDir);
    end
    saveas(gcf, fullfile(resultsDir, 'batched_confusion_matrix.png'));
    saveas(gcf, fullfile(logsDir, 'batched_confusion_matrix.png'));

    % Plot training and validation metrics (function assumed to exist)
    plotTrainingMetrics(train_losses, val_losses, [], train_accs, val_accs, [], train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, [], [], [], validationFrequency);

    % Save the model and training logs
    save("models/node_gcn_" + string(seed) + ".mat", "testAcc", "parameters", "precision", "recall", "f1", ...
        "train_losses", "val_losses", "train_accs", "val_accs", "train_precision", "train_recall", "train_f1", ...
        "val_precision", "val_recall", "val_f1");
end

%% Create Mini-Batch for Single Graph (Node Classification)
function [A_batch, X_batch, labels_batch] = createMiniBatch(adjacencyData, featureData, labelData, batchIndices)
    % For node classification, create the submatrix induced by the batch indices.
    % Here, adjacencyData, featureData, and labelData are already the training/validation/test splits.
    A_batch = adjacencyData(batchIndices, batchIndices);
    X_batch = featureData(batchIndices, :);
    labels_batch = labelData(batchIndices);
end

%% Other Helper Functions

function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData, cacheFileName)
    projectRoot = getenv('AV_PROJECT_HOME');
    cacheFile = fullfile(projectRoot, 'data', cacheFileName);
    
    if exist(cacheFile, 'file')
        % Load the entire file so that MATLAB does not warn about missing variables.
        loadedData = load(cacheFile);
        
        if isfield(loadedData, 'labels')
            labels = loadedData.labels;
        else
            % If 'labels' wasn't saved previously, use the input labelData.
            labels = labelData;
        end
        
        % These variables are assumed to be present.
        adjacency = loadedData.adjacency;
        features = loadedData.features;
    else
        [adjacency, features] = preprocessPredictors(adjacencyData, featureData);
        labels = labelData;
        save(cacheFile, 'adjacency', 'features', 'labels', '-v7.3');
    end

    % Convert numeric labels to categorical with appropriate symbols.
    labelNumbers = unique(labels);
    labelNames = labelSymbol(labelNumbers);
    labels = categorical(labels, labelNumbers, labelNames);
end



function [adjacency, features] = preprocessPredictors(adjacencyData, featureData)
    % For a single graph, convert the adjacency matrix to sparse and leave features unchanged.
    adjacency = sparse(adjacencyData);
    features = featureData;
end

function sym = labelSymbol(labelNumbers)
    % Converts label numbers into string symbols.
    if iscategorical(labelNumbers)
        labelNumbers = double(labelNumbers);
    end
    sym = strings(size(labelNumbers));
    for k = 1:numel(labelNumbers)
        switch labelNumbers(k)
            case 1
                sym(k) = "Probabilistic_Methods";
            case 2
                sym(k) = "Neural_Networks";
            case 3
                sym(k) = "Rule_Learning";
            case 4
                sym(k) = "Case_Based";
            case 5
                sym(k) = "Reinforcement_Learning";
            case 6
                sym(k) = "Theory";
            case 7
                sym(k) = "Genetic_Algorithms";
            otherwise
                error("Invalid label number: %g. Supported labels are 1 through 7.", labelNumbers(k));
        end
    end
end

function Y = model(parameters, X, A)
    % Normalize the adjacency matrix, then apply a series of graph convolutions.
    ANorm = normalizeAdjacency(A);
    fprintf('conv1 shape: A x X x W: %s x %s x %s\n', mat2str(size(ANorm)), mat2str(size(X)), mat2str(size(parameters.mult1.Weights)));
    conv1 = ANorm * X * parameters.mult1.Weights;
    relu1 = relu(conv1);
    fprintf('conv2 shape: A x relu1 x W: %s x %s x %s\n', mat2str(size(ANorm)), mat2str(size(relu1)), mat2str(size(parameters.mult2.Weights)));
    conv2 = ANorm * relu1 * parameters.mult2.Weights;
    relu2 = relu(conv2);
    fprintf('conv3 shape: A x relu2 x W: %s x %s x %s\n', mat2str(size(ANorm)), mat2str(size(relu2)), mat2str(size(parameters.mult3.Weights)));
    conv3 = ANorm * relu2 * parameters.mult3.Weights;
    Y = softmax(conv3, "DataFormat","BC");
end

function [loss, gradients] = modelLoss(parameters, X, A, T, classWeights)
    % Ensure parameters are of dlarray class for automatic differentiation.
    if ~isa(parameters.mult1.Weights, 'dlarray')
        parameters.mult1.Weights = dlarray(parameters.mult1.Weights);
        parameters.mult2.Weights = dlarray(parameters.mult2.Weights);
        parameters.mult3.Weights = dlarray(parameters.mult3.Weights);
    end

    % Forward pass and loss computation
    Y = model(parameters, X, A);
    loss = crossentropy(Y, T, classWeights, "DataFormat","BC", "WeightsFormat","UC");
    gradients = dlgradient(loss, parameters);
end

function ANorm = normalizeAdjacency(A)
    % Add self-loops and symmetrically normalize the adjacency matrix.
    A = A + speye(size(A));
    degree = sum(A, 2);
    degreeInvSqrt = sqrt(1./degree);
    D = spdiags(degreeInvSqrt, 0, size(A,1), size(A,1));
    ANorm = D * A * D;
end

function weights = initializeGlorot(sz, fanOut, fanIn, dataType)
    % Initialize weights using Glorot initialization.
    stddev = sqrt(2 / (fanIn + fanOut));
    weights = stddev * randn(sz, dataType);
end

function [precision, recall, f1] = calculatePrecisionRecall(predictions, trueLabels)
    % Compute precision, recall, and F1 score per class.
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
    % Add macro-average as the last element.
    precision(end+1) = mean(precision(1:numClasses));
    recall(end+1) = mean(recall(1:numClasses));
    f1(end+1) = mean(f1(1:numClasses));
end
