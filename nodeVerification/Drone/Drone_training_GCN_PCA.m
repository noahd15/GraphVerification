%% Data Loading and Partitioning
canUseGPU = false;

% Find the project root
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME environment variable is not set. Please set it to your project root directory.');
end

% Load in reduced data from PCA
dataFile = fullfile(projectRoot, 'data/reducedDatasetNode.mat');
data = load(dataFile);

disp(data);

% Extract feature and label data
featureData = data.featureData_reduced; % Use the reduced features
labelData = data.labelData;
adjacencyData = data.adjacencyData;

whos adjacencyData
whos featureData
whos labelData

% Get the indices for train, validation, and test sets
idxTrain = data.idxTrain;
idxValidation = data.idxValidation;
idxTest = data.idxTest;

rng(2024);
numGraphs = size(labelData, 1);
numObservations = size(adjacencyData, 3);

% Extract training, validation, and test data
adjacencyDataTrain = adjacencyData(:, :, idxTrain);
featureDataTrain = featureData(:, :, idxTrain);
labelDataTrain = labelData(idxTrain, :);

% Use different cache filenames for the reduced features
[~, XTrain_full, labelsTrain_full] = preprocessData(adjacencyDataTrain, featureDataTrain, labelDataTrain, 'preprocessedPredictors_train_PCA.mat');
[AValidation, XValidation, labelsValidation ] = preprocessData(adjacencyData(:, :, idxValidation), featureData(:, :, idxValidation), labelData(idxValidation, :), 'preprocessedPredictors_val_PCA.mat');
[ATest, XTest, labelsTest] = preprocessData(adjacencyData(:, :, idxTest), featureData(:, :, idxTest), labelData(idxTest, :), 'preprocessedPredictors_test_PCA.mat');

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
    % Get the correct input feature dimension from the reduced features
    [~, numInputFeatures, ~] = size(featureData);
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

    % % Layer 4 - Final Linear Layer (not used in this version)
    % sz = [numHiddenFeatureMaps, numClasses];
    % parameters.fc.Weights = dlarray(initializeGlorot(sz, numClasses, numHiddenFeatureMaps, "double"));
    % parameters.fc.Bias = dlarray(zeros(1, numClasses, "double"));

    %% Training Setup
    numEpochs = 50;
    learnRate = 0.001;
    validationFrequency = 10;  % Used here only for diagnostic printing

    % For mini-batch processing, set a batch size (adjust as needed)
    batchSize = 256;
    numTrain = size(adjacencyDataTrain, 3);
    numBatches = ceil(numTrain / batchSize);

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
        % Shuffle training indices each epoch
        localTrainIndices = 1:numTrain;
        shuffledIndices = localTrainIndices(randperm(numTrain));
        epochLoss = 0;  % Accumulate loss over mini-batches in this epoch

        for batch = 1:numBatches
            globalStep = globalStep + 1;
            startIdx = (batch-1)*batchSize + 1;
            endIdx = min(batch*batchSize, numTrain);
            batchIndices = shuffledIndices(startIdx:endIdx);

            % Create mini-batch using helper function
            [A_batch, X_batch, labels_batch] = createMiniBatch(adjacencyDataTrain, featureDataTrain, labelDataTrain, batchIndices);

            % Convert features to dlarray
            X_batch = dlarray(X_batch);

            % For mini-batch, the labels are numeric. Convert them to categorical using a fixed mapping.
            labels_batch = categorical(labels_batch, 1:numClasses, labelSymbol(1:numClasses));
            T_batch = onehotencode(labels_batch, 2, 'ClassNames', classes);

            if canUseGPU
                X_batch = gpuArray(X_batch);
            end

            % Evaluate model loss and gradients
            [loss, gradients] = dlfeval(@modelLoss, parameters, X_batch, A_batch, T_batch, classWeights);

            % Update parameters using Adam optimizer
            [parameters, trailingAvg, trailingAvgSq] = adamupdate(parameters, gradients, trailingAvg, trailingAvgSq, globalStep, learnRate);

            % Accumulate loss (for averaging later)
            epochLoss = epochLoss + double(loss);
        end

        % Compute average loss over mini-batches for this epoch
        train_losses(epoch) = epochLoss / numBatches;

        % Evaluate on the full preprocessed training set to compute training metrics
        [ATrain_full, XTrain_full, labelsTrain_full] = preprocessData(adjacencyDataTrain, featureDataTrain, labelDataTrain, 'preprocessedPredictors_train_PCA.mat');
        XTrain_full = dlarray(XTrain_full);
        % labelsTrain_full is already categorical, so no extra conversion is needed.
        TTrain_full = onehotencode(labelsTrain_full, 2, 'ClassNames', classes);
        YTrain = model(parameters, XTrain_full, ATrain_full);
        YTrainClass = onehotdecode(YTrain, classes, 2);
        train_accs(epoch) = mean(YTrainClass == labelsTrain_full);
        [prec, rec, f1] = calculatePrecisionRecall(YTrainClass, labelsTrain_full);
        train_precision(epoch) = prec(end);
        train_recall(epoch) = rec(end);
        train_f1(epoch) = f1(end);

        % Evaluate on validation set (validation labels are already categorical from preprocessing)
        TValidation = onehotencode(labelsValidation, 2, 'ClassNames', classes);
        YValidation = model(parameters, XValidation, AValidation);
        YvalClass = onehotdecode(YValidation, classes, 2);
        val_accs(epoch) = mean(YvalClass == labelsValidation);
        lossValidation = crossentropy(YValidation, TValidation, DataFormat="BC");
        val_losses(epoch) = double(lossValidation);
        [prec_val, rec_val, f1_val] = calculatePrecisionRecall(YvalClass, labelsValidation);
        val_precision(epoch) = prec_val(end);
        val_recall(epoch) = rec_val(end);
        val_f1(epoch) = f1_val(end);

        % Print epoch diagnostics
        fprintf('\nEpoch %d/%d:\n', epoch, numEpochs);
        fprintf('TRAIN: Loss=%.4f  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f\n', ...
            train_losses(epoch), train_accs(epoch), train_precision(epoch), train_recall(epoch), train_f1(epoch));
        fprintf('VAL:   Loss=%.4f  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f\n', ...
            val_losses(epoch), val_accs(epoch), val_precision(epoch), val_recall(epoch), val_f1(epoch));
        fprintf('Time elapsed: %.2f seconds\n', toc(t));
    end

    %% Final Testing (Performed only once after training)
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
    % Use fixed mapping for class names
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
    projectRoot = getenv('AV_PROJECT_HOME');
    if isempty(projectRoot)
        projectRoot = pwd;
    end
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

    % Plot training and validation metrics over epochs (if desired)
    plotTrainingMetrics(train_losses, val_losses, [], train_accs, val_accs, [], train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, [], [], [], validationFrequency);

    % Save the model and training logs
    save("models/drone_node_gcn_" + string(seed) + ".mat", "testAcc", "parameters", "precision", "recall", "f1", ...
        "train_losses", "val_losses", "train_accs", "val_accs", "train_precision", "train_recall", "train_f1", ...
        "val_precision", "val_recall", "val_f1");
end

%% Create Mini-Batch
function [A_batch, X_batch, labels_batch] = createMiniBatch(adjacencyData, featureData, labelData, batchIndices)
    % Initialize empty matrices for this mini-batch
    A_batch = sparse([]);
    X_batch = [];
    labels_batch = [];

    for j = 1:length(batchIndices)
        idx = batchIndices(j);
        % Determine the number of nodes in the current graph
        numNodes = find(any(adjacencyData(:, :, idx)), 1, "last");
        if isempty(numNodes) || numNodes == 0
            continue;
        end
        A = adjacencyData(1:numNodes, 1:numNodes, idx);
        X = featureData(1:numNodes, :, idx);
        T = labelData(idx, 1:numNodes);
        % Append this graph's data to the mini-batch
        A_batch = blkdiag(A_batch, A);
        X_batch = [X_batch; X];
        labels_batch = [labels_batch; T(:)];
    end
end

%% Other Helper Functions
function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData, cacheFileName)
    projectRoot = getenv('AV_PROJECT_HOME');
    cacheFile = fullfile(projectRoot, 'data', cacheFileName);
    if exist(cacheFile, 'file')
        load(cacheFile, 'adjacency', 'features');
    else
        [adjacency, features] = preprocessPredictors(adjacencyData, featureData);
        save(cacheFile, 'adjacency', 'features', '-v7.3');
    end

    labels = [];
    for i = 1:size(adjacencyData, 3)
        numNodes = find(any(adjacencyData(:, :, i)), 1, "last");
        if isempty(numNodes)
            numNodes = 0;
        end
        T = labelData(i, 1:numNodes);
        labels = [labels; T(:)];
    end

    labelNumbers = unique(labels);
    labelNames = labelSymbol(labelNumbers);
    labels = categorical(labels, labelNumbers, labelNames);
end

function [adjacency, features] = preprocessPredictors(adjacencyData, featureData)
    adjacency = sparse([]);
    features = [];
    for i = 1:size(adjacencyData, 3)
        numNodes = find(any(adjacencyData(:, :, i)), 1, "last");
        if isempty(numNodes) || numNodes == 0
            continue;
        end
        A = adjacencyData(1:numNodes, 1:numNodes, i);
        X = featureData(1:numNodes, :, i);
        adjacency = blkdiag(adjacency, A);
        features = [features; X];
        if mod(i, 500) == 0
            fprintf('Processing graph %d\n', i);
        end
    end
end

function sym = labelSymbol(labelNumbers)
    if iscategorical(labelNumbers)
        labelNumbers = double(labelNumbers);
    end
    sym = strings(size(labelNumbers));
    for k = 1:numel(labelNumbers)
        switch labelNumbers(k)
            case 1
                sym(k) = "Not Compromised";
            case 2
                sym(k) = "Compromised";
            case 3
                sym(k) = "Highly Compromised";
            otherwise
                error("Invalid label number: %g. Supported labels are 0,1,2,3.", labelNumbers(k));
        end
    end
end

function Y = model(parameters, X, A)
    ANorm = normalizeAdjacency(A);
    % fprintf('Conv1 size: A x X x W1 = %s x %s x %s\n', mat2str(size(ANorm)), mat2str(size(X)), mat2str(size(parameters.mult1.Weights)));
    conv1 = ANorm * X * parameters.mult1.Weights;
    relu1 = relu(conv1);
    % fprintf('Conv2 size: A x relu1 x W2 = %s x %s x %s\n', mat2str(size(ANorm)), mat2str(size(relu1)), mat2str(size(parameters.mult2.Weights)));
    conv2 = ANorm * relu1 * parameters.mult2.Weights;
    relu2 = relu(conv2);
    % fprintf('Conv3 size: A x relu2 x W3 = %s x %s x %s\n', mat2str(size(ANorm)), mat2str(size(relu2)), mat2str(size(parameters.mult3.Weights)));
    conv3 = ANorm * relu2 * parameters.mult3.Weights;
    Y = softmax(conv3, DataFormat="BC");

    % lin1 = conv3 * parameters.fc.Weights + parameters.fc.Bias;
    % Y = softmax(lin1, DataFormat="BC");

end

function [loss, gradients] = modelLoss(parameters, X, A, T, classWeights)
    % Make sure parameters are properly traced for automatic differentiation
    % Convert parameters to dlarray if they aren't already
    if ~isa(parameters.mult1.Weights, 'dlarray')
        parameters.mult1.Weights = dlarray(parameters.mult1.Weights);
        parameters.mult2.Weights = dlarray(parameters.mult2.Weights);
        parameters.mult3.Weights = dlarray(parameters.mult3.Weights);
        % parameters.fc.Weights = dlarray(parameters.fc.Weights);
        % parameters.fc.Bias = dlarray(parameters.fc.Bias);
    end

    % Forward pass through the model
    Y = model(parameters, X, A);

    % Calculate loss
    loss = crossentropy(Y, T, classWeights, DataFormat="BC", WeightsFormat="UC");

    % Calculate gradients
    gradients = dlgradient(loss, parameters);
end

function ANorm = normalizeAdjacency(A)
    A = A + speye(size(A));
    degree = sum(A, 2);
    degreeInvSqrt = sparse(sqrt(1./degree));
    ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);
end

function weights = initializeGlorot(sz, fanOut, fanIn, dataType)
    % Initialize weights using Glorot initialization
    % This helps with training deep networks by keeping the variance of activations
    % roughly the same across layers
    stddev = sqrt(2 / (fanIn + fanOut));
    weights = stddev * randn(sz, dataType);
end

function [precision, recall, f1] = calculatePrecisionRecall(predictions, trueLabels)
    % Get unique classes
    classes = categories(trueLabels);
    numClasses = numel(classes);

    % Initialize arrays
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1 = zeros(numClasses, 1);

    % Calculate metrics for each class
    for i = 1:numClasses
        % True positives: predicted class i and actual class i
        truePositives = sum(predictions == classes(i) & trueLabels == classes(i));

        % False positives: predicted class i but not actual class i
        falsePositives = sum(predictions == classes(i) & trueLabels ~= classes(i));

        % False negatives: not predicted class i but actual class i
        falseNegatives = sum(predictions ~= classes(i) & trueLabels == classes(i));

        % Calculate precision: TP / (TP + FP)
        if (truePositives + falsePositives) > 0
            precision(i) = truePositives / (truePositives + falsePositives);
        else
            precision(i) = 0;
        end

        % Calculate recall: TP / (TP + FN)
        if (truePositives + falseNegatives) > 0
            recall(i) = truePositives / (truePositives + falseNegatives);
        else
            recall(i) = 0;
        end

        % Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        if (precision(i) + recall(i)) > 0
            f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        else
            f1(i) = 0;
        end
    end

    % Add macro average as the last element
    precision(end+1) = mean(precision(1:numClasses));
    recall(end+1) = mean(recall(1:numClasses));
    f1(end+1) = mean(f1(1:numClasses));
end