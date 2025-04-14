%% Data Loading and Partitioning
canUseGPU = false;

projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME environment variable is not set. Please set it to your project root directory.');
end

dataFile = fullfile(projectRoot, 'data', 'node.mat');
data = load(dataFile);

featureData = data.features;
labelData = double(permute(data.labels, [2 1]));
if size(labelData, 1) < size(labelData, 2)
    labelData = labelData';
end
labelData = labelData + 1;  

rng(2024);
numGraphs = size(labelData, 1);

adjacencyData = edges2Adjacency(data);
numObservations = size(adjacencyData, 3);

[idxTrain, idxValidation, idxTest] = trainingPartitions(numObservations, [0.8 0.1 0.1]);

adjacencyDataTrain = adjacencyData(:, :, idxTrain);
featureDataTrain = featureData(:, :, idxTrain);
labelDataTrain = labelData(idxTrain, :);

[ AValidation, XValidation, labelsValidation ] = preprocessData(adjacencyData(:, :, idxValidation), featureData(:, :, idxValidation), labelData(idxValidation, :), 'preprocessedPredictors_val.mat');

[~, XTrain_full, labelsTrain_full] = preprocessData(adjacencyDataTrain, featureDataTrain, labelDataTrain, 'preprocessedPredictors_train.mat');
classes = categories(labelsTrain_full);  
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

    parameters = struct;
    numHiddenFeatureMaps = 32;
    numInputFeatures = size(XTrain_full, 2);  

    sz = [numInputFeatures, numHiddenFeatureMaps];
    parameters.mult1.Weights = initializeGlorot(sz, numHiddenFeatureMaps, numInputFeatures, "double");

    sz = [numHiddenFeatureMaps, numHiddenFeatureMaps];
    parameters.mult2.Weights = initializeGlorot(sz, numHiddenFeatureMaps, numHiddenFeatureMaps, "double");

    sz = [numHiddenFeatureMaps, numClasses];
    parameters.mult3.Weights = initializeGlorot(sz, numClasses, numHiddenFeatureMaps, "double");

    %% Training Setup
    numEpochs = 200;
    learnRate = 0.001;
    validationFrequency = 10;  

    batchSize = 256;
    numTrain = size(adjacencyDataTrain, 3);
    numBatches = ceil(numTrain / batchSize);

    trailingAvg = [];
    trailingAvgSq = [];

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
    globalStep = 0;  

    %% Training Loop (Mini-Batches)
    for epoch = 1:numEpochs
        localTrainIndices = 1:numTrain;
        shuffledIndices = localTrainIndices(randperm(numTrain));
        epochLoss = 0;  

        for batch = 1:numBatches
            globalStep = globalStep + 1;
            startIdx = (batch-1)*batchSize + 1;
            endIdx = min(batch*batchSize, numTrain);
            batchIndices = shuffledIndices(startIdx:endIdx);

            [A_batch, X_batch, labels_batch] = createMiniBatch(adjacencyDataTrain, featureDataTrain, labelDataTrain, batchIndices);

            X_batch = dlarray(X_batch);

            labels_batch = categorical(labels_batch, 1:numClasses, labelSymbol(1:numClasses));
            T_batch = onehotencode(labels_batch, 2, 'ClassNames', classes);

            if canUseGPU
                X_batch = gpuArray(X_batch);
            end

            [loss, gradients] = dlfeval(@modelLoss, parameters, X_batch, A_batch, T_batch, classWeights);

            [parameters, trailingAvg, trailingAvgSq] = adamupdate(parameters, gradients, trailingAvg, trailingAvgSq, globalStep, learnRate);

            epochLoss = epochLoss + double(loss);
        end

        train_losses(epoch) = epochLoss / numBatches;

        [ATrain_full, XTrain_full, labelsTrain_full] = preprocessData(adjacencyDataTrain, featureDataTrain, labelDataTrain, 'preprocessedPredictors_train.mat');
        XTrain_full = dlarray(XTrain_full);
        TTrain_full = onehotencode(labelsTrain_full, 2, 'ClassNames', classes);
        YTrain = model(parameters, XTrain_full, ATrain_full);
        YTrainClass = onehotdecode(YTrain, classes, 2);
        train_accs(epoch) = mean(YTrainClass == labelsTrain_full);
        [prec, rec, f1] = calculatePrecisionRecall(YTrainClass, labelsTrain_full);
        train_precision(epoch) = prec(end);
        train_recall(epoch) = rec(end);
        train_f1(epoch) = f1(end);

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

        fprintf('\nEpoch %d/%d:\n', epoch, numEpochs);
        fprintf('TRAIN: Loss=%.4f  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f\n', train_losses(epoch), train_accs(epoch), train_precision(epoch), train_recall(epoch), train_f1(epoch));
        fprintf('VAL:   Loss=%.4f  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f\n', val_losses(epoch), val_accs(epoch), val_precision(epoch), val_recall(epoch), val_f1(epoch));
        fprintf('Time elapsed: %.2f seconds\n', toc(t));
    end

    %% Final Testing (Performed only once after training)
    [ATest, XTest, labelsTest] = preprocessData(adjacencyData(:, :, idxTest), featureData(:, :, idxTest), labelData(idxTest, :), 'preprocessedPredictors_test.mat');
    TTest = onehotencode(labelsTest, 2, 'ClassNames', classes);
    YTest = model(parameters, XTest, ATest);
    YTestClass = onehotdecode(YTest, classes, 2);
    testAcc = mean(YTestClass == labelsTest);
    [precision, recall, f1] = calculatePrecisionRecall(YTestClass, labelsTest);

    fprintf('\n===== FINAL TEST RESULTS =====\n');
    fprintf('OVERALL METRICS:\n  Accuracy:  %.4f\n  Precision: %.4f\n  Recall:    %.4f\n  F1 Score:  %.4f\n', ...
        testAcc, precision(end), recall(end), f1(end));

    fprintf('\nPER-CLASS METRICS:\n');
    fprintf('%-20s %-12s %-12s %-12s\n', 'Class', 'Precision', 'Recall', 'F1 Score');
    fprintf('%-20s %-12s %-12s %-12s\n', '-----', '---------', '------', '--------');
    classNames = labelSymbol(1:numClasses);
    for j = 1:length(classNames)
        if j <= length(precision)
            fprintf('%-20s %-12.4f %-12.4f %-12.4f\n', classNames(j), precision(j), recall(j), f1(j));
        end
    end

    figure('Position', [100, 100, 800, 600]);
    cm = confusionchart(labelsTest, YTestClass, 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
    title("GCN Confusion Matrix");
    xlabel('Predicted Class');
    ylabel('True Class');
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], 'String', 'Column normalized: Each cell shows what percentage of predictions for a class were correct', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    annotation('textbox', [0.15, 0.95, 0.7, 0.05], 'String', 'Row normalized: Each cell shows what percentage of actual instances of a class were correctly predicted', 'EdgeColor', 'none', 'HorizontalAlignment', 'center');

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

    plotTrainingMetrics(train_losses, val_losses, [], train_accs, val_accs, [], train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, [], [], [], validationFrequency);

    save("models/node_gcn_" + string(seed) + ".mat", "testAcc", "parameters", "precision", "recall", "f1", ...
        "train_losses", "val_losses", "train_accs", "val_accs", "train_precision", "train_recall", "train_f1", ...
        "val_precision", "val_recall", "val_f1");
end

%% Helper Function: Create Mini-Batch
function [A_batch, X_batch, labels_batch] = createMiniBatch(adjacencyData, featureData, labelData, batchIndices)
    A_batch = sparse([]);
    X_batch = [];
    labels_batch = [];

    for j = 1:length(batchIndices)
        idx = batchIndices(j);
        numNodes = find(any(adjacencyData(:, :, idx)), 1, "last");
        if isempty(numNodes) || numNodes == 0
            continue;
        end
        A = adjacencyData(1:numNodes, 1:numNodes, idx);
        X = featureData(1:numNodes, :, idx);
        T = labelData(idx, 1:numNodes);
        A_batch = blkdiag(A_batch, A);
        X_batch = [X_batch; X];
        labels_batch = [labels_batch; T(:)];
    end
end

%% (The existing helper functions remain unchanged)
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
    Z1 = X;
    Z2 = relu(ANorm * Z1 * parameters.mult1.Weights);
    Z3 = relu(ANorm * Z2 * parameters.mult2.Weights);
    Z4 = ANorm * Z3 * parameters.mult3.Weights;
    Y = softmax(Z4, DataFormat="BC");
end

function [loss, gradients] = modelLoss(parameters, X, A, T, classWeights)
    Y = model(parameters, X, A);
    loss = crossentropy(Y, T, classWeights, DataFormat="BC", WeightsFormat="UC");
    gradients = dlgradient(loss, parameters);
end

function ANorm = normalizeAdjacency(A)
    A = A + speye(size(A));
    degree = sum(A, 2);
    degreeInvSqrt = sparse(sqrt(1./degree));
    ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);
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
    precision(end+1) = mean(precision(1:numClasses));
    recall(end+1) = mean(recall(1:numClasses));
    f1(end+1) = mean(f1(1:numClasses));
end