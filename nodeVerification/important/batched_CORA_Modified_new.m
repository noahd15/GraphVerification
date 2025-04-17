% Cora GCN Training and Verification Demo
% Updated to use full graph adjacency and node‑level splits
% Modified to be more similar to the PCA training script

%% Setup
canUseGPU = false;

% Find the project root
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME environment variable is not set. Please set it to your project root directory.');
end

% Load Cora dataset
dataFile = fullfile(projectRoot, 'data', 'cora_node.mat');
data = load(dataFile);

% Full Cora graph
A_full = data.edge_indices(:,:,1);    % 2708×2708 sparse (possibly single)
X_full = data.features(:,:,1);        % 2708×featureDim
y_full = double(data.labels(:)) + 1;   % 2708×1, labels 1–7
[numNodes, featureDim] = size(X_full);

% Train/Val/Test node splits
rng(2024);
indices = randperm(numNodes);
nTrain = round(0.8 * numNodes);
nVal = round(0.1 * numNodes);
idxTrain = indices(1:nTrain);
idxValidation = indices(nTrain+1 : nTrain+nVal);
idxTest = indices(nTrain+nVal+1 : end);

% Convert labels to categorical for consistency with PCA script
classes = unique(y_full);
numClasses = length(classes);
classNames = string(1:numClasses);
y_train_cat = categorical(y_full(idxTrain), classes, classNames);
y_val_cat = categorical(y_full(idxValidation), classes, classNames);
y_test_cat = categorical(y_full(idxTest), classes, classNames);

% Calculate class weights for balanced training
counts = countcats(y_train_cat);
classWeights = 1 ./ counts;
classWeights = classWeights / sum(classWeights) * numel(classes);
classWeights = classWeights(:)';

%% Network Initialization
seeds = [1];
for i = 1:length(seeds)
    seed = seeds(i);
    rng(seed);

    % Initialize network parameters structure
    parameters = struct;
    numHiddenFeatureMaps = 32;
    fprintf('Input feature dimension: %d\n', featureDim);

    % Layer 1 - First Graph Convolution
    sz = [featureDim, numHiddenFeatureMaps];
    parameters.mult1.Weights = dlarray(initializeGlorot(sz, numHiddenFeatureMaps, featureDim, "double"));

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

    % For mini-batch processing, set a batch size (adjust as needed)
    batchSize = 1024;
    numBatches = ceil(nTrain / batchSize);

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
        localTrainIndices = idxTrain(randperm(nTrain));
        epochLoss = 0;  % Accumulate loss over mini-batches in this epoch

        for batch = 1:numBatches
            globalStep = globalStep + 1;
            startIdx = (batch-1)*batchSize + 1;
            endIdx = min(batch*batchSize, nTrain);
            batchIndices = localTrainIndices(startIdx:endIdx);

            % Create mini-batch
            [A_batch, X_batch, y_batch] = createMiniBatch(A_full, X_full, y_full, batchIndices);
            
            % Convert features to dlarray
            X_batch = dlarray(X_batch);
            
            % Convert labels to categorical and one-hot encode
            y_batch_cat = categorical(y_batch, classes, classNames);
            T_batch = onehotencode(y_batch_cat, 2, 'ClassNames', string(classes));

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

        % Evaluate on the full training set
        X_train_full = dlarray(X_full(idxTrain, :));
        A_train_full = A_full(idxTrain, idxTrain);
        T_train_full = onehotencode(y_train_cat, 2, 'ClassNames', string(classes));
        
        if canUseGPU
            X_train_full = gpuArray(X_train_full);
        end
        
        Y_train = model(parameters, X_train_full, A_train_full);
        Y_train_class = onehotdecode(Y_train, string(classes), 2);
        train_accs(epoch) = mean(Y_train_class == y_train_cat);
        [prec, rec, f1] = calculatePrecisionRecall(Y_train_class, y_train_cat);
        train_precision(epoch) = prec(end);
        train_recall(epoch) = rec(end);
        train_f1(epoch) = f1(end);

        % Evaluate on validation set
        X_val_full = dlarray(X_full(idxValidation, :));
        A_val_full = A_full(idxValidation, idxValidation);
        T_val_full = onehotencode(y_val_cat, 2, 'ClassNames', string(classes));
        
        if canUseGPU
            X_val_full = gpuArray(X_val_full);
        end
        
        Y_val = model(parameters, X_val_full, A_val_full);
        Y_val_class = onehotdecode(Y_val, string(classes), 2);
        val_accs(epoch) = mean(Y_val_class == y_val_cat);
        val_losses(epoch) = double(crossentropy(Y_val, T_val_full, DataFormat="BC"));
        [prec_val, rec_val, f1_val] = calculatePrecisionRecall(Y_val_class, y_val_cat);
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
    X_test_full = dlarray(X_full(idxTest, :));
    A_test_full = A_full(idxTest, idxTest);
    T_test_full = onehotencode(y_test_cat, 2, 'ClassNames', string(classes));
    
    if canUseGPU
        X_test_full = gpuArray(X_test_full);
    end
    
    Y_test = model(parameters, X_test_full, A_test_full);
    Y_test_class = onehotdecode(Y_test, string(classes), 2);
    testAcc = mean(Y_test_class == y_test_cat);
    [precision, recall, f1] = calculatePrecisionRecall(Y_test_class, y_test_cat);

    fprintf('\n===== FINAL TEST RESULTS =====\n');
    fprintf('OVERALL METRICS:\n  Accuracy:  %.4f\n  Precision: %.4f\n  Recall:    %.4f\n  F1 Score:  %.4f\n', ...
        testAcc, precision(end), recall(end), f1(end));

    % Print per-class metrics
    fprintf('\nPER-CLASS METRICS:\n');
    fprintf('%-20s %-12s %-12s %-12s\n', 'Class', 'Precision', 'Recall', 'F1 Score');
    fprintf('%-20s %-12s %-12s %-12s\n', '-----', '---------', '------', '--------');
    for j = 1:length(classes)
        if j <= length(precision)
            fprintf('%-20s %-12.4f %-12.4f %-12.4f\n', classNames(j), precision(j), recall(j), f1(j));
        end
    end

    % Create confusion matrix with enhanced labels and titles
    figure('Position', [100, 100, 800, 600]);
    cm = confusionchart(y_test_cat, Y_test_class, 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
    title("Cora GCN Confusion Matrix");
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
    saveas(gcf, fullfile(resultsDir, 'cora_confusion_matrix.png'));
    saveas(gcf, fullfile(logsDir, 'cora_confusion_matrix.png'));

    % Plot training and validation metrics over epochs
    plotTrainingMetrics(train_losses, val_losses, [], train_accs, val_accs, [], train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, [], [], [], validationFrequency);

    % Save the model and training logs
    save("models/cora_node_gcn_" + string(seed) + ".mat", "testAcc", "parameters", "precision", "recall", "f1", ...
        "train_losses", "val_losses", "train_accs", "val_accs", "train_precision", "train_recall", "train_f1", ...
        "val_precision", "val_recall", "val_f1");
end

%% Helper Functions
function [A_batch, X_batch, y_batch] = createMiniBatch(A_full, X_full, y_full, batchIndices)
    A_batch = A_full(batchIndices, batchIndices);
    X_batch = X_full(batchIndices, :);
    y_batch = y_full(batchIndices);
end

function Y = model(parameters, X, A)
    ANorm = normalizeAdjacency(A);
    conv1 = ANorm * X * parameters.mult1.Weights;
    relu1 = relu(conv1);
    conv2 = ANorm * relu1 * parameters.mult2.Weights;
    relu2 = relu(conv2);
    conv3 = ANorm * relu2 * parameters.mult3.Weights;
    Y = softmax(conv3, DataFormat="BC");
end

function [loss, gradients] = modelLoss(parameters, X, A, T, classWeights)
    % Make sure parameters are properly traced for automatic differentiation
    % Convert parameters to dlarray if they aren't already
    if ~isa(parameters.mult1.Weights, 'dlarray')
        parameters.mult1.Weights = dlarray(parameters.mult1.Weights);
        parameters.mult2.Weights = dlarray(parameters.mult2.Weights);
        parameters.mult3.Weights = dlarray(parameters.mult3.Weights);
    end

    % Forward pass through the model
    Y = model(parameters, X, A);

    % Calculate loss
    loss = crossentropy(Y, T, classWeights, DataFormat="BC", WeightsFormat="UC");

    % Calculate gradients
    gradients = dlgradient(loss, parameters);
end

function ANorm = normalizeAdjacency(A)
    % Ensure numeric type consistency (convert sparse single to double)
    if isa(A, 'single')
        A = double(A);
    end
    % Add self-loops
    A = A + speye(size(A));
    % Compute symmetric normalization
    d = sum(A, 2);
    D = spdiags(d.^(-0.5), 0, size(A,1), size(A,1));
    ANorm = D * A * D;
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
