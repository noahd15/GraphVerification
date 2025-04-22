canUseGPU = false;

projectRoot = getenv('AV_PROJECT_HOME');
addpath(genpath(fullfile(projectRoot, '/node_verification/functions/')));

if isempty(projectRoot)
    error('Set AV_PROJECT_HOME to your project root.');
end

% Load original data
raw = load(fullfile(projectRoot, 'data', 'karate_node.mat'));
features = raw.features;
adjacency = raw.edge_indices;
labels = raw.labels;

% Ensure feature size is exactly the number of features in karate club
featureDim = size(features, 2); % Number of features

num_features = featureDim;

% Get number of nodes
numNodes = size(features, 1);

% Ensure consistent split
rng(2024);
[idxTrain, idxValidation, idxTest] = trainingPartitions(numNodes, [0.4 0.35 0.25]);

% PCA(adjacency, features, labels, idxTrain, idxValidation, idxTest, num_features, 'reduced_dataset.mat', 1e-4);

% Load reduced dataset
data = raw;
A_full = data.edge_indices(:,:,1);       % 2708×2708
X_full = double(data.features);          % 2708×featureDim  
y_full = double(data.labels(:)) + 1;     % 2708×1

% Create label categories using same split
classes       = unique(y_full);
numClasses    = numel(classes);
disp(numClasses)
classNames    = string(1:numClasses);

y_train_cat   = categorical(y_full(idxTrain), classes, classNames);
y_val_cat     = categorical(y_full(idxValidation), classes, classNames);
y_test_cat    = categorical(y_full(idxTest), classes, classNames);

% Compute class weights for imbalanced loss
dropoutRate = 0.1; % Lower dropout for better learning
counts = countcats(y_train_cat);
classWeights = (1 ./ counts) ./ sum(1./counts) * numel(classes);

% Class distribution statistics
trainCounts = countcats(y_train_cat);
valCounts   = countcats(y_val_cat);
testCounts  = countcats(y_test_cat);

trainPercent = 100 * trainCounts / sum(trainCounts);
valPercent   = 100 * valCounts   / sum(valCounts);
testPercent  = 100 * testCounts  / sum(testCounts);

fprintf("\nClass distribution (Training):\n");
for i = 1:numel(classes)
    fprintf("  Class %s: %.2f%%\n", classNames(i), trainPercent(i));
end

fprintf("\nClass distribution (Validation):\n");
for i = 1:numel(classes)
    fprintf("  Class %s: %.2f%%\n", classNames(i), valPercent(i));
end

fprintf("\nClass distribution (Test):\n");
for i = 1:numel(classes)
    fprintf("  Class %s: %.2f%%\n", classNames(i), testPercent(i));
end

% Prepare training and validation data
X_train_full = dlarray(X_full(idxTrain, :));
A_train_full = A_full(idxTrain, idxTrain);
T_train_full = onehotencode(y_train_cat, 2, 'ClassNames', string(classes));

X_val_full = dlarray(X_full(idxValidation, :));
A_val_full = A_full(idxValidation, idxValidation);
T_val_full = onehotencode(y_val_cat, 2, 'ClassNames', string(classes));

if canUseGPU
    X_val_full = gpuArray(X_val_full);
    A_val_full = gpuArray(A_val_full);
    T_val_full = gpuArray(T_val_full);
end

%% Network Initialization
seeds = [0, 1, 2];
for i = 1:numel(seeds)
    seed = seeds(i);
    rng(seeds(i));
    parameters = struct;
    numHiddenFeatureMaps = num_features * 1; % Increase hidden units
    disp(numHiddenFeatureMaps)
    validationFrequency = 1;
    fprintf('FeatureDim = %d\n', featureDim);

    % Layer weights
    parameters.mult1.Weights = dlarray(initializeGlorot([featureDim, numHiddenFeatureMaps], numHiddenFeatureMaps, featureDim,   "double"));
    disp(size(parameters.mult1.Weights))
    parameters.mult2.Weights = dlarray(initializeGlorot([numHiddenFeatureMaps, numHiddenFeatureMaps], numHiddenFeatureMaps, numHiddenFeatureMaps, "double"));
    parameters.mult3.Weights = dlarray(initializeGlorot([numHiddenFeatureMaps, numClasses],         numClasses,        numHiddenFeatureMaps, "double"));

    if canUseGPU
        X_train_full = gpuArray(X_train_full);
    end

    %% Training Setup
    numEpochs = 100;
    learnRate = 0.01; % Lower learning rate for stability
    trailingAvg   = [];
    trailingAvgSq = [];
    
    train_losses  = zeros(numEpochs,1);
    train_accs    = zeros(numEpochs,1);
    train_prec    = zeros(numEpochs,1);
    train_rec     = zeros(numEpochs,1);
    train_f1      = zeros(numEpochs,1);
    val_losses    = zeros(numEpochs,1);
    val_accs      = zeros(numEpochs,1);
    val_prec      = zeros(numEpochs,1);
    val_rec       = zeros(numEpochs,1);
    val_f1        = zeros(numEpochs,1);
    t = tic;

    %% Full‑Batch Training Loop
    for epoch = 1:numEpochs
        % Compute loss & gradients on entire training set
        [loss, gradients] = dlfeval(@modelLoss, parameters, X_train_full, A_train_full, T_train_full, classWeights, 0.5, true);


        % Update parameters
        [parameters, trailingAvg, trailingAvgSq] = ...
            adamupdate(parameters, gradients, trailingAvg, trailingAvgSq, epoch, learnRate);

        % Store training loss
        train_losses(epoch) = double(loss);

        % --- Metrics on train set ---
        Y_train = model(parameters, X_train_full, A_train_full, dropoutRate, true);
        Y_train_cls   = onehotdecode(Y_train, string(classes), 2);
        train_accs(epoch) = mean(Y_train_cls == y_train_cat);
        [p, r, f]     = calculatePrecisionRecall(Y_train_cls, y_train_cat);
        train_prec(epoch) = p(end);
        train_rec(epoch)  = r(end);
        train_f1(epoch)   = f(end);


        if mod(epoch, validationFrequency)==0
            X_val_full = dlarray(X_full(idxValidation, :));
            if canUseGPU, X_val_full = gpuArray(X_val_full); end
            Y_val = model(parameters, X_val_full, A_val_full, dropoutRate, false); % Use eval mode for validation
            Y_val_cls = onehotdecode(Y_val, string(classes), 2);
            val_accs(epoch)   = mean(Y_val_cls == y_val_cat);
            val_losses(epoch) = double(crossentropy(Y_val, T_val_full, DataFormat="BC"));
            [pv, rv, fv]      = calculatePrecisionRecall(Y_val_cls, y_val_cat);
            val_prec(epoch)   = pv(end);
            val_rec(epoch)    = rv(end);
            val_f1(epoch)     = fv(end);
        end

        fprintf('Epoch %3d/%d — Loss=%.4f | TrainAcc=%.2f%% | ValAcc=%.2f%% | Elapsed=%.1fs\n', ...
            epoch, numEpochs, train_losses(epoch), train_accs(epoch)*100, val_accs(epoch)*100, toc(t));
    end

    %% Final Test
    X_test_full = dlarray(X_full(idxTest, :));
    A_test_full = A_full(idxTest, idxTest);
    T_test_full = onehotencode(y_test_cat, 2, 'ClassNames', string(classes));
    if canUseGPU, X_test_full = gpuArray(X_test_full); end

    Y_test = model(parameters, X_test_full, A_test_full, 0.5, false);
    Y_test_cls = onehotdecode(Y_test, string(classes), 2);
    testAcc    = mean(Y_test_cls == y_test_cat);
    [pt, rt, ft] = calculatePrecisionRecall(Y_test_cls, y_test_cat);

    testPrec = pt(end);
    testRec    = rt(end);
    testF1        = ft(end);
    
    fprintf('\n=== FINAL TEST ===\nAccuracy: %.4f | Macro‑F1: %.4f\n\n', ...
            testAcc, ft(end));

    % Overall metrics
    fprintf('OVERALL METRICS:\n');
    fprintf('  Accuracy:  %.4f\n', testAcc);
    fprintf('  Precision: %.4f\n', pt(end));
    fprintf('  Recall:    %.4f\n', rt(end));
    fprintf('  F1 Score:  %.4f\n\n', ft(end));

    % Per‑class metrics
    fprintf('PER‑CLASS METRICS:\n');
    fprintf('%-10s %-10s %-10s %-10s\n', 'Class','Prec','Rec','F1');
    for j = 1:numel(classes)
        fprintf('%-10s %-10.4f %-10.4f %-10.4f\n', ...
            classNames(j), pt(j), rt(j), ft(j));
    end
    
    % Scalar test‐set loss
    test_loss = double(crossentropy(Y_test, T_test_full, DataFormat="BC"));

    plotTrainingMetrics( ...
    train_losses, val_losses, ...
    train_accs,   val_accs, ...
    train_prec, train_rec, train_f1, ...
    val_prec,   val_rec,   val_f1, ...
    validationFrequency, seed, ...
    y_test_cat(:),    Y_test_cls(:), ...
    testAcc, testPrec, testRec, testF1 );
    
    figure;
    plot(1:numEpochs, train_losses, '-o', 'DisplayName', 'Train Loss');
    hold on;
    plot(1:numEpochs, val_losses, '-o', 'DisplayName', 'Validation Loss');
    xlabel('Epoch');
    ylabel('Loss');
    legend;
    title('Training and Validation Loss');
    
   % Save the model and training logs
    save("models/karate_node_gcn_" + string(seed) + "_" + string(num_features) + ".mat", ...
     "testAcc", "parameters", ...
     "testPrec", "testRec", "testF1", ...           
     "train_losses", "val_losses", ...
     "train_accs",   "val_accs",   ...
     "train_prec",   "train_rec",   "train_f1",    ...  
     "val_prec",     "val_rec",     "val_f1");         

    fprintf('\n===== SAVED & FINAL TEST SUMMARY =====\n');
    fprintf(' Test Accuracy : %.4f\n', testAcc);
    fprintf(' Test Precision: %.4f\n', pt(end));
    fprintf(' Test Recall   : %.4f\n', rt(end));
    fprintf(' Test F1 Score : %.4f\n', ft(end));
end

function Y = model(parameters, X, A, dropoutRate, isTraining)
    ANorm = normalizeAdjacency(A);
    conv1 = ANorm * X * parameters.mult1.Weights;
    relu1 = relu(conv1);
    if isTraining
        relu1 = dropout(relu1, dropoutRate);
    end
    conv2 = ANorm * relu1 * parameters.mult2.Weights;
    relu2 = relu(conv2);
    if isTraining
        relu2 = dropout(relu2, dropoutRate);
    end
    conv3 = ANorm * relu2 * parameters.mult3.Weights;
    Y = softmax(conv3, DataFormat="BC");
    % whos Y
end


function [loss, gradients] = modelLoss(parameters, X, A, T, classWeights, dropoutRate, isTraining)
    Y = model(parameters, X, A, dropoutRate, isTraining);
    loss = crossentropy(Y, T, classWeights, DataFormat="BC", WeightsFormat="C");
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

function out = dropout(X, rate)
    if rate == 0
        out = X;
        return;
    end
    if isa(X, 'gpuArray')
        mask = gpuArray.rand(size(X)) > rate;
    else
        mask = rand(size(X)) > rate;
    end
    out = X .* mask / (1 - rate);  
end


function weights = initializeGlorot(sz, fanOut, fanIn, dataType)
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