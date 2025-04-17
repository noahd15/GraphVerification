canUseGPU = false;

projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('Set AV_PROJECT_HOME to your project root.');
end

data = load(fullfile(projectRoot, 'data', 'cora_node.mat'));
A_full   = data.edge_indices(:,:,1);
X_full   = data.features(:,:,1);
y_full   = double(data.labels(:)) + 1;
[numNodes, featureDim] = size(X_full);

rng(2024);
indices      = randperm(numNodes);
nTrain       = round(0.8 * numNodes);
nVal         = round(0.1 * numNodes);
idxTrain     = indices(1:nTrain);
idxValidation= indices(nTrain+1 : nTrain+nVal);
idxTest      = indices(nTrain+nVal+1 : end);

classes       = unique(y_full);
numClasses    = numel(classes);
classNames    = string(1:numClasses);
y_train_cat   = categorical(y_full(idxTrain), classes, classNames);
y_val_cat     = categorical(y_full(idxValidation), classes, classNames);
y_test_cat    = categorical(y_full(idxTest), classes, classNames);

counts       = countcats(y_train_cat);
classWeights = (1 ./ counts) ./ sum(1./counts) * numel(classes);

%% Network Initialization
seeds = [1];
for i = 1:numel(seeds)
    rng(seeds(i));
    parameters = struct;
    numHiddenFeatureMaps = 32;
    validationFrequency = 1;
    fprintf('FeatureDim = %d\n', featureDim);

    % Layer weights
    parameters.mult1.Weights = dlarray(initializeGlorot([featureDim,   numHiddenFeatureMaps], numHiddenFeatureMaps, featureDim,   "double"));
    parameters.mult2.Weights = dlarray(initializeGlorot([numHiddenFeatureMaps, numHiddenFeatureMaps], numHiddenFeatureMaps, numHiddenFeatureMaps, "double"));
    parameters.mult3.Weights = dlarray(initializeGlorot([numHiddenFeatureMaps, numClasses],         numClasses,        numHiddenFeatureMaps, "double"));

    %% Prepare full‑batch data once
    X_train_full = dlarray(X_full(idxTrain, :));
    A_train_full = A_full(idxTrain, idxTrain);
    T_train_full = onehotencode(y_train_cat, 2, 'ClassNames', string(classes));
    if canUseGPU
        X_train_full = gpuArray(X_train_full);
    end

    %% Training Setup
    numEpochs = 100;
    learnRate = 0.001;
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
        [loss, gradients] = dlfeval(@modelLoss, parameters, X_train_full, A_train_full, T_train_full, classWeights);

        % Update parameters
        [parameters, trailingAvg, trailingAvgSq] = ...
            adamupdate(parameters, gradients, trailingAvg, trailingAvgSq, epoch, learnRate);

        % Store training loss
        train_losses(epoch) = double(loss);

        % --- Metrics on train set ---
        Y_train       = model(parameters, X_train_full, A_train_full);
        Y_train_cls   = onehotdecode(Y_train, string(classes), 2);
        train_accs(epoch) = mean(Y_train_cls == y_train_cat);
        [p, r, f]     = calculatePrecisionRecall(Y_train_cls, y_train_cat);
        train_prec(epoch) = p(end);
        train_rec(epoch)  = r(end);
        train_f1(epoch)   = f(end);

        % --- Metrics on validation set ---
        X_val_full = dlarray(X_full(idxValidation, :));
        A_val_full = A_full(idxValidation, idxValidation);
        T_val_full = onehotencode(y_val_cat, 2, 'ClassNames', string(classes));
        if canUseGPU, X_val_full = gpuArray(X_val_full); end

        Y_val       = model(parameters, X_val_full, A_val_full);
        Y_val_cls   = onehotdecode(Y_val, string(classes), 2);
        val_accs(epoch)   = mean(Y_val_cls == y_val_cat);
        val_losses(epoch) = double(crossentropy(Y_val, T_val_full, DataFormat="BC"));
        [pv, rv, fv]      = calculatePrecisionRecall(Y_val_cls, y_val_cat);
        val_prec(epoch)   = pv(end);
        val_rec(epoch)    = rv(end);
        val_f1(epoch)      = fv(end);

        fprintf('Epoch %3d/%d — Loss=%.4f | TrainAcc=%.2f%% | ValAcc=%.2f%% | Elapsed=%.1fs\n', ...
            epoch, numEpochs, train_losses(epoch), train_accs(epoch)*100, val_accs(epoch)*100, toc(t));
    end

    %% Final Test
    X_test_full = dlarray(X_full(idxTest, :));
    A_test_full = A_full(idxTest, idxTest);
    T_test_full = onehotencode(y_test_cat, 2, 'ClassNames', string(classes));
    if canUseGPU, X_test_full = gpuArray(X_test_full); end

    Y_test     = model(parameters, X_test_full, A_test_full);
    Y_test_cls = onehotdecode(Y_test, string(classes), 2);
    testAcc    = mean(Y_test_cls == y_test_cat);
    [pt, rt, ft] = calculatePrecisionRecall(Y_test_cls, y_test_cat);

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

    % Confusion chart (use Y_test_cls, not the old variable name)
    figure('Position',[100,100,800,600]);
    cm = confusionchart( ...
        y_test_cat, Y_test_cls, ...
        'ColumnSummary','column-normalized', ...
        'RowSummary','row-normalized' ...
    );
    title("Cora GCN Confusion Matrix");
    xlabel('Predicted Class');
    ylabel('True Class');

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

    %% Final Test (after your epoch loop)
    Y_test     = model(parameters, X_test_full, A_test_full);
    Y_test_cls = onehotdecode(Y_test, string(classes), 2);
    testAcc    = mean(Y_test_cls == y_test_cat);
    [pt, rt, ft] = calculatePrecisionRecall(Y_test_cls, y_test_cat);
    test_loss = double(crossentropy(Y_test, T_test_full, DataFormat="BC"));

    % --- Plot train/val curves, but no test lines for loss & accuracy ---
    plotTrainingMetrics( ...
        train_losses,    val_losses,    NaN,       ...  % third slot is test_loss: use NaN
        train_accs,      val_accs,      NaN,       ...  % sixth slot is test_acc:  use NaN
        train_prec,      train_rec,     train_f1,  ...  % train P/R/F1
        val_prec,        val_rec,       val_f1,    ...  %   val P/R/F1
        pt(end),         rt(end),       ft(end),   ...  %   test P/R/F1 (scalars)
        validationFrequency );
    % Plot metrics (train / val / test) – all scalars for test
    plotTrainingMetrics( ...
        train_losses,    val_losses,    test_loss,  ...   % Loss curves
        train_accs,      val_accs,      testAcc,    ...   % Accuracy curves
        train_prec,      train_rec,     train_f1,   ...   % Train P/R/F1
        val_prec,        val_rec,       val_f1,     ...   % Val   P/R/F1
        pt(end),         rt(end),       ft(end),    ...   % Test  P/R/F1
        validationFrequency );

    % Save the model and training logs
    save("models/cora_node_gcn_" + "0" + ".mat", "testAcc", "parameters", "precision", "recall", "f1", ...
        "train_losses", "val_losses", "train_accs", "val_accs", "train_precision", "train_recall", "train_f1", ...
        "val_precision", "val_recall", "val_f1");

    fprintf('\n===== SAVED & FINAL TEST SUMMARY =====\n');
    fprintf(' Test Accuracy : %.4f\n', testAcc);
    fprintf(' Test Precision: %.4f\n', pt(end));
    fprintf(' Test Recall   : %.4f\n', rt(end));
    fprintf(' Test F1 Score : %.4f\n', ft(end));
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
    % Ensure parameters are dlarray
    if ~isa(parameters.mult1.Weights, 'dlarray')
        parameters.mult1.Weights = dlarray(parameters.mult1.Weights);
        parameters.mult2.Weights = dlarray(parameters.mult2.Weights);
        parameters.mult3.Weights = dlarray(parameters.mult3.Weights);
    end

    % Forward pass
    Y = model(parameters, X, A);

    % --- FIX: Use WeightsFormat = 'C' for a 1×numClasses weight vector ---
    loss = crossentropy( ...
        Y, T, classWeights, ...
        DataFormat="BC", ...
        WeightsFormat="C" ...    % <-- was 'UC', now 'C'
    );

    % Backward pass
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
