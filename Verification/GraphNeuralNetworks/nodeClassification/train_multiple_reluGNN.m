canUseGPU = false;

% find the project root
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME environment variable is not set. Please set it to your project root directory.');
end

% load in data
dataFile = fullfile(projectRoot, 'data', 'node.mat');
data = load(dataFile);

% Extract the feature data and the label data
featureData = data.features;


labelData = double(permute(data.labels, [2 1]));
if size(labelData, 1) < size(labelData, 2)
    labelData = labelData';
end
labelData = labelData + 1;  % This converts 0,1,2 into 1,2,3

rng(2024);

numGraphs = size(labelData, 1);

% Convert data to adjacency form
adjacencyData = edges2Adjacency(data);
% grab the size of the 20k graphs
numObservations = size(adjacencyData, 3);

% split into train, test, val
[idxTrain,idxValidation,idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

% shape is 18, 18, num_graphs
adjacencyDataTrain = adjacencyData(:, :, idxTrain);
adjacencyDataValidation = adjacencyData(:, :, idxValidation);
adjacencyDataTest = adjacencyData(:, :, idxTest);

% shape is 18, 18, num_graphs
featureDataTrain = featureData(:, :, idxTrain);
featureDataValidation = featureData(:, :, idxValidation);
featureDataTest = featureData(:, :, idxTest);

% shape is num_graphs, 18
labelDataTrain = labelData(idxTrain, :);
labelDataValidation = labelData(idxValidation, :);
labelDataTest = labelData(idxTest, :);

% Convert data for training
[ATrain,XTrain,labelsTrain] = preprocessData(adjacencyDataTrain,featureDataTrain,labelDataTrain, 'preprocessedPredictors_train.mat');
[AValidation,XValidation,labelsValidation] = preprocessData(adjacencyDataValidation,featureDataValidation,labelDataValidation, 'preprocessedPredictors_val.mat');


classes = categories(labelsTrain);   % This becomes a 1×C string/categorical array
numClasses = numel(classes);           % This is the number of classes

classList = categories(labelsTrain);
counts = countcats(labelsTrain);
classWeights = 1 ./ counts;
classWeights = classWeights / sum(classWeights) * numel(classList);
classWeights = classWeights(:)';

% Normalize training data
% muX = mean(XTrain);
% sigsqX = var(XTrain,1);

% disp(XTrain);
% XTrain = (XTrain - muX) ./ sqrt(sigsqX);
% XValidation = (XValidation - muX)./sqrt(sigsqX);
% Create neural network model
% seeds = [5,6,7,8,9];  % Or use [0,1,2,3,4]
seeds = [1];

for i=1:length(seeds)

    % Set fix random seed for reproducibility
    seed = seeds(i);
    rng(seed);

    % Initialize models
    parameters = struct;

    % Layer 1
    numHiddenFeatureMaps = 110;
    numInputFeatures = size(XTrain,2);

    sz = [numInputFeatures numHiddenFeatureMaps];
    numOut = numHiddenFeatureMaps;
    numIn = numInputFeatures;
    parameters.mult1.Weights = initializeGlorot(sz,numOut,numIn,"double");

    % Layer 2
    sz = [numHiddenFeatureMaps numHiddenFeatureMaps];
    numOut = numHiddenFeatureMaps;
    numIn = numHiddenFeatureMaps;
    parameters.mult2.Weights = initializeGlorot(sz,numOut,numIn,"double");

    % Layer 3
    classes = categories(labelsTrain);
    numClasses = numel(classes);

    sz = [numHiddenFeatureMaps numClasses];
    numOut = numClasses;
    numIn = numHiddenFeatureMaps;
    parameters.mult3.Weights = initializeGlorot(sz,numOut,numIn,"double");


    % Training

    numEpochs = 100;
    learnRate = 0.01;

    validationFrequency = 10;

    % initialize params for adam
    trailingAvg = [];
    trailingAvgSq = [];

    % convert data to dlarray for training
    XTrain = dlarray(XTrain);
    XValidation = dlarray(XValidation);

    % gpu?
    if canUseGPU
        XTrain = gpuArray(XTrain);
    end

    % convert labels to onehot vector encoding
    TTrain = onehotencode(labelsTrain,2,ClassNames=classes);
    TValidation = onehotencode(labelsValidation,2,ClassNames=classes);

    epoch = 0; %initialize epoch
    best_val = 0;
    best_params = [];

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

    test_losses = zeros(numEpochs, 1);
    test_accs = zeros(numEpochs, 1);
    test_precision = zeros(numEpochs, 1);
    test_recall = zeros(numEpochs, 1);
    test_f1 = zeros(numEpochs, 1);

    % disp(XTrain);
    t = tic;
    % Begin training (custom train loop)
    while epoch < numEpochs
        epoch = epoch + 1;

        % Evaluate the model loss and gradients.
        [loss,gradients] = dlfeval(@modelLoss,parameters,XTrain,ATrain,TTrain, classWeights);

        % Store training metrics
        train_losses(epoch) = double(loss);
        YTrain = model(parameters, XTrain, ATrain);
        YTrainClass = onehotdecode(YTrain, classes, 2);
        train_acc = mean(YTrainClass == labelsTrain);
        train_accs(epoch) = train_acc;

        % Calculate precision and recall for training
        [prec, rec, f1] = calculatePrecisionRecall(YTrainClass, labelsTrain);
        train_precision(epoch) = prec(end); % Store macro average
        train_recall(epoch) = rec(end);     % Store macro average
        train_f1(epoch) = f1(end);          % Store macro average

        % Display training metrics in a more readable format
        fprintf('\n----- Epoch %d/%d -----\n', epoch, numEpochs);
        fprintf('TRAINING METRICS:\n');
        fprintf('  Loss:      %.4f\n', double(loss));
        fprintf('  Accuracy:  %.4f\n', train_acc);
        fprintf('  Precision: %.4f\n', train_precision(epoch));
        fprintf('  Recall:    %.4f\n', train_recall(epoch));
        fprintf('  F1 Score:  %.4f\n', train_f1(epoch));

        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg,trailingAvgSq,epoch,learnRate);

        % Get validation data
        YValidation = model(parameters,XValidation,AValidation); % output inference
        Yclass = onehotdecode(YValidation,classes,2); % convert to onehot vector
        accVal = mean(Yclass == labelsValidation); % compute accuracy over all validation data
        lossValidation = crossentropy(YValidation,TValidation,DataFormat="BC");

        % Store validation metrics
        val_losses(epoch) = double(lossValidation);
        val_accs(epoch) = accVal;

        % Calculate precision and recall for validation
        [prec, rec, f1] = calculatePrecisionRecall(Yclass, labelsValidation);
        val_precision(epoch) = prec(end); % Store macro average
        val_recall(epoch) = rec(end);     % Store macro average
        val_f1(epoch) = f1(end);          % Store macro average

        % update best model
        if accVal > best_val
            best_val = accVal;
            best_params = parameters;
        end

        % Get test metrics during training
        if epoch == 1 || mod(epoch,validationFrequency) == 0
            % Prepare test data
            [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest,featureDataTest,labelDataTest, 'preprocessedPredictors_test.mat');
            XTest = dlarray(XTest);
            TTest = onehotencode(labelsTest,2,ClassNames=classes);

            % Calculate test metrics
            YTest = model(parameters,XTest,ATest);
            YTestClass = onehotdecode(YTest,classes,2);
            test_acc = mean(YTestClass == labelsTest);
            test_loss = crossentropy(YTest,TTest,DataFormat="BC");

            % Store test metrics
            test_losses(epoch) = double(test_loss);
            test_accs(epoch) = test_acc;

            % Calculate precision and recall for test
            [prec, rec, f1] = calculatePrecisionRecall(YTestClass, labelsTest);
            test_precision(epoch) = prec(end); % Store macro average
            test_recall(epoch) = rec(end);     % Store macro average
            test_f1(epoch) = f1(end);          % Store macro average
        else
            % For epochs where we don't calculate test metrics, use the previous values
            if epoch > 1
                test_losses(epoch) = test_losses(epoch-1);
                test_accs(epoch) = test_accs(epoch-1);
                test_precision(epoch) = test_precision(epoch-1);
                test_recall(epoch) = test_recall(epoch-1);
                test_f1(epoch) = test_f1(epoch-1);
            end
        end

        % Display the metrics.
        if epoch == 1 || mod(epoch,validationFrequency) == 0
            % Display validation metrics
            fprintf('VALIDATION METRICS:\n');
            fprintf('  Loss:      %.4f\n', double(lossValidation));
            fprintf('  Accuracy:  %.4f\n', accVal);
            fprintf('  Precision: %.4f\n', val_precision(epoch));
            fprintf('  Recall:    %.4f\n', val_recall(epoch));
            fprintf('  F1 Score:  %.4f\n', val_f1(epoch));

            % Display test metrics
            fprintf('TEST METRICS:\n');
            fprintf('  Loss:      %.4f\n', double(test_loss));
            fprintf('  Accuracy:  %.4f\n', test_acc);
            fprintf('  Precision: %.4f\n', test_precision(epoch));
            fprintf('  Recall:    %.4f\n', test_recall(epoch));
            fprintf('  F1 Score:  %.4f\n', test_f1(epoch));

            % Display time elapsed
            fprintf('Time elapsed: %.2f seconds\n', toc(t));
            fprintf('--------------------------------------\n');
        end

    end

    % save best model
    parameters = best_params;
    % Final Testing

    [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest,featureDataTest,labelDataTest, 'preprocessedPredictors_test.mat');
    % XTest = (XTest - muX)./sqrt(sigsqX);
    YTest = model(parameters,XTest,ATest);
    YTest = onehotdecode(YTest,classes,2);
    accuracy = mean(YTest == labelsTest);

    % Calculate final precision, recall, and F1 score
    [precision, recall, f1] = calculatePrecisionRecall(YTest, labelsTest);

    % Display final metrics in a more readable format
    fprintf('\n===== FINAL TEST RESULTS =====\n');
    fprintf('OVERALL METRICS:\n');
    fprintf('  Accuracy:  %.4f\n', accuracy);
    fprintf('  Precision: %.4f (macro avg)\n', precision(end));
    fprintf('  Recall:    %.4f (macro avg)\n', recall(end));
    fprintf('  F1 Score:  %.4f (macro avg)\n', f1(end));

    % Display per-class metrics
    fprintf('\nPER-CLASS METRICS:\n');
    classNames = categories(labelsTest);
    fprintf('%-20s %-12s %-12s %-12s\n', 'Class', 'Precision', 'Recall', 'F1 Score');
    fprintf('%-20s %-12s %-12s %-12s\n', '-----', '---------', '------', '--------');
    for idx = 1:length(classNames)
        fprintf('%-20s %-12.4f %-12.4f %-12.4f\n', classNames{idx}, precision(idx), recall(idx), f1(idx));
    end
    fprintf('==============================\n');

    % Create confusion matrix with enhanced labels and titles
    figure('Position', [100, 100, 800, 600]);
    cm = confusionchart(labelsTest, YTest, ...
        'ColumnSummary', 'column-normalized', ...
        'RowSummary', 'row-normalized');

    % Add main title - confusionchart only accepts simple title without formatting
    title("GCN Confusion Matrix");

    % Add axis labels - confusionchart only accepts simple labels without formatting
    xlabel('Predicted Class');
    ylabel('True Class');

    % Add explanation text for the normalized values
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
        'String', 'Column normalized: Each cell shows what percentage of predictions for a class were correct', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');
    annotation('textbox', [0.15, 0.95, 0.7, 0.05], ...
        'String', 'Row normalized: Each cell shows what percentage of actual instances of a class were correctly predicted', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    % Note: We're not customizing the appearance further to avoid errors
    % The default confusion matrix appearance is already good

    % Save confusion matrix
    projectRoot = getenv('AV_PROJECT_HOME');
    if isempty(projectRoot)
        projectRoot = pwd;
    end
    resultsDir = fullfile(projectRoot, 'results');
    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end
    saveas(gcf, fullfile(resultsDir, 'confusion_matrix.png'));

    % Plot training, validation, and test metrics
    plotTrainingMetrics(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, ...
        train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, ...
        test_precision, test_recall, test_f1, validationFrequency);

    save("models/gcn_"+string(seed)+".mat", "accuracy", "parameters", "precision", "recall", "f1", ...
        "train_losses", "val_losses", "test_losses", "train_accs", "val_accs", "test_accs", ...
        "train_precision", "train_recall", "train_f1", "val_precision", "val_recall", "val_f1", ...
        "test_precision", "test_recall", "test_f1", "best_val");

end

function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData, cacheFileName)
    % this loads in the cached file so we don't have to wait on predictors each time
    projectRoot = getenv('AV_PROJECT_HOME');
    cacheFile = fullfile(projectRoot, 'data', cacheFileName);
    if exist(cacheFile, 'file')
        disp(['Loading cached ', cacheFileName, '...']);
        load(cacheFile, 'adjacency', 'features');
    else
        disp([cacheFileName,' not found. Running preprocessPredictors...']);
        [adjacency, features] = preprocessPredictors(adjacencyData, featureData);
        save(cacheFile, 'adjacency', 'features', '-v7.3');
    end

    labels = [];
    for i = 1:size(adjacencyData, 3)
        numNodes = find(any(adjacencyData(:,:,i)), 1, "last");
        if isempty(numNodes)
            numNodes = 0;
        end
        T = labelData(i, 1:numNodes);
        labels = [labels; T(:)];

    end


    labels2 = nonzeros(labelData);
    assert(isequal(labels2,labels2))

    labelNumbers = unique(labels);
    labelNames =  labelSymbol(labelNumbers);
    labels = categorical(labels, labelNumbers, labelNames);
    % whos adjacency
    % whos features
    % whos labels
end

function [adjacency, features] = preprocessPredictors(adjacencyData, featureData)
    adjacency = sparse([]);
    features = [];

    for i = 1:size(adjacencyData, 3)
        % Number of actual nodes
        numNodes = find(any(adjacencyData(:,:,i)), 1, "last");
        if isempty(numNodes) || numNodes==0
            continue
        end

        A = adjacencyData(1:numNodes, 1:numNodes, i);
        X = featureData(1:numNodes, :, i);

        adjacency = blkdiag(adjacency, A);

        % Concatenate feature rows
        features = [features; X];

        if mod(i, 500) == 0
            fprintf('Processing graph %d\n', i);
        end
    end
end


function sym = labelSymbol(labelNumbers)
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

function Y = model(parameters,X,A)
    ANorm = normalizeAdjacency(A);

    Z1 = X;

    Z2 = ANorm * Z1 * parameters.mult1.Weights;
    Z2 = relu(Z2) + Z1;
    % Z2 = relu(Z2);

    Z3 = ANorm * Z2 * parameters.mult2.Weights;
    Z3 = relu(Z3) + Z2;
    % Z3 = relu(Z3);

    Z4 = ANorm * Z3 * parameters.mult3.Weights;
    Y = softmax(Z4,DataFormat="BC");

end

function [loss,gradients] = modelLoss(parameters,X,A,T, classWeights)

    Y = model(parameters,X,A);
    loss = crossentropy(Y,T,classWeights,DataFormat="BC", WeightsFormat="UC");
    % loss = crossentropy(Y,T,DataFormat="BC" );

    gradients = dlgradient(loss, parameters);

end

function ANorm = normalizeAdjacency(A)

    % Add self connections to adjacency matrix.
    A = A + speye(size(A));

    % Compute inverse square root of degree.
    degree = sum(A, 2);
    degreeInvSqrt = sparse(sqrt(1./degree));

    % Normalize adjacency matrix.
    ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);

end

