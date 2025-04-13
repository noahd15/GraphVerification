%% Train multiple models to evaluate and certify accuracy and certified accuracy
% For now, do 5 different random seeds and save the model to analyze later

%% GPU Setup Check
if gpuDeviceCount > 0
    g = gpuDevice(1);
    fprintf('GPU detected: %s. Using GPU features.\n', g.Name);
    canUseGPU = true;
else
    fprintf('No GPU detected. Running on CPU.\n');
    canUseGPU = false;
end

% For debugging/verification purposes, force CPU usage
canUseGPU = false;

projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME environment variable is not set. Please set it to your project root directory.');
end
dataFile = fullfile(projectRoot, 'data', 'node.mat');
data = load(dataFile);

data = load(dataFile);
disp(data);

% Extract the feature data and the label data
featureData = data.features;
labelData = data.labels;
% Ensure every label cell is a column vector
for i = 1:length(labelData)
    if isrow(labelData{i})
        labelData{i} = labelData{i}';
    end
end

% Convert data to adjacency form
adjacencyData = edges2Adjacency(data);
% Print first adjacency matrix and features
% disp("First adjacency matrix:");
% disp(adjacencyData{1});
% disp("First feature matrix:");
% disp(featureData{1});

% Check adjacencyData dimensions
% disp("Size of adjacencyData: " + mat2str(size(adjacencyData)));

% Partition data
numObservations = size(adjacencyData, 2);
[idxTrain,idxValidation,idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);

% Partition adjacency data - CELL ARRAYS ARE 1xN, so index along second dimension
adjacencyDataTrain = adjacencyData(1,idxTrain);
adjacencyDataValidation = adjacencyData(1,idxValidation);
adjacencyDataTest = adjacencyData(1,idxTest);

% Do the same for feature data - also a 1xN cell array
featureDataTrain = featureData(1,idxTrain);
featureDataValidation = featureData(1,idxValidation);
featureDataTest = featureData(1,idxTest);

% And for label data - also a 1xN cell array
labelDataTrain = labelData(1,idxTrain);
labelDataValidation = labelData(1,idxValidation);
labelDataTest = labelData(1,idxTest);

% Convert data for training
[ATrain,XTrain,labelsTrain] = preprocessData(adjacencyDataTrain,featureDataTrain,labelDataTrain);
[AValidation,XValidation,labelsValidation] = preprocessData(adjacencyDataValidation,featureDataValidation,labelDataValidation);

% Print first 10 adjacency matrices and features
% disp("First 10 adjacency matrices:");
% disp(ATrain(1:10));

% Normalize training data
muX = mean(XTrain);
sigsqX = var(XTrain,1);

if isempty(XTrain)
    error("Feature matrix XTrain is empty.");
end

XTrain = (XTrain - muX) ./ sqrt(sigsqX);
XValidation = (XValidation - muX)./sqrt(sigsqX);

%% Calculate Class Counts from Training Labels (for class weights)
classes = categories(labelsTrain);
numClasses = numel(classes);
% Get class counts
train_class_counts = zeros(1, numClasses);
for i = 1:numClasses
    train_class_counts(i) = sum(labelsTrain == classes{i});
end

train_total_nodes = numel(labelsTrain);
fprintf('Class distribution (train): ');
fprintf('%d ', train_class_counts);
fprintf('\n');

% Calculate Class Weights Inversely Proportional to Class Frequencies
class_weights = zeros(1, numClasses);
for class_idx = 1:numClasses
    class_count = train_class_counts(class_idx);
    if class_count > 0
        class_weights(class_idx) = train_total_nodes / (numClasses * class_count);
    else
        class_weights(class_idx) = 1.0;
    end
end
class_weights = class_weights / sum(class_weights) * numClasses;
fprintf('Class weights: ');
fprintf('%g ', class_weights);
fprintf('\n');

%% Create neural network model
% seeds = [5,6,7,8,9];  % Or use [0,1,2,3,4]
seeds = [1];
for i=1:length(seeds)
    % Set fix random seed for reproducibility
    seed = seeds(i);
    rng(seed);

    % Initialize models
    parameters = struct;
    
    % Layer 1
    hidden_size = 32;
    numInputFeatures = size(XTrain,2);
    
    parameters.mult1.Weights = initializeGlorot([numInputFeatures, hidden_size], ...
        hidden_size, numInputFeatures, "double");
    parameters.bias1 = dlarray(zeros(1, hidden_size, "double"));
    
    % Layer 2
    parameters.mult2.Weights = initializeGlorot([hidden_size, hidden_size], ...
        hidden_size, hidden_size, "double");
    parameters.bias2 = dlarray(zeros(1, hidden_size, "double"));
    
    % Layer 3
    parameters.mult3.Weights = initializeGlorot([hidden_size, numClasses], ...
        numClasses, hidden_size, "double");
    parameters.bias3 = dlarray(zeros(1, numClasses, "double"));
    
    %% Training
    numEpochs = 200;
    learnRate = 0.01;
    validationFrequency = 100;
    
    % Initialize params for adam
    trailingAvg = [];
    trailingAvgSq = [];
    
    % Convert data to dlarray for training
    XTrain = dlarray(XTrain);
    XValidation = dlarray(XValidation);
    
    % Use GPU if available
    if canUseGPU
        XTrain = gpuArray(XTrain);
        XValidation = gpuArray(XValidation);
        
        % Move parameters to GPU
        parameters.mult1.Weights = gpuArray(parameters.mult1.Weights);
        parameters.bias1 = gpuArray(parameters.bias1);
        parameters.mult2.Weights = gpuArray(parameters.mult2.Weights);
        parameters.bias2 = gpuArray(parameters.bias2);
        parameters.mult3.Weights = gpuArray(parameters.mult3.Weights);
        parameters.bias3 = gpuArray(parameters.bias3);
    end
    
    % Convert labels to onehot vector encoding
    TTrain = onehotencode(labelsTrain, 2, ClassNames=classes);
    TValidation = onehotencode(labelsValidation, 2, ClassNames=classes);    
    
    epoch = 0; %initialize epoch
    best_val = 0;
    best_params = [];
    
    % Create figures for real-time plotting
    figure(1);
    loss_plot = plot(NaN, NaN, '-o', 'LineWidth', 1.5); hold on;
    val_loss_plot = plot(NaN, NaN, '-x', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Loss');
    title('Training and Validation Loss');
    legend('Train Loss','Validation Loss','Location','best');
    grid on;

    figure(2);
    acc_plot = plot(NaN, NaN, '-o', 'LineWidth', 1.5); hold on;
    val_acc_plot = plot(NaN, NaN, '-x', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Accuracy');
    title('Training and Validation Accuracy');
    legend('Train Accuracy','Validation Accuracy','Location','best');
    grid on;
    
    % Training arrays for plotting
    train_losses = zeros(numEpochs, 1);
    val_losses = zeros(numEpochs, 1);
    train_accs = zeros(numEpochs, 1);
    val_accs = zeros(numEpochs, 1);
    
    t = tic;
    % Begin training (custom train loop)
    while epoch < numEpochs
        epoch = epoch + 1;
    
        % Evaluate the model loss and gradients.
        [loss,gradients] = dlfeval(@modelLoss, parameters, XTrain, ATrain, TTrain);
        train_losses(epoch) = loss;
        
        % Calculate training accuracy
        YTrain = model(parameters, XTrain, ATrain);
        YTrainClass = onehotdecode(YTrain, classes, 2);
        train_acc = mean(YTrainClass == labelsTrain);
        train_accs(epoch) = train_acc;
    
        % Update the network parameters using the Adam optimizer.
        [parameters, trailingAvg, trailingAvgSq] = adamupdate(parameters, gradients, ...
            trailingAvg, trailingAvgSq, epoch, learnRate);

        % Get validation data
        YValidation = model(parameters, XValidation, AValidation); % output inference
        Yclass = onehotdecode(YValidation, classes, 2); % convert to classes
        accVal = mean(Yclass == labelsValidation); % compute accuracy over all validation data
        val_accs(epoch) = accVal;
        
        % Calculate validation loss
        lossValidation = crossentropy(YValidation, TValidation, DataFormat="BC");
        val_losses(epoch) = lossValidation;

        % Update best model
        if accVal > best_val
            best_val = accVal;
            best_params = parameters;
        end
    
        % Display the validation metrics periodically
        if epoch == 1 || mod(epoch, validationFrequency) == 0
            disp("Epoch = " + string(epoch));
            disp("Train loss = " + string(loss) + " | Train acc = " + string(train_acc));
            disp("Valid loss = " + string(lossValidation) + " | Valid acc = " + string(accVal));
            toc(t);
            disp('--------------------------------------');
        end
    end
    
    % Save best model
    parameters = best_params;
    
    %% Testing
    [ATest, XTest, labelsTest] = preprocessData(adjacencyDataTest, featureDataTest, labelDataTest);
    XTest = (XTest - muX)./sqrt(sigsqX);
    XTest = dlarray(XTest);
    
    if canUseGPU
        XTest = gpuArray(XTest);
    end
    
    YTest = model(parameters, XTest, ATest);
    YTest = onehotdecode(YTest, classes, 2);
    
    accuracy = mean(YTest == labelsTest);
    disp("Test accuracy = " + string(accuracy));
    
    % Visualize test results
    figure(3)
    cm = confusionchart(labelsTest, YTest, ...
        ColumnSummary="column-normalized", ...
        RowSummary="row-normalized");
    title("GCN Node Classification Confusion Chart (seed=" + string(seed) + ")");
    
    % Final plots
    figure(4);
    subplot(1,2,1);
    plot(1:epoch, train_losses(1:epoch), '-o', 'LineWidth', 1.5); hold on;
    plot(1:epoch, val_losses(1:epoch), '-x', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Loss');
    title('Training and Validation Loss');
    legend('Train Loss','Validation Loss','Location','best');
    grid on;
    
    subplot(1,2,2);
    plot(1:epoch, train_accs(1:epoch), '-o', 'LineWidth', 1.5); hold on;
    plot(1:epoch, val_accs(1:epoch), '-x', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Accuracy');
    title('Training and Validation Accuracy');
    legend('Train Accuracy','Validation Accuracy','Location','best');
    grid on;
    
    % Save model
    if ~exist('models', 'dir')
        mkdir('models');
    end
    

    % Save model parameters and training results
    save("models/node_gcn_" + string(seed) + ".mat", "accuracy", "parameters", "muX", "sigsqX", "best_val", "classes");
end

%% Load and inspect saved model
data = load("models/node_gcn_1.mat");
disp(data.parameters); % Check if the structure contains mult1, mult2, mult3, etc.

%% Helper functions
function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData)
    [adjacency, features] = preprocessPredictors(adjacencyData, featureData);

    labels = [];
    nodeIndices = zeros(0,1); % Track which nodes have labels as column vector
    nodeCount = 0;
    
    % Convert labels to categorical and track node indices
    for i = 1:size(adjacencyData, 2)
        % Extract the cell content first using curly braces
        labelContent = labelData{1,i};
        
        % Extract features for this graph
        A_i = adjacencyData{1,i};
        nodesInGraph = size(A_i, 1);
        
        % Check if the content is a cell itself (nested cell array)
        if iscell(labelContent)
            if ~isempty(labelContent)
                labelContent = labelContent{1}; % Extract from nested cell
            else
                nodeCount = nodeCount + nodesInGraph;
                continue; % Skip empty cells
            end
        end
        
        % Now extract non-zero values if it's numeric
        if isnumeric(labelContent)
            % T = nonzeros(labelContent);
            T = labelContent;
            if length(T) == nodesInGraph
                % All nodes have labels - ensure column format
                newIndices = ((nodeCount+1):(nodeCount+nodesInGraph))';
                nodeIndices = [nodeIndices; newIndices];
            else
                % Only some nodes have labels - ensure column format
                newIndices = ((nodeCount+1):(nodeCount+length(T)))';
                nodeIndices = [nodeIndices; newIndices];
            end
        else
            % If it's not numeric, convert or handle appropriately
            T = labelContent;
            newIndices = ((nodeCount+1):(nodeCount+length(T)))';
            nodeIndices = [nodeIndices; newIndices];
        end

        labels = [labels; T];
        nodeCount = nodeCount + nodesInGraph;
    end
    
    % Filter features and adjacency matrix to include only labeled nodes
    if ~isempty(nodeIndices)
        features = features(nodeIndices, :);
        adjacency = adjacency(nodeIndices, nodeIndices);
    end
    
    labelNumbers = unique(labels);
    labelNames = labelSymbol(labelNumbers);
    labels = categorical(labels, labelNumbers, labelNames);
end

function [adjacency, features] = preprocessPredictors(adjacencyData, featureData)
    adjacency = sparse([]);
    features = [];

    for i = 1:size(adjacencyData, 2)
        % Extract the content from the cell arrays using curly braces
        A = adjacencyData{1,i}; 
        X = featureData{1,i};
        
        % Ensure A is in sparse format
        if ~issparse(A)
            A = sparse(A);
        end
        
        % Append extracted data
        adjacency = blkdiag(adjacency, A);
        features = [features; X];
    end
end

function Y = model(parameters, X, A)
    % Compute normalized adjacency matrix
    A_norm = computeA_norm(A);
    
    % Forward pass through GCN layers
    X1 = relu(graphConv_withA_norm(X, A_norm, parameters.mult1.Weights, parameters.bias1));
    X2 = relu(graphConv_withA_norm(X1, A_norm, parameters.mult2.Weights, parameters.bias2));
    X3 = graphConv_withA_norm(X2, A_norm, parameters.mult3.Weights, parameters.bias3);
    
    % Apply softmax for classification
    Y = softmax(X3, 'DataFormat', 'BC');
end

function [loss, gradients] = modelLoss(parameters, X, A, T)
    Y = model(parameters, X, A);
    loss = crossentropy(Y, T, DataFormat="BC");
    gradients = dlgradient(loss, parameters);
end

function A_norm = computeA_norm(A)
    if isempty(A)
        A_norm = sparse([]);
        return;
    end
    
    % Add self-loops and normalize
    A = A + speye(size(A));
    
    % Compute degree matrix and normalize
    eps_val = 1e-10;
    degree = sum(A, 2);
    degreeInvSqrt = sparse(1 ./ sqrt(max(degree, eps_val)));
    
    % D^(-1/2) * A * D^(-1/2)
    A_norm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);
end

function X_out = graphConv_withA_norm(X, A_norm, W, b)
    % Message passing using normalized adjacency
    if isempty(A_norm)
        X_out = double(X) * double(W) + repmat(b, size(X, 1), 1);
    else
        X_out = double(A_norm) * (double(X) * double(W)) + repmat(b, size(X, 1), 1);
    end
end

function Y = relu(X)
    Y = max(0, X);
end

function weights = initializeGlorot(sz, numOut, numIn, dataType)
    Z = 2 * rand(sz, dataType) - 1;
    bound = sqrt(6 / (numIn + numOut));
    weights = bound * Z;
    % Wrap in dlarray so dlgradient can trace it
    weights = dlarray(weights);
end