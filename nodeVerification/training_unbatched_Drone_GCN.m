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
    
    numEpochs = 1;
    learnRate = 0.01;
    
    validationFrequency = 1;
    
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
    
    % disp(XTrain);
    t = tic;
    % Begin training (custom train loop)
    while epoch < numEpochs
        epoch = epoch + 1;
    
        % Evaluate the model loss and gradients.
        [loss,gradients] = dlfeval(@modelLoss,parameters,XTrain,ATrain,TTrain);

        disp(loss);
        disp("Loss training = " + string(loss));
        YTrain = model(parameters, XTrain, ATrain);
        YTrainClass = onehotdecode(YTrain, classes, 2);
        train_acc = mean(YTrainClass == labelsTrain);
        % train_accs(epoch) = train_acc;
        disp("Accuracy training = "+string(train_acc));
    
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg,trailingAvgSq,epoch,learnRate);

        % Get validation data
        YValidation = model(parameters,XValidation,AValidation); % output inference
        Yclass = onehotdecode(YValidation,classes,2); % convert to onehot vector
        accVal = mean(Yclass == labelsValidation); % compute accuracy over all validation data

        % update best model
        if accVal > best_val
            best_val = accVal;
            best_params = parameters;
        end
    
        % Display the validation metrics.
        if epoch == 1 || mod(epoch,validationFrequency) == 0
            lossValidation = crossentropy(YValidation,TValidation,DataFormat="BC");
            disp("Epoch = "+string(epoch));
            disp("Loss validation = "+string(lossValidation));
            disp("Accuracy validation = "+string(accVal));
            toc(t);
            disp('--------------------------------------');
        end
    
    end
    
    % save best model
    parameters = best_params;
    % Testing
    
    [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest,featureDataTest,labelsTest, 'preprocessedPredictors_test.mat');
    % XTest = (XTest - muX)./sqrt(sigsqX);
    % XTest = dlarray(XTest);
    
    YTest = model(parameters,XTest,ATest);
    YTest = onehotdecode(YTest,classes,2);
    acTest = mean(YTest == labelsTest); % compute accuracy over all validation 
    
    accuracy = mean(YTest == labelsTest);
    disp("Test accuracy = "+string(accuracy));
    
    save("models/gcn_"+string(seed)+".mat", "accuracy", "parameters", "muX", "sigsqX", "best_val");

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

function [loss,gradients] = modelLoss(parameters,X,A,T)

    Y = model(parameters,X,A);
    loss = crossentropy(Y,T,DataFormat="BC");
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

