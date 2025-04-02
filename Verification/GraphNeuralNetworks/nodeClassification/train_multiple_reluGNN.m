%% Train multiple models to evaluate and certify accuracy and certified accuracy
% For now, do 5 different random seeds and save the model to analyze later

%% Download data and preprocess it


dataFile = "C:/Users/Noah/OneDrive - Vanderbilt/Spring 2025/CS 6315/Project/AV_Project/Data/node.mat";



data = load(dataFile);
disp(data);
% Extract the feature data and the label numbers from the loaded structure. 
% Permute the feature data so that the third dimension corresponds to the observations. 
featureData = data.features;
labelData = data.labels;


% Sort the label numbers in descending order.

% convert data to adjacency form
adjacencyData = edges2Adjacency(data);

% Check adjacencyData dimensions
disp("Size of adjacencyData: " + mat2str(size(adjacencyData)));

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


% convert data for training
[ATrain,XTrain,labelsTrain] = preprocessData(adjacencyDataTrain,featureDataTrain,labelDataTrain);
[AValidation,XValidation,labelsValidation] = preprocessData(adjacencyDataValidation,featureDataValidation,labelDataValidation);

% normalize training data
muX = mean(XTrain);
sigsqX = var(XTrain,1);

if isempty(XTrain)
    error("Feature matrix XTrain is empty.");
end

XTrain = (XTrain - muX) ./ sqrt(sigsqX);
XValidation = (XValidation - muX)./sqrt(sigsqX);


%% Create neural network model

% seeds = [0,1,2,3,4];
seeds = [5,6,7,8,9];

for i=1:length(seeds)
    
    % Set fix random seed for reproducibility
    seed = seeds(i);
    rng(seed);

    % Initialize models
    parameters = struct;
    
    % Layer 1
    numHiddenFeatureMaps = 32;
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
    
    
    %% Training
    
    numEpochs = 1500;
    learnRate = 0.01;
    
    validationFrequency = 100;
    
    % initialize params for adam
    trailingAvg = [];
    trailingAvgSq = [];
    
    % convert data to dlarray for training
    XTrain = dlarray(XTrain);
    XValidation = dlarray(XValidation);
    
    canUseGPU = false;
    % gpu?
    if canUseGPU
        XTrain = gpuArray(XTrain);
    end
    
    % Convert labels to onehot vector encoding
    TTrain = onehotencode(labelsTrain, 1, ClassNames=classes);
    TValidation = onehotencode(labelsValidation, 1, ClassNames=classes);    
    
    epoch = 0; %initialize epoch
    best_val = 0;
    best_params = [];
    
    t = tic;
    % Begin training (custom train loop)
    while epoch < numEpochs
        epoch = epoch + 1;
    
        % Evaluate the model loss and gradients.
        [loss,gradients] = dlfeval(@modelLoss,parameters,XTrain,ATrain,TTrain);
    
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
    
    
    %% Testing
    
    [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest,featureDataTest,labelDataTest);
    XTest = (XTest - muX)./sqrt(sigsqX);
    XTest = dlarray(XTest);
    
    YTest = model(parameters,XTest,ATest);
    YTest = onehotdecode(YTest,classes,2);
    
    accuracy = mean(YTest == labelsTest);
    disp("Test accuracy = "+string(accuracy));
    
    % Visualize test results
    figure
    cm = confusionchart(labelsTest,YTest, ...
        ColumnSummary="column-normalized", ...
        RowSummary="row-normalized");
    title("GCN QM7 Confusion Chart");
    
    % Save model
    save("models/gcn_"+string(seed)+".mat", "accuracy", "parameters", "muX", "sigsqX", "best_val");

end



%% Helper functions %%
%%%%%%%%%%%%%%%%%%%%%%

function [adjacency,features,labels] = preprocessData(adjacencyData,featureData, labelData)
    [adjacency, features] = preprocessPredictors(adjacencyData,featureData);

    labels = [];
    
    % Convert labels to categorical.
    for i = 1:size(adjacencyData,2)
        % Extract the cell content first using curly braces, then get nonzeros
        labelContent = labelData{1,i};
        
        % Check if the content is a cell itself (nested cell array)
        if iscell(labelContent)
            if ~isempty(labelContent)
                labelContent = labelContent{1}; % Extract from nested cell
            else
                continue; % Skip empty cells
            end
        end
        
        % Now extract non-zero values if it's numeric
        if isnumeric(labelContent)
            T = nonzeros(labelContent);
        else
            % If it's not numeric, convert or handle appropriately
            T = labelContent;
        end

        labels = [labels; T];
    end
    
    labelNumbers = unique(labels);
    labelNames = labelSymbol(labelNumbers);
    labels = categorical(labels, labelNumbers, labelNames);
end

function [adjacency,features] = preprocessPredictors(adjacencyData,featureData)
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
    % Normalize adjacency matrix
    A_norm = normalizeAdjacency(A);
    
    % Forward pass
    X1 = relu(A_norm * X * parameters.mult1.Weights);
    X2 = relu(A_norm * X1 * parameters.mult2.Weights);
    X3 = A_norm * X2 * parameters.mult3.Weights;
    
    % Use softmax with the correct syntax
    Y = softmax(X3, 'DataFormat', 'BC');
end

function [loss,gradients] = modelLoss(parameters,X,A,T)

    Y = model(parameters,X,A);
    loss = crossentropy(Y,T,DataFormat="BC");
    gradients = dlgradient(loss, parameters);

end

function ANorm = normalizeAdjacency(A)
    if isempty(A)
        ANorm = sparse([]);
        return;
    end
    
    % Add self-connections to adjacency matrix.
    A = A + speye(size(A));
    
    % Compute inverse square root of degree.
    degree = sum(A, 2);
    degreeInvSqrt = sparse(sqrt(1 ./ degree));
    degreeInvSqrt(isinf(degreeInvSqrt)) = 0; % Handle divide-by-zero cases
    
    % Normalize adjacency matrix.
    ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);
end

