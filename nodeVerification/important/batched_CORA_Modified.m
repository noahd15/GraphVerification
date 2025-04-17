% Cora GCN Training and Verification Demo
% Updated to use full graph adjacency and node‑level splits

%% Setup
canUseGPU = false;
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME not set.');
end

dataFile = fullfile(projectRoot, 'data', 'cora_node.mat');
data = load(dataFile);

% Full Cora graph
A_full   = data.edge_indices(:,:,1);    % 2708×2708 sparse (possibly single)
X_full   = data.features(:,:,1);        % 2708×featureDim
y_full   = double(data.labels(:)) + 1;   % 2708×1, labels 1–7
[numNodes, featureDim] = size(X_full);

% Train/Val/Test node splits
rng(2024);
indices = randperm(numNodes);
nTrain = round(0.8 * numNodes);
nVal   = round(0.1 * numNodes);
idxTrain      = indices(1:nTrain);
idxValidation = indices(nTrain+1 : nTrain+nVal);
idxTest       = indices(nTrain+nVal+1 : end);

numClasses = max(y_full);

%% Initialize network parameters
rng(1);
parameters = struct;
numHidden = 32;
parameters.mult1.Weights = dlarray(initializeGlorot([featureDim, numHidden], numHidden, featureDim));
parameters.mult2.Weights = dlarray(initializeGlorot([numHidden, numHidden], numHidden, numHidden));
parameters.mult3.Weights = dlarray(initializeGlorot([numHidden, numClasses], numClasses, numHidden));

%% Training settings
numEpochs = 200;
learnRate = 1e-3;
batchSize = 1024;
numBatches = ceil(nTrain / batchSize);

% Preallocate metrics
train_losses = zeros(numEpochs,1);
val_losses   = zeros(numEpochs,1);
train_accs   = zeros(numEpochs,1);
val_accs     = zeros(numEpochs,1);

% Optimizer state
trailingAvg   = [];
trailingAvgSq = [];

%% Training loop
for epoch = 1:numEpochs
    idxTrain = idxTrain(randperm(nTrain));  % shuffle
    epochLoss = 0;
    for b = 1:numBatches
        batchIdx = idxTrain((b-1)*batchSize + 1 : min(b*batchSize,nTrain));
        [A_batch, X_batch, y_batch] = createMiniBatch(A_full, X_full, y_full, batchIdx);
        Xb = dlarray(X_batch);
        T  = onehotencode(categorical(y_batch,1:numClasses), 2, 'ClassNames', string(1:numClasses));
        if canUseGPU, Xb = gpuArray(Xb); end
        [loss, grads] = dlfeval(@modelLoss, parameters, Xb, A_batch, T);
        [parameters, trailingAvg, trailingAvgSq] = adamupdate(parameters, grads, trailingAvg, trailingAvgSq, epoch, learnRate);
        epochLoss = epochLoss + double(loss);
    end
    train_losses(epoch) = epochLoss / numBatches;
    
    % --- Evaluate Full Graph ---
    Xall = dlarray(X_full);
    if canUseGPU, Xall = gpuArray(Xall); end
    Yall = model(parameters, Xall, A_full);    % 2708×numClasses
    predsAll = extractdata(Yall);
    [~, preds] = max(predsAll,[],2);

    % Train accuracy
    train_accs(epoch) = mean(preds(idxTrain) == y_full(idxTrain));

    % Validation loss & accuracy
    valIdx = idxValidation;
    % Slice numeric predictions and wrap as dlarray with 'BC'
    Yval_data = predsAll(valIdx,:);
    Yval_dl   = dlarray(Yval_data, 'BC');
    % Build one-hot targets for validation
    Tval = onehotencode(categorical(y_full(valIdx),1:numClasses), 2, 'ClassNames', string(1:numClasses));
    Tval_dl   = dlarray(Tval, 'BC');
    val_losses(epoch) = double(crossentropy(Yval_dl, Tval_dl));
    [~, pval] = max(Yval_data,[],2);
    val_accs(epoch) = mean(pval == y_full(valIdx));
    
    fprintf("Epoch %d/%d — train loss=%.4f acc=%.4f | val loss=%.4f acc=%.4f, \n", ...
        epoch, numEpochs, train_losses(epoch), train_accs(epoch), val_losses(epoch), val_accs(epoch));
end

%% Final Test
Yall = model(parameters, dlarray(X_full), A_full);
[~, p_test] = max(extractdata(Yall(idxTest,:)),[],2);
testAcc = mean(p_test == y_full(idxTest));
fprintf("Test Accuracy: %.4f\n", testAcc);

%% Helper Functions
function [A_batch, X_batch, labels_batch] = createMiniBatch(A_full, X_full, y_full, batchIndices)
    A_batch      = A_full(batchIndices, batchIndices);
    X_batch      = X_full(batchIndices, :);
    labels_batch = y_full(batchIndices);
end

function Y = model(params, X, A)
    AN = normalizeAdjacency(A);
    h1 = AN * X * params.mult1.Weights;
    h1 = relu(h1);
    h2 = AN * h1 * params.mult2.Weights;
    h2 = relu(h2);
    logits = AN * h2 * params.mult3.Weights;
    Y = softmax(logits, 'DataFormat', 'BC');
end

function [loss, gradients] = modelLoss(params, X, A, T)
    Y = model(params, X, A);
    loss = crossentropy(Y, T, 'DataFormat', 'BC');
    gradients = dlgradient(loss, params);
end

function AN = normalizeAdjacency(A)
    % Ensure numeric type consistency (convert sparse single to double)
    if isa(A, 'single')
        A = double(A);
    end
    % Add self-loops
    A = A + speye(size(A));
    % Compute symmetric normalization
    d = sum(A,2);
    D = spdiags(d.^(-0.5), 0, size(A,1), size(A,1));
    AN = D * A * D;
end

function W = initializeGlorot(sz, fanOut, fanIn)
    stddev = sqrt(2/(fanIn+fanOut));
    W = stddev * randn(sz, 'double');
end
