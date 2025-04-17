% Cora GCN Training with Extended Metrics (Accuracy, Precision, Recall, F1)

%% Setup
canUseGPU = false;
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    error('AV_PROJECT_HOME not set.');
end

dataFile = fullfile(projectRoot, 'data', 'cora_node.mat');
data = load(dataFile);

% Full Cora graph
A_full   = data.edge_indices(:,:,1);
X_full   = data.features(:,:,1);
y_full   = double(data.labels(:)) + 1;
[numNodes, featureDim] = size(X_full);
numClasses = max(y_full);

% Train/Val/Test node splits
rng(2024);
indices = randperm(numNodes);
nTrain = round(0.8 * numNodes);
nVal   = round(0.1 * numNodes);
idxTrain      = indices(1:nTrain);
idxValidation = indices(nTrain+1 : nTrain+nVal);
idxTest       = indices(nTrain+nVal+1 : end);

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

% Preallocate metrics arrays
train_losses    = zeros(numEpochs,1);
train_accs      = zeros(numEpochs,1);
train_prec      = zeros(numEpochs,1);
train_rec       = zeros(numEpochs,1);
train_f1        = zeros(numEpochs,1);
val_losses      = zeros(numEpochs,1);
val_accs        = zeros(numEpochs,1);
val_prec        = zeros(numEpochs,1);
val_rec         = zeros(numEpochs,1);
val_f1          = zeros(numEpochs,1);

% Optimizer state for Adam
trailingAvg   = [];
trailingAvgSq = [];

%% Training loop
for epoch = 1:numEpochs
    % Shuffle training nodes
    idxTrain = idxTrain(randperm(nTrain));
    epochLoss = 0;
    for b = 1:numBatches
        batchIdx = idxTrain((b-1)*batchSize+1 : min(b*batchSize,nTrain));
        [A_batch, X_batch, y_batch] = createMiniBatch(A_full, X_full, y_full, batchIdx);
        Xb = dlarray(X_batch);
        T  = onehotencode(categorical(y_batch,1:numClasses), 2, 'ClassNames', string(1:numClasses));
        if canUseGPU, Xb = gpuArray(Xb); end
        [loss, grads] = dlfeval(@modelLoss, parameters, Xb, A_batch, T);
        [parameters, trailingAvg, trailingAvgSq] = adamupdate(parameters, grads, trailingAvg, trailingAvgSq, epoch, learnRate);
        epochLoss = epochLoss + double(loss);
    end
    train_losses(epoch) = epochLoss/numBatches;
    
    % ---- Evaluate on full graph ----
    Xall = dlarray(X_full);
    if canUseGPU, Xall = gpuArray(Xall); end
    Yall = model(parameters, Xall, A_full);  % [numNodes x numClasses]
    scores = extractdata(Yall);
    [~, preds] = max(scores,[],2);
    
    % Train metrics
    train_accs(epoch) = mean(preds(idxTrain) == y_full(idxTrain));
    [p, r, f] = calcPRF(preds(idxTrain), y_full(idxTrain), numClasses);
    train_prec(epoch) = p; train_rec(epoch) = r; train_f1(epoch) = f;
    
    % Validation metrics
    [~, p_val] = max(scores(idxValidation,:),[],2);
    val_losses(epoch) = mean(crossentropy(dlarray(scores(idxValidation,:),'BC'), onehotencode(categorical(y_full(idxValidation),1:numClasses),2,'ClassNames',string(1:numClasses))));
    val_accs(epoch) = mean(p_val == y_full(idxValidation));
    [p, r, f] = calcPRF(p_val, y_full(idxValidation), numClasses);
    val_prec(epoch) = p; val_rec(epoch) = r; val_f1(epoch) = f;
    
    fprintf('Epoch %3d/%d | Train L=%.3f A=%.3f P=%.3f R=%.3f F1=%.3f | Val L=%.3f A=%.3f P=%.3f R=%.3f F1=%.3f\n', ...
        epoch, numEpochs, train_losses(epoch), train_accs(epoch), train_prec(epoch), train_rec(epoch), train_f1(epoch), ...
        val_losses(epoch), val_accs(epoch), val_prec(epoch), val_rec(epoch), val_f1(epoch));
end

%% Final Test Metrics
scores = extractdata(model(parameters, dlarray(X_full), A_full));
[~, p_test] = max(scores(idxTest,:),[],2);
[test_p, test_r, test_f] = calcPRF(p_test, y_full(idxTest), numClasses);
test_acc = mean(p_test == y_full(idxTest));
fprintf('Test Results | Acc=%.3f P=%.3f R=%.3f F1=%.3f\n', test_acc, test_p, test_r, test_f);

%% Helper Functions
function [A_batch, X_batch, y_batch] = createMiniBatch(A_full, X_full, y_full, idx)
    A_batch = A_full(idx,idx);
    X_batch = X_full(idx,:);
    y_batch = y_full(idx);
end

function Y = model(params, X, A)
    AN = normalizeAdjacency(A);
    h1 = relu(AN*X*params.mult1.Weights);
    h2 = relu(AN*h1*params.mult2.Weights);
    logits = AN*h2*params.mult3.Weights;
    Y = softmax(logits,'DataFormat','BC');
end

function [loss, grads] = modelLoss(params, X, A, T)
    Y = model(params, X, A);
    loss = crossentropy(Y, T);
    grads = dlgradient(loss, params);
end

function AN = normalizeAdjacency(A)
    if isa(A,'single'), A = double(A); end
    A = A + speye(size(A)); d = sum(A,2);
    D = spdiags(d.^(-0.5),0,size(A,1),size(A,1));
    AN = D*A*D;
end

function W = initializeGlorot(sz, fanOut, fanIn)
    stdv = sqrt(2/(fanIn+fanOut)); W = stdv*randn(sz,'double');
end

function [P, R, F1] = calcPRF(preds, truths, C)
    % Macro-averaged precision, recall, F1
    prec = zeros(C,1); rec = zeros(C,1);
    for c=1:C
        tp = sum(preds==c & truths==c);
        fp = sum(preds==c & truths~=c);
        fn = sum(preds~=c & truths==c);
        prec(c) = tp/(tp+fp+eps);
        rec(c)  = tp/(tp+fn+eps);
    end
    P = mean(prec); R = mean(rec);
    F1 = 2*(P*R)/(P+R+eps);
end
