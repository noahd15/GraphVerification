% Simplified GNN Training Script

% GPU Check
useGPU = gpuDeviceCount > 0;
if useGPU
    gpuDevice(1);
    fprintf('GPU detected: Using GPU.\n');
else
    fprintf('No GPU detected: Running on CPU.\n');
end

% Load & Split Data
data = load('converted_dataset.mat');
n = numel(data.edge_indices);
idx = randperm(n);
split = round(0.7 * n);
trainIdx = idx(1:split); testIdx = idx(split+1:end);

train = structfun(@(x) x(trainIdx), data, 'UniformOutput', false);
test  = structfun(@(x) x(testIdx),  data, 'UniformOutput', false);

% Move to GPU if available
if useGPU
    fields = {'features','edge_indices','labels'};
    for f = fields
        train.(f{1}) = cellfun(@gpuArray, train.(f{1}), 'UniformOutput', false);
        test.(f{1})  = cellfun(@gpuArray, test.(f{1}),  'UniformOutput', false);
    end
end

% Weight initializer
init = @(dims,in,out) dlarray(sqrt(6/(in+out))*(2*rand(dims,'single')-1));
inputDim = size(train.features{1},2);
hiddenDim = 32;
params = struct(...
    'W1', init([inputDim,hiddenDim],inputDim,hiddenDim), 'b1', zeros(1,hiddenDim,'single'), ...
    'W2', init([hiddenDim,hiddenDim],hiddenDim,hiddenDim),       'b2', zeros(1,hiddenDim,'single'), ...
    'W3', init([hiddenDim,hiddenDim],hiddenDim,hiddenDim),       'b3', zeros(1,hiddenDim,'single'), ...
    'Wlin', init([hiddenDim,1],hiddenDim,1),                     'blin', 0);

if useGPU
    fn = fieldnames(params);
    for i = 1:numel(fn), params.(fn{i}) = gpuArray(params.(fn{i})); end
end

% Training parameters
epochs = 4; lr = 2e-5; dropoutProb = 0.2;
results = struct('trainLoss',zeros(epochs,1),'testLoss',zeros(epochs,1),'testAcc',zeros(epochs,1));

for epoch = 1:epochs
    [params, results.trainLoss(epoch)] = trainEpoch(train, params, lr, dropoutProb);
    [results.testLoss(epoch), results.testAcc(epoch)] = evaluate(test, params);
    fprintf('Epoch %d | Train Loss: %.4f | Test Loss: %.4f | Test Acc: %.4f\n', ...
        epoch, results.trainLoss(epoch), results.testLoss(epoch), results.testAcc(epoch));
end

% Save Model & Plot
save('trained_model.mat','params');
figure; plot(1:epochs, results.trainLoss, '-o', 1:epochs, results.testLoss, '-x'); xlabel('Epoch'); ylabel('Loss'); legend('Train','Test'); saveas(gcf,'loss.png');
figure; plot(1:epochs, results.testAcc, '-o'); xlabel('Epoch'); ylabel('Accuracy'); saveas(gcf,'accuracy.png');

function [params, avgLoss] = trainEpoch(data, params, lr, dropoutProb)
    N = numel(data.features);
    totalLoss = 0;
    for i = 1:N
        X = data.features{i}; A = data.edge_indices{i}; y = data.labels{i};
        [out, X1, X2, X3d, m1, m2, m3, A_norm, X3p] = ...
            forward(X,A,params.W1,params.b1,params.W2,params.b2,params.W3,params.b3,params.Wlin,params.blin,dropoutProb);
        loss = crossEntropyLoss(out,y);
        [dW1,db1,dW2,db2,dW3,db3,dWlin,dblin] = ...
            backward(X,A,y,params.W1,params.b1,params.W2,params.b2,params.W3,params.b3,params.Wlin,params.blin, ...
                     X1,X2,X3d,m1,m2,m3,A_norm,X3p);
        % Gradient descent update
        params.W1   = params.W1   - lr*dW1;
        params.b1   = params.b1   - lr*db1;
        params.W2   = params.W2   - lr*dW2;
        params.b2   = params.b2   - lr*db2;
        params.W3   = params.W3   - lr*dW3;
        params.b3   = params.b3   - lr*db3;
        params.Wlin = params.Wlin - lr*dWlin;
        params.blin = params.blin - lr*dblin;
        totalLoss = totalLoss + loss;
    end
    avgLoss = totalLoss / N;
end

function [avgLoss, accuracy] = evaluate(data, params)
    N = numel(data.features);
    totalLoss = 0;
    correct   = 0;
    for i = 1:N
        X = data.features{i}; A = data.edge_indices{i}; y = data.labels{i};
        [out, ~,~,~,~,~,~,~,~] = forward(X,A,params.W1,params.b1,params.W2,params.b2,params.W3,params.b3,params.Wlin,params.blin,0);
        out = gather(out); y = gather(y);
        pred = out > 0.5;
        correct = correct + (pred == double(y));
        totalLoss = totalLoss + crossEntropyLoss(out,y);
    end
    avgLoss = totalLoss / N;
    accuracy = correct / N;
end
function [output, X1, X2, X3_drop, mask1, mask2, mask3, A_norm, X3_pool] = ...
    forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropoutProb)

    A_norm = computeA_norm(A);

    X1 = relu(graphConv_withA_norm(X, A_norm, W1, b1));
    [X1, mask1] = dropout(X1, dropoutProb);

    X2 = relu(graphConv_withA_norm(X1, A_norm, W2, b2));
    [X2, mask2] = dropout(X2, dropoutProb);

    X3 = graphConv_withA_norm(X2, A_norm, W3, b3);
    [X3_drop, mask3] = dropout(X3, dropoutProb);

    X3_pool = mean(X3_drop, 1);
    output  = X3_pool * Wlin + blin;
end

function loss = crossEntropyLoss(predictions, targets)
    % Ensure CPU types for math
    predictions = gather(double(predictions));
    targets     = gather(double(targets));

    % Sigmoid → probability
    preds = 1./(1 + exp(-predictions));

    % Clip for numerical stability
    eps = 1e-10;
    preds = max(min(preds, 1-eps), eps);

    % Binary cross‑entropy (mean over nodes)
    loss = -mean(targets .* log(preds) + (1-targets) .* log(1-preds));
end


function A_norm = computeA_norm(A)
    eps_val = 1e-10;
    if isempty(A)
        A_norm = [];
    else
        D = diag(1./sqrt(max(sum(A,2), eps_val)));
        A_norm = D * A * D;
    end
end

function X_out = graphConv_withA_norm(X, A_norm, W, b)
    if isempty(A_norm)
        X_out = X * W + b;
    else
        X_out = A_norm * X * W + b;
    end
end

function Y = relu(X)
    Y = max(0, X);
end

function [X_drop, mask] = dropout(X, p)
    mask = (rand(size(X)) > p);
    X_drop = X .* mask;
end

function [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = backward( ...
    X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, ...
    X1, X2, X3_drop, mask1, mask2, mask3, A_norm, X3_pool)

    % Final output and sigmoid
    bounded = max(min(X3_pool * Wlin + blin, 15), -15);
    output  = 1 ./ (1 + exp(-bounded));
    dOutput = output - double(gather(y));

    % Gradients for linear layer
    dWlin = X3_pool' * dOutput;
    dblin = sum(dOutput,1);

    % Backprop through pooling
    n = size(X3_drop,1);
    dX3_pool = repmat(dOutput * Wlin', n, 1) / n;

    % Dropout backward (layer3)
    dX3 = dX3_pool .* mask3;

    % Layer‑3 graph conv
    if isempty(A_norm)
        dW3 = (X2 .* mask2)' * dX3;
        dX2tmp = dX3 * W3';
    else
        dW3 = (A_norm * (X2 .* mask2))' * dX3;
        dX2tmp = (A_norm' * dX3) * W3';
    end
    db3 = sum(dX3,1);

    % Layer‑2 ReLU + dropout backward
    dX2tmp = dX2tmp .* (X2 > 0);
    dX2 = dX2tmp .* mask2;
    if isempty(A_norm)
        dW2 = (X1 .* mask1)' * dX2;
        dX1tmp = dX2 * W2';
    else
        dW2 = (A_norm * (X1 .* mask1))' * dX2;
        dX1tmp = (A_norm' * dX2) * W2';
    end
    db2 = sum(dX2,1);

    % Layer‑1 ReLU + dropout backward
    dX1tmp = dX1tmp .* (X1 > 0);
    dX1 = dX1tmp .* mask1;
    if isempty(A_norm)
        dW1 = X' * dX1;
    else
        dW1 = (A_norm * X)' * dX1;
    end
    db1 = sum(dX1,1);
end
