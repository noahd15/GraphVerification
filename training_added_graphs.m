%% GPU Check
if gpuDeviceCount > 0
    g = gpuDevice(1);
    fprintf('GPU detected: %s. Using GPU features.\n', g.Name);
    useGPU = true;
else
    fprintf('No GPU detected. Running on CPU.\n');
    useGPU = false;
end

useGPU = false;  

data = load('converted_dataset.mat');
edge_indices = data.edge_indices;
features = data.features;          
labels = data.labels;             

if useGPU
    for i = 1:length(features)
        features{i} = gpuArray(features{i});
        edge_indices{i} = gpuArray(edge_indices{i});
        labels{i} = gpuArray(labels{i});
    end
end

first_features = features{1};

function weights = initializeGlorot(sz, numOut, numIn)
    Z = 2 * rand(sz, 'single') - 1;
    bound = sqrt(6 / (numIn + numOut));
    weights = bound * Z;
    weights = dlarray(weights);
end

W1 = initializeGlorot([size(first_features, 2), 64], 64, size(first_features, 2));
b1 = zeros(1, 64, 'single');
W2 = initializeGlorot([64, 64], 64, 64);
b2 = zeros(1, 64, 'single');
W3 = initializeGlorot([64, 64], 64, 64);
b3 = zeros(1, 64, 'single');
Wlin = initializeGlorot([64, 1], 1, 64);  
blin = 0;

if useGPU
    W1   = gpuArray(W1);
    b1   = gpuArray(b1);
    W2   = gpuArray(W2);
    b2   = gpuArray(b2);
    W3   = gpuArray(W3);
    b3   = gpuArray(b3);
    Wlin = gpuArray(Wlin);
    blin = gpuArray(blin);
end

num_epochs   = 20;
learning_rate = 0.001;
dropout_prob  = 0.5;

train_losses = zeros(num_epochs,1);
test_losses  = zeros(num_epochs,1);
train_accs   = zeros(num_epochs,1);
test_accs    = zeros(num_epochs,1);

for epoch = 1:num_epochs
    epoch_loss = 0;
    
    for i = 1:length(edge_indices)
        X = features{i};
        A = edge_indices{i};
        y = labels{i}; 
        
        [output, X1, X2, X3_drop, dropout_mask, A_norm, X3_pool] = ...
            forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropout_prob);
        
        loss = crossEntropyLoss(output, y);
        
        [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = ...
            backward(X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, ...
                     X1, X2, X3_drop, dropout_mask, A_norm, X3_pool);
        
        W1   = W1   - learning_rate * dW1;
        b1   = b1   - learning_rate * db1;
        W2   = W2   - learning_rate * dW2;
        b2   = b2   - learning_rate * db2;
        W3   = W3   - learning_rate * dW3;
        b3   = b3   - learning_rate * db3;
        Wlin = Wlin - learning_rate * dWlin;
        blin = blin - learning_rate * dblin;
        
        epoch_loss = epoch_loss + loss;
    end
    
    epoch_loss = epoch_loss / length(edge_indices);
    if useGPU
        epoch_loss = gather(epoch_loss);
    end
    
    [train_loss_epoch, train_acc_epoch] = computeLossAccuracy( ...
        features, edge_indices, labels, W1, b1, W2, b2, W3, b3, Wlin, blin, useGPU, 0.0);
    
    [test_loss_epoch, test_acc_epoch] = computeLossAccuracy( ...
        features, edge_indices, labels, W1, b1, W2, b2, W3, b3, Wlin, blin, useGPU, 0.0);
    
    train_losses(epoch) = train_loss_epoch;
    test_losses(epoch)  = test_loss_epoch;
    train_accs(epoch)   = train_acc_epoch;
    test_accs(epoch)    = test_acc_epoch;
    
    fprintf('Epoch %d:\n', epoch);
    fprintf('  Training Loss (this loop) : %.4f\n', epoch_loss); 
    fprintf('  Train Loss (eval)         : %.4f | Train Acc : %.4f\n', ...
        train_loss_epoch, train_acc_epoch);
    fprintf('  Test  Loss                : %.4f | Test  Acc : %.4f\n', ...
        test_loss_epoch, test_acc_epoch);
end
figure;
plot(1:num_epochs, train_losses, '-o', 'LineWidth', 1.5); hold on;
plot(1:num_epochs, test_losses, '-x', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss');
title('Training and Testing Loss');
legend('Train Loss','Test Loss','Location','best');
grid on;

figure;
plot(1:num_epochs, train_accs, '-o', 'LineWidth', 1.5); hold on;
plot(1:num_epochs, test_accs, '-x', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Accuracy');
title('Training and Testing Accuracy');
legend('Train Accuracy','Test Accuracy','Location','best');
grid on;

function A_norm = computeA_norm(A)
    eps_val = 1e-10;
    if isempty(A)
        A_norm = [];
    else
        D = diag(1 ./ sqrt(max(sum(A, 2), eps_val)));
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

function Y = sigmoid(X)
    Y = 1 ./ (1 + exp(-X));
end

function [X_drop, mask] = dropout(X, p)
    mask = (rand(size(X)) > p);
    X_drop = X .* mask;
end

function dX = dropoutGrad(dY, mask)
    dX = dY .* mask;
end

function [output, X1, X2, X3_drop, dropout_mask, A_norm, X3_pool] = forward( ...
    X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropout_prob)

    A_norm = computeA_norm(A);
    
    X1 = graphConv_withA_norm(X, A_norm, W1, b1);
    X1 = relu(X1);
    
    X2 = graphConv_withA_norm(X1, A_norm, W2, b2);
    X2 = relu(X2);
    
    X3 = graphConv_withA_norm(X2, A_norm, W3, b3);
    
    [X3_drop, dropout_mask] = dropout(X3, dropout_prob);
    
    X3_pool = mean(X3_drop, 1);
    linear_output = X3_pool * Wlin + blin;
    output = sigmoid(linear_output);
end

function loss = crossEntropyLoss(predictions, targets)
    if isa(predictions, 'dlarray')
        predictions = gather(predictions);
    end
    if isa(targets, 'dlarray')
        targets = gather(targets);
    end
    predictions = double(predictions);
    targets     = double(targets);
    
    epsilon = 1e-10;
    predictions = max(predictions, epsilon);
    predictions = min(predictions, 1 - epsilon);
    
    loss = - ( targets .* log(predictions) + (1 - targets) .* log(1 - predictions) );
end

function [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = backward( ...
    X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, ...
    X1, X2, X3_drop, dropout_mask, A_norm, X3_pool)

    linear_output = X3_pool * Wlin + blin;
    bounded       = max(min(linear_output, 15), -15);
    output        = 1 ./ (1 + exp(-bounded));
    
    if ~isa(output, 'double')
        output = double(gather(output));
    end
    if ~isa(y, 'double')
        y = double(gather(y));
    end
    
    dOutput = output - y;  
    
    dWlin = X3_pool' * dOutput;
    dblin = dOutput;
    
    n = size(X3_drop, 1);
    dX3_pool = repmat(dOutput * Wlin', n, 1) / n;
    
    dX3 = dropoutGrad(dX3_pool, dropout_mask);
    
    if isempty(A_norm)
        dX3_temp = dX3 * W3';
    else
        dX3_temp = A_norm' * dX3 * W3';
    end
    
    if isempty(A_norm)
        dW3 = X2' * dX3;
    else
        dW3 = X2' * (A_norm' * dX3);
    end
    db3 = sum(dX3, 1);
    
    dX2_from3 = dX3_temp;
    dX2 = dX2_from3 .* (X2 > 0);
    
    if isempty(A_norm)
        dW2 = X1' * dX2;
    else
        dW2 = X1' * (A_norm' * dX2);
    end
    db2 = sum(dX2, 1);
    
    if isempty(A_norm)
        dX1_from2 = dX2 * W2';
    else
        dX1_from2 = A_norm * (dX2 * W2');
    end
    
    dX1 = dX1_from2 .* (X1 > 0);
    
    if isempty(A_norm)
        dW1 = X' * dX1;
    else
        dW1 = X' * (A_norm' * dX1);
    end
    db1 = sum(dX1, 1);
end
function [avg_loss, accuracy] = computeLossAccuracy( ...
    features, edge_indices, labels, ...
    W1, b1, W2, b2, W3, b3, Wlin, blin, ...
    useGPU, dropout_prob)

total_samples = length(features);
num_correct   = 0;
sum_loss      = 0;

for i = 1:total_samples
    X = features{i};
    A = edge_indices{i};
    y = labels{i};
    
    [output, ~, ~, ~, ~, ~, ~] = ...
        forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropout_prob);
    
    if useGPU
        output = gather(output);
        y = gather(y);
    end
    output = double(output);
    y      = double(y);
    
    prediction = output > 0.5;
    
    if prediction == y
        num_correct = num_correct + 1;
    end
    
    loss_val = crossEntropyLoss(output, y);
    if useGPU
        loss_val = gather(loss_val);
    end
    
    sum_loss = sum_loss + loss_val;
end

avg_loss = sum_loss / total_samples;
accuracy = num_correct / total_samples;
end
