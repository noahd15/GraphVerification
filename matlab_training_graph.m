% GPU Check
if gpuDeviceCount > 0
    g = gpuDevice(1);
    fprintf('GPU detected: %s. Using GPU features.\n', g.Name);
    useGPU = true;
else
    fprintf('No GPU detected. Running on CPU.\n');
    useGPU = false;
end

useGPU = false;  

data = load('data/dataset_matlab_graph.mat');

% Split data into training and test set
num_samples = length(data.edge_indices);
rand_indices = randperm(num_samples);
split_ratio = 0.7;
num_train = round(split_ratio * num_samples);

train_indices = rand_indices(1:num_train);
test_indices = rand_indices(num_train+1:end);

train.edge_indices = data.edge_indices(train_indices);
train.features = data.features(train_indices);
train.labels = data.labels(train_indices);

test.edge_indices = data.edge_indices(test_indices);
test.features = data.features(test_indices);
test.labels = data.labels(test_indices);

test_edge_indices = test.edge_indices;
test_features = test.features;
test_labels = test.labels;

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

hidden_size = 32;
W1 = initializeGlorot([size(first_features,2), hidden_size], hidden_size, size(first_features,2));
b1 = zeros(1, hidden_size, 'single');
W2 = initializeGlorot([hidden_size, hidden_size], hidden_size, hidden_size);
b2 = zeros(1, hidden_size, 'single');
W3 = initializeGlorot([hidden_size, hidden_size], hidden_size, hidden_size);
b3 = zeros(1, hidden_size, 'single');
Wlin = initializeGlorot([hidden_size, 1], 1, hidden_size);
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

num_epochs   = 10;
learning_rate = 0.00002;
dropout_prob  = 0.2;

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
        
        [output, X1, X2, X3_drop, mask1, mask2, mask3, A_norm, X3_pool] = ...
            forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropout_prob);
        
        loss = crossEntropyLoss(output, y);
        
        [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = ...
            backward(X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, ...
                     X1, X2, X3_drop, mask1, mask2, mask3, A_norm, X3_pool);
        
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
        test_features, test_edge_indices, test_labels, W1, b1, W2, b2, W3, b3, Wlin, blin, useGPU, 0.0);
    
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

% Save the trained model parameters
model.W1 = W1;
model.b1 = b1;
model.W2 = W2;
model.b2 = b2;
model.W3 = W3;
model.b3 = b3;
model.Wlin = Wlin;
model.blin = blin;

if ~exist('logs', 'dir')
    mkdir('logs');
end

if ~exist('logs/trained_models', 'dir')
    mkdir('logs/trained_models');
end

if ~exist('logs/results', 'dir')
    mkdir('logs/results');
end

save('logs/trained_models/trained_graph_model.mat', 'model');
fprintf('Model saved to trained_model.mat\n');

figure;
plot(1:num_epochs, train_losses, '-o', 'LineWidth', 1.5); hold on;
plot(1:num_epochs, test_losses, '-x', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss');
title('Training and Testing Loss');
legend('Train Loss','Test Loss','Location','best');
saveas(gcf, 'logs/results/training_testing_graph_loss.png');
grid on;

figure;
plot(1:num_epochs, train_accs, '-o', 'LineWidth', 1.5); hold on;
plot(1:num_epochs, test_accs, '-x', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Accuracy');
title('Training and Testing Accuracy');
legend('Train Accuracy','Test Accuracy','Location','best');
saveas(gcf, 'logs/results/training_testing_graph_accuracy.png');
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

function [output, X1, X2, X3_drop, mask1, mask2, mask3, A_norm, X3_pool] = forward( ...
    X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropout_prob)

    A_norm = computeA_norm(A);

    % First graph convolution + ReLU + dropout
    X1 = relu(graphConv_withA_norm(X, A_norm, W1, b1));
    [X1_drop, mask1] = dropout(X1, dropout_prob);

    % Second graph convolution + ReLU + dropout
    X2 = relu(graphConv_withA_norm(X1_drop, A_norm, W2, b2));
    [X2_drop, mask2] = dropout(X2, dropout_prob);

    % Third graph convolution + dropout
    X3 = graphConv_withA_norm(X2_drop, A_norm, W3, b3);
    [X3_drop, mask3] = dropout(X3, 0.2);

    % Pooling + final linear
    X3_pool = mean(X3_drop, 1);
    linear_output = X3_pool * Wlin + blin;
    output = linear_output;  % or apply sigmoid if desired
 %   output = sigmoid(linear_output);
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
    X1, X2, X3_drop, mask1, mask2, mask3, A_norm, X3_pool)

    % Forward’s final output
    bounded = max(min(X3_pool * Wlin + blin, 15), -15);
    output  = 1 ./ (1 + exp(-bounded));

    dOutput = output - double(y);
    dWlin   = X3_pool' * dOutput;
    dblin   = dOutput;

    % Backprop through pooling
    n        = size(X3_drop, 1);
    dX3_pool = repmat(dOutput * Wlin', n, 1) / n;

    % Backprop through dropout on X3
    dX3 = dropoutGrad(dX3_pool, mask3);

    % GraphConv W3, using X2_drop in forward
    if isempty(A_norm)
        dW3    = (X2 .* mask2)' * dX3;
        dX2tmp = dX3 * W3';
    else
        dW3    = (A_norm * (X2 .* mask2))' * dX3;
        dX2tmp = (A_norm' * dX3) * W3';
    end
    db3 = sum(dX3, 1);

    % Backprop through ReLU + dropout (layer 2)
    dX2tmp = dX2tmp .* (X2 > 0);
    dX2    = dropoutGrad(dX2tmp, mask2);

    if isempty(A_norm)
        dW2    = (X1 .* mask1)' * dX2;
        dX1tmp = dX2 * W2';
    else
        dW2    = (A_norm * (X1 .* mask1))' * dX2;
        dX1tmp = (A_norm' * dX2) * W2';
    end
    db2 = sum(dX2, 1);

    % Backprop through ReLU + dropout (layer 1)
    dX1tmp = dX1tmp .* (X1 > 0);
    dX1    = dropoutGrad(dX1tmp, mask1);

    if isempty(A_norm)
        dW1 = X' * dX1;
    else
        dW1 = (A_norm * X)' * dX1;
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
