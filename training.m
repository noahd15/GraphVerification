%% Check for GPU availability
if gpuDeviceCount > 0
    g = gpuDevice(1);
    fprintf('GPU detected: %s. Using GPU features.\n', g.Name);
    useGPU = true;
else
    fprintf('No GPU detected. Running on CPU.\n');
    useGPU = false;
end

%% Load data (assumes data contains cell arrays: edge_indices, features, labels)
data = load('converted_dataset.mat');
edge_indices = data.edge_indices;  % Each cell is an edge-index (adjacency) matrix
features = data.features;          % Each cell is a feature matrix
labels = data.labels;              % Each cell is a label

% Optionally move data to GPU
if useGPU
    for i = 1:length(features)
        features{i} = gpuArray(features{i});
        edge_indices{i} = gpuArray(edge_indices{i});
        labels{i} = gpuArray(labels{i});
    end
end

first_features = features{1};

%% Initialize weights and biases
W1 = randn(size(first_features, 2), 64);
b1 = zeros(1, 64);
W2 = randn(64, 64);
b2 = zeros(1, 64);
W3 = randn(64, 64);
b3 = zeros(1, 64);
Wlin = randn(64, 1);
blin = 0;

if useGPU
    W1 = gpuArray(W1);
    b1 = gpuArray(b1);
    W2 = gpuArray(W2);
    b2 = gpuArray(b2);
    W3 = gpuArray(W3);
    b3 = gpuArray(b3);
    Wlin = gpuArray(Wlin);
    blin = gpuArray(blin);
end

%% Training parameters
num_epochs = 10;
learning_rate = 0.01;
dropout_prob = 0.5;

%% Training Loop
for epoch = 1:num_epochs
    for i = 1:length(edge_indices)
        X = features{i};
        A = edge_indices{i};
        y = labels{i};  % Assumed to be scalar or 0/1 for binary classification

        % Forward pass (dropout applied with dropout_prob)
        [output, X1, X2, X3_drop, dropout_mask, A_norm, X3_pool] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropout_prob);

        % Compute loss
        loss = crossEntropyLoss(output, y);

        % Backward pass
        [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = backward(X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, X1, X2, X3_drop, dropout_mask, A_norm, X3_pool);

        % Update weights
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
        W3 = W3 - learning_rate * dW3;
        b3 = b3 - learning_rate * db3;
        Wlin = Wlin - learning_rate * dWlin;
        blin = blin - learning_rate * dblin;
    end
    % Gather loss if on GPU
    if useGPU, loss = gather(loss); end
    fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
end

%% Testing Loop
num_correct = 0;
total_samples = length(edge_indices);

for i = 1:total_samples
    X = features{i};
    A = edge_indices{i};
    y = labels{i};

    % For testing, disable dropout (set dropout_prob = 0)
    [output, ~, ~, ~, ~, ~, ~] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);

    % Prediction using a 0.5 threshold
    prediction = (output > 0.5);
    num_correct = num_correct + (prediction == y);

    % Optionally compute test loss
    loss = crossEntropyLoss(output, y);
end

accuracy = num_correct / total_samples;
if useGPU, accuracy = gather(accuracy); end
fprintf('Test Accuracy: %.4f\n', accuracy);

%% Helper Functions

function A_norm = computeA_norm(A)
    % Computes the symmetric normalized adjacency matrix.
    % Add a small constant (eps) to avoid division by zero.
    if isempty(A)
        A_norm = [];
    else
        eps_val = 1e-10;
        D = diag( (sum(A, 2) + eps_val).^-0.5 );
        A_norm = D * A * D;
    end
end

function X_out = graphConv_withA_norm(X, A_norm, W, b)
    % Graph convolution using the precomputed A_norm.
    if isempty(A_norm)
        X_out = X * W + b;
    else
        X_out = A_norm * X * W + b;
    end
end

function Y = relu(X)
    % ReLU activation function.
    Y = max(0, X);
end

function Y = sigmoid(X)
    % Sigmoid activation.
    Y = 1 ./ (1 + exp(-X));
end

function [X_drop, mask] = dropout(X, p)
    % Applies dropout to X with dropout probability p.
    mask = (rand(size(X)) > p);
    X_drop = X .* mask;
end

function dX = dropoutGrad(dY, mask)
    % Backpropagate through dropout using the same mask.
    dX = dY .* mask;
end

function [output, X1, X2, X3_drop, dropout_mask, A_norm, X3_pool] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropout_prob)
    % Computes a forward pass through the GCN.
    A_norm = computeA_norm(A);
    
    % Layer 1
    X1 = graphConv_withA_norm(X, A_norm, W1, b1);
    X1 = relu(X1);
    
    % Layer 2
    X2 = graphConv_withA_norm(X1, A_norm, W2, b2);
    X2 = relu(X2);
    
    % Layer 3
    X3 = graphConv_withA_norm(X2, A_norm, W3, b3);
    
    % Apply dropout (store mask for backprop)
    [X3_drop, dropout_mask] = dropout(X3, dropout_prob);
    
    % Mean pooling over nodes (pooling along rows)
    X3_pool = mean(X3_drop, 1);
    
    % Linear layer followed by sigmoid activation
    linear_output = X3_pool * Wlin + blin;
    output = sigmoid(linear_output);
end

function loss = crossEntropyLoss(predictions, targets)
    % Computes cross-entropy loss with numerical stability adjustments.
    epsilon = 1e-10;
    if size(predictions,2) == 1
        % Binary classification using logistic loss
        predictions = max(predictions, epsilon);
        predictions = min(predictions, 1 - epsilon);
        N = size(predictions, 1);
        targets = double(targets);
        predictions = double(predictions);
        loss = -sum(targets .* log(predictions) + (1 - targets) .* log(1 - predictions)) / N;
    else
        % Multi-class classification using softmax and cross-entropy loss
        predictions = predictions - max(predictions, [], 2); % numerical stability
        exp_preds = exp(predictions);
        softmax_probs = exp_preds ./ sum(exp_preds, 2);
        softmax_probs = max(softmax_probs, epsilon);
        N = size(predictions, 1);
        loss = -sum(log(softmax_probs(sub2ind(size(softmax_probs), (1:N)', targets)))) / N;
    end
end

function [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = backward(X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, X1, X2, X3_drop, dropout_mask, A_norm, X3_pool)
    % Backward pass for the network.
    linear_output = X3_pool * Wlin + blin;
    output = 1 ./ (1 + exp(-linear_output));
    dOutput = double(output) - double(y);  % scalar difference

    % Gradients for linear layer
    dWlin = X3_pool' * dOutput;
    dblin = dOutput;

    % Backprop through mean pooling (assume n nodes)
    n = size(X3_drop, 1);
    dX3_pool = repmat(dOutput * Wlin', n, 1) / n;
    
    % Backprop through dropout
    dX3 = dropoutGrad(dX3_pool, dropout_mask);
    
    % Backprop through Layer 3: X3 = A_norm * X2 * W3 + b3
    if isempty(A_norm)
        dX3 = dX3 * W3';
    else
        dX3 = A_norm' * dX3 * W3';
    end
    
    if isempty(A_norm)
        dW3 = X2' * dX3;
    else
        dW3 = X2' * (A_norm' * dX3);
    end
    db3 = sum(dX3, 1);
    
    if isempty(A_norm)
        dX2_from3 = dX3 * W3';
    else
        dX2_from3 = A_norm * (dX3 * W3');
    end

    % Backprop through ReLU of Layer 2
    dX2 = dX2_from3 .* (X2 > 0);
    
    % Backprop through Layer 2: X2 = A_norm * X1 * W2 + b2
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

    % Backprop through ReLU of Layer 1
    dX1 = dX1_from2 .* (X1 > 0);
    
    % Backprop through Layer 1: X1 = A_norm * X * W1 + b1
    if isempty(A_norm)
        dW1 = X' * dX1;
    else
        dW1 = X' * (A_norm' * dX1);
    end
    db1 = sum(dX1, 1);
end
