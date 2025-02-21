data = load('converted_dataset.mat');
edge_indices = data.edge_indices;  % A cell array where each cell is an edge_index array
features = data.features;          % A cell array of feature matrices
labels = data.labels;              % A cell array of labels

% To access the first graph's data:
first_edge_index = edge_indices{1};
first_features   = features{1};
first_label      = labels{1};

W1 = randn(size(first_features, 2), 64);
b1 = zeros(1, 64);
W2 = randn(64, 64);
b2 = zeros(1, 64);
W3 = randn(64, 64);
b3 = zeros(1, 64);
Wlin = randn(64, 1);
blin = 0;
X1 = [];
X2 = [];
X3 = [];
% -------------------------------------------------------

function dX = graphConvGrad(dY, A, W)
    dX = A * (dY * W');
end

function X = relu(X)
    X(X < 0) = 0;
end

function X = dropout(X, p)
    mask = (rand(size(X)) > p);
    X = X .* mask;
end

function dX = dropoutGrad(dY, p)
    mask = (rand(size(dY)) > p);
    dX = dY .* mask;
end

function X_out = graphConv(X, A, W, b)
    % Compute the normalized adjacency matrix
    D = diag(sum(A, 2).^(-0.5));
    A_norm = D * A * D;
    
    % Perform graph convolution
    X_out = A_norm * X * W + b;
end

% Forward pass through the GCN
function [output, X1, X2, X3] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin)
    % Layer 1
    X1 = graphConv(X, A, W1, b1);
    X1 = relu(X1);
    
    % Layer 2
    X2 = graphConv(X1, A, W2, b2);
    X2 = relu(X2);
    
    % Layer 3
    X3 = graphConv(X2, A, W3, b3);
    
    % Dropout
    X3 = dropout(X3, 0.5);
    
    % Mean pooling
    X3 = mean(X3, 1);
    
    % Linear layer
    output = X3 * Wlin + blin;
end

% -------------------------------------------------------
% Loss function
function loss = crossEntropyLoss(predictions, labels)
    % Compute the cross-entropy loss
    predictions = double(predictions);
    labels = double(labels);
    loss = -mean(labels .* log(predictions + 1e-10));
end

% Backward pass
function [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = backward(X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, X1, X2, X3)
    % Forward pass
    [output, X1, X2, X3] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin);
    
    % Compute the derivatives
    dOutput = double(output) - double(y);
    dWlin = X3' * dOutput;
    dblin = sum(dOutput);
    
    dX3 = dOutput * Wlin';
    dX3 = dX3 ./ size(X3, 1);
    
    dX3 = dropoutGrad(dX3, 0.5);
    
    dX2 = graphConvGrad(dX3, A, W3);
    dW3 = X2' * dX3;
    db3 = sum(dX3, 1);
    dX2 = dX2 .* (X2 > 0);
    
    dX1 = graphConvGrad(dX2, A, W2);
    dW2 = X1' * dX2;
    db2 = sum(dX2, 1);
    dX1 = dX1 .* (X1 > 0);
    
    dX = graphConvGrad(dX1, A, W1);
    dW1 = X' * dX1;
    db1 = sum(dX1, 1);
end

% Training loop
num_epochs = 100;
learning_rate = 0.01;

for epoch = 1:num_epochs
    for i = 1:length(edge_indices)
        X = features{i};
        A = edge_indices{i};
        y = labels{i};
        
        % Forward pass
        [output, X1, X2, X3] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin);
        
        % Compute loss
        loss = crossEntropyLoss(output, y);
        
        % Backward pass
        [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = backward(X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, X1, X2, X3);
        
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
end

% Testing loop
num_correct = 0;
total_samples = length(edge_indices);

for i = 1:total_samples
    X = features{i};
    A = adjacencyMatrix(edge_indices{i});
    y = labels{i};
    
    % Forward pass
    output = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin);
    
    % Compute predictions
    prediction = (output > 0.5);
    
    % Update accuracy
    num_correct = num_correct + (prediction == y);
end

accuracy = num_correct / total_samples;
disp(['Test Accuracy: ', num2str(accuracy)]);