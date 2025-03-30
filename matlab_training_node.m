%% GPU Check
if gpuDeviceCount > 0
    g = gpuDevice(1);
    fprintf('GPU detected: %s. Using GPU features.\n', g.Name);
    useGPU = true;
else
    fprintf('No GPU detected. Running on CPU.\n');
    useGPU = false;
end

% For debugging purposes, force CPU usage:
useGPU = false;  

%% Load Data
data = load('data/dataset_matlab_node.mat');

% Ensure every label cell is a column vector.
for i = 1:length(data.labels)
    if isrow(data.labels{i})
        data.labels{i} = data.labels{i}';
    end
end

% (Optional) You can also inspect one sample:
% disp('Size of first label: ');
% disp(size(data.labels{1}));

%% Split Data into Training and Test Sets
num_samples = length(data.edge_indices);
rand_indices = randperm(num_samples);
split_ratio = 0.7;
num_train = round(split_ratio * num_samples);

train_indices = rand_indices(1:num_train);
test_indices = rand_indices(num_train+1:end);

train.edge_indices = data.edge_indices(train_indices);
train.features    = data.features(train_indices);
train.labels      = data.labels(train_indices);

test.edge_indices = data.edge_indices(test_indices);
test.features    = data.features(test_indices);
test.labels      = data.labels(test_indices);

test_edge_indices = test.edge_indices;
test_features = test.features;
test_labels   = test.labels;

% For convenience, use entire dataset variables too.
edge_indices = data.edge_indices;
features = data.features;          
labels = data.labels;             

%% Move Data to GPU if Applicable
if useGPU
    for i = 1:length(features)
        features{i}    = gpuArray(features{i});
        edge_indices{i}= gpuArray(edge_indices{i});
        labels{i}      = gpuArray(labels{i});
    end
end

first_features = features{1};

%% Glorot Weight Initialization Function
function weights = initializeGlorot(sz, numOut, numIn)
    Z = 2 * rand(sz, 'single') - 1;
    bound = sqrt(6 / (numIn + numOut));
    weights = bound * Z;
    weights = dlarray(weights);
end

%% Determine Number of Classes
first_label = labels{1};
% If labels are node-level, then first_label should be an N x 1 vector.
[~, num_classes_from_label] = size(first_label);
fprintf('First label size: %dx%d\n', size(first_label,1), size(first_label,2));
% If your labels are not one-hot, num_classes_from_label might be 1.
% You may set num_classes manually if you know the number of classes:
num_classes = 3; % (Set to 2 for binary classification, or remove this if determined by your data)
fprintf('Using num_classes = %d\n', num_classes);

%% Initialize Model Parameters
hidden_size = 32;
W1 = initializeGlorot([size(first_features,2), hidden_size], hidden_size, size(first_features,2));
b1 = zeros(1, hidden_size, 'single');
W2 = initializeGlorot([hidden_size, hidden_size], hidden_size, hidden_size);
b2 = zeros(1, hidden_size, 'single');
W3 = initializeGlorot([hidden_size, hidden_size], hidden_size, hidden_size);
b3 = zeros(1, hidden_size, 'single');
Wlin = initializeGlorot([hidden_size, num_classes], num_classes, hidden_size);
blin = zeros(1, num_classes, 'single');

if useGPU
    W1   = gpuArray(W1);  b1   = gpuArray(b1);
    W2   = gpuArray(W2);  b2   = gpuArray(b2);
    W3   = gpuArray(W3);  b3   = gpuArray(b3);
    Wlin = gpuArray(Wlin); blin = gpuArray(blin);
end

%% Calculate Class Counts from Training Labels (Node-Level)
num_train_samples = length(train.labels);
train_total_nodes = 0;
train_class_counts = zeros(1, num_classes);

for i = 1:num_train_samples
    label = train.labels{i};
    % Count occurrences for classes 0, 1, ... (num_classes-1)
    for class_idx = 1:num_classes
        train_class_counts(class_idx) = train_class_counts(class_idx) + sum(label == (class_idx-1));
    end
    train_total_nodes = train_total_nodes + numel(label);
end

fprintf('Class distribution (train): ');
fprintf('%d ', train_class_counts);
fprintf('\n');

%% Calculate Class Weights Inversely Proportional to Class Frequencies
class_weights = zeros(1, num_classes);
for class_idx = 1:num_classes
    class_count = train_class_counts(class_idx);
    if class_count > 0
        class_weights(class_idx) = train_total_nodes / (num_classes * class_count);
    else
        class_weights(class_idx) = 1.0;
    end
end
class_weights = class_weights / sum(class_weights) * num_classes;
fprintf('Class weights: ');
fprintf('%g ', class_weights);
fprintf('\n');

%% Training Settings
num_epochs    = 20;
learning_rate = 0.00002;
dropout_prob  = 0.2;

train_losses = zeros(num_epochs,1);
test_losses  = zeros(num_epochs,1);
train_accs   = zeros(num_epochs,1);
test_accs    = zeros(num_epochs,1);

%% Main Training Loop
for epoch = 1:num_epochs
    epoch_loss = 0;
    
    for i = 1:length(edge_indices)
        X = features{i};
        A = edge_indices{i};
        y = labels{i}; 
        
        % Print dimensions for debugging (for first iteration only)
        if epoch == 1 && i == 1
            fprintf('Dimensions - X: %dx%d, A: %dx%d, W1: %dx%d\n', ...
                size(X,1), size(X,2), size(A,1), size(A,2), size(W1,1), size(W1,2));
        end
        
        [output, X1, X2, X3_drop, mask1, mask2, mask3, A_norm, X3_pool] = ...
            forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, dropout_prob);

        loss = crossEntropyLoss(output, y, class_weights);
        
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
        features, edge_indices, labels, W1, b1, W2, b2, W3, b3, Wlin, blin, useGPU, 0.0, class_weights);

    [test_loss_epoch, test_acc_epoch] = computeLossAccuracy( ...
        test_features, test_edge_indices, test_labels, W1, b1, W2, b2, W3, b3, Wlin, blin, useGPU, 0.0, class_weights);
    
    train_losses(epoch) = train_loss_epoch;
    test_losses(epoch)  = test_loss_epoch;
    train_accs(epoch)   = train_acc_epoch;
    test_accs(epoch)    = test_acc_epoch;
    
    fprintf('Epoch %d:\n', epoch);
    fprintf('  Training Loss (this loop): %.4f\n', epoch_loss); 
    fprintf('  Train Loss (eval): %.4f | Train Acc: %.4f\n', train_loss_epoch, train_acc_epoch);
    fprintf('  Test  Loss: %.4f | Test Acc: %.4f\n', test_loss_epoch, test_acc_epoch);
end

%% Save Model Parameters
model.W1   = W1;   model.b1   = b1;
model.W2   = W2;   model.b2   = b2;
model.W3   = W3;   model.b3   = b3;
model.Wlin = Wlin; model.blin = blin;

if ~exist('logs', 'dir'), mkdir('logs'); end
if ~exist('logs/trained_models', 'dir'), mkdir('logs/trained_models'); end
if ~exist('logs/results', 'dir'), mkdir('logs/results'); end

save('logs/trained_models/trained_node_model.mat', 'model');
fprintf('Model saved to trained_model.mat\n');

%% Plot Loss and Accuracy
figure;
plot(1:num_epochs, train_losses, '-o', 'LineWidth', 1.5); hold on;
plot(1:num_epochs, test_losses, '-x', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Loss');
title('Training and Testing Loss');
legend('Train Loss','Test Loss','Location','best');
saveas(gcf, 'logs/results/training_testing_node_loss.png');
grid on;

figure;
plot(1:num_epochs, train_accs, '-o', 'LineWidth', 1.5); hold on;
plot(1:num_epochs, test_accs, '-x', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('Accuracy');
title('Training and Testing Accuracy');
legend('Train Accuracy','Test Accuracy','Location','best');
saveas(gcf, 'logs/results/training_testing_node_accuracy.png');
grid on;

%% --- Supporting Functions ---

function A_norm = computeA_norm(A)
    eps_val = 1e-10;
    if size(A,1) ~= 18
        if size(A, 1) > 18
            A = A(1:18, 1:18);
        elseif size(A, 1) < 18
            A_new = zeros(18, 18);
            A_new(1:size(A,1), 1:size(A,2)) = A;
            A = A_new;
        end
        if size(A,2) > 18
            A = A(:, 1:18);
        elseif size(A,2) < 18
            A_new = zeros(18, 18);
            A_new(1:size(A,1), 1:size(A,2)) = A;
            A = A_new;
        end
        D = diag(1 ./ sqrt(max(sum(A, 2), eps_val)));
        A_norm = D * A * D;
    else
        D = diag(1 ./ sqrt(max(sum(A, 2), eps_val)));
        A_norm = D * A * D;
    end
end

function X_out = graphConv_withA_norm(X, A_norm, W, b)
    if isempty(A_norm)
        X_out = X * W + b;
    else
        [n_nodes_A, ~] = size(A_norm);
        [n_nodes_X, ~] = size(X);
        if n_nodes_A ~= n_nodes_X
            fprintf('Warning: Dimension mismatch. Adjusting A_norm to match X.\n');
            if size(A_norm, 2) == 2
                max_node = max(max(A_norm));
                adj_matrix = zeros(max_node, max_node);
                for i = 1:size(A_norm, 1)
                    adj_matrix(A_norm(i,1), A_norm(i,2)) = 1;
                end
                A_norm = adj_matrix;
                [n_nodes_A, ~] = size(A_norm);
            end
            if n_nodes_A < n_nodes_X
                pad_size = n_nodes_X - n_nodes_A;
                A_norm = padarray(A_norm, [pad_size pad_size], 0, 'post');
            elseif n_nodes_A > n_nodes_X
                A_norm = A_norm(1:n_nodes_X, 1:n_nodes_X);
            end
        end
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
    X1 = relu(graphConv_withA_norm(X, A_norm, W1, b1));
    [X1_drop, mask1] = dropout(X1, dropout_prob);
    X2 = relu(graphConv_withA_norm(X1_drop, A_norm, W2, b2));
    [X2_drop, mask2] = dropout(X2, dropout_prob);
    X3 = graphConv_withA_norm(X2_drop, A_norm, W3, b3);
    [X3_drop, mask3] = dropout(X3, 0.2);
    linear_output = X3_drop * Wlin + blin;
    output = linear_output;
    X3_pool = X3_drop;
end

function loss = crossEntropyLoss(logits, targets, class_weights)
    if isa(logits, 'dlarray')
        logits = gather(logits);
    end
    if isa(targets, 'dlarray')
        targets = gather(targets);
    end
    logits = double(logits);
    targets = double(targets);
    [batch_size, num_classes] = size(logits);
    if isvector(targets) && length(targets) == batch_size
        targets = reshape(targets, batch_size, 1);
    elseif size(targets, 1) == 1 && size(targets, 2) == batch_size
        targets = targets';
    elseif size(targets, 2) > 1 && size(targets, 1) ~= batch_size
        if numel(targets) == batch_size
            targets = reshape(targets, batch_size, 1);
        end
    end
    [target_rows, target_cols] = size(targets);
    if target_cols > 1 && target_cols == num_classes
        one_hot_targets = targets;
    else
        one_hot_targets = zeros(batch_size, num_classes);
        for i = 1:min(batch_size, target_rows)
            if i <= numel(targets)
                target_val = round(targets(i));
                class_idx = target_val + 1;
                if class_idx < 1
                    class_idx = 1;
                elseif class_idx > num_classes
                    class_idx = num_classes;
                end
                one_hot_targets(i, class_idx) = 1;
            end
        end
    end
    max_logits = max(logits, [], 2);
    exp_logits = exp(logits - max_logits);
    sum_exp = sum(exp_logits, 2);
    probs = exp_logits ./ sum_exp;
    epsilon = 1e-10;
    probs = max(probs, epsilon);
    if nargin >= 3 && ~isempty(class_weights)
        weighted_losses = zeros(batch_size, 1);
        for i = 1:batch_size
            for c = 1:num_classes
                if one_hot_targets(i, c) > 0
                    weighted_losses(i) = weighted_losses(i) - class_weights(c) * log(probs(i, c));
                end
            end
        end
        loss = mean(weighted_losses);
    else
        losses = -sum(one_hot_targets .* log(probs), 2);
        loss = mean(losses);
    end
end

function [dW1, db1, dW2, db2, dW3, db3, dWlin, dblin] = backward( ...
    X, A, y, W1, b1, W2, b2, W3, b3, Wlin, blin, ...
    X1, X2, X3_drop, mask1, mask2, mask3, A_norm, X3_pool)
    [~, num_classes] = size(Wlin);
    linear_output = X3_drop * Wlin + blin;
    if num_classes > 1
        batch_size = size(linear_output, 1);
        one_hot_y = zeros(batch_size, num_classes);
        for i = 1:batch_size
            class_idx = y(i) + 1;
            if class_idx >= 1 && class_idx <= num_classes
                one_hot_y(i, class_idx) = 1;
            end
        end
        max_scores = max(linear_output, [], 2);
        exp_scores = exp(linear_output - max_scores);
        sum_exp = sum(exp_scores, 2);
        softmax_output = exp_scores ./ sum_exp;
        dOutput = softmax_output - one_hot_y;
    else
        bounded = max(min(linear_output, 15), -15);
        output = 1 ./ (1 + exp(-bounded));
        dOutput = output - double(y);
    end
    dWlin = X3_drop' * dOutput;
    dblin = sum(dOutput, 1);
    dX3_drop = dOutput * Wlin';
    dX3 = dropoutGrad(dX3_drop, mask3);
    if isempty(A_norm)
        dW3 = (X2 .* mask2)' * dX3;
        dX2tmp = dX3 * W3';
    else
        dW3 = (A_norm * (X2 .* mask2))' * dX3;
        dX2tmp = (A_norm' * dX3) * W3';
    end
    db3 = sum(dX3, 1);
    dX2tmp = dX2tmp .* (X2 > 0);
    dX2 = dropoutGrad(dX2tmp, mask2);
    if isempty(A_norm)
        dW2 = (X1 .* mask1)' * dX2;
        dX1tmp = dX2 * W2';
    else
        dW2 = (A_norm * (X1 .* mask1))' * dX2;
        dX1tmp = (A_norm' * dX2) * W2';
    end
    db2 = sum(dX2, 1);
    dX1tmp = dX1tmp .* (X1 > 0);
    dX1 = dropoutGrad(dX1tmp, mask1);
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
    useGPU, dropout_prob, class_weights)
% computeLossAccuracy computes the average loss and accuracy for a set of samples.
%
% Inputs:
%   features      - cell array of feature matrices (each: N x F)
%   edge_indices  - cell array of edge index/adjacency matrices (each: N x N)
%   labels        - cell array of labels (each: N x 1 or N x num_classes if one-hot)
%   W1, b1, ...   - model parameters
%   useGPU        - boolean flag to determine if GPU is used
%   dropout_prob  - dropout probability (set to 0 during evaluation)
%   class_weights - optional vector of class weights (for weighted loss)
%
% Outputs:
%   avg_loss      - average loss over all samples
%   accuracy      - overall accuracy (between 0 and 1)

if nargin < 14
    class_weights = [];
end

total_samples = length(features);
total_nodes = 0;
correct_nodes = 0;
sum_loss = 0;

for i = 1:total_samples
    % Extract data for sample i
    X = features{i};
    A = edge_indices{i};
    y = labels{i};
    
    % Forward pass with dropout set to 0 (evaluation mode)
    [output, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
    
    if useGPU
        output = gather(output);
        y = gather(y);
    end
    
    % Convert to double for consistency
    output = double(output);
    y = double(y);
    
    % Obtain predicted class for each node by finding the max probability
    [~, predicted_class] = max(output, [], 2);
    % MATLAB uses 1-indexing; subtract 1 to match labels if they are 0-indexed
    predicted_class = double(predicted_class - 1);
    
    % If labels are in one-hot format (i.e., more than one column), convert them
    [~, num_cols] = size(y);
    if num_cols > 1
        [~, y_indices] = max(y, [], 2);
        y = double(y_indices - 1);
    end
    
    % Calculate the number of correct predictions for this sample
    correct = (predicted_class == y);
    num_correct = sum(correct);
    num_nodes = numel(y);
    
    % Accumulate the counts for overall accuracy
    correct_nodes = correct_nodes + num_correct;
    total_nodes = total_nodes + num_nodes;
    
    % Compute the loss for this sample using the crossEntropyLoss function
    loss_val = crossEntropyLoss(output, y, class_weights);
    if useGPU
        loss_val = gather(loss_val);
    end
    sum_loss = sum_loss + loss_val;
end

% Average loss is the mean loss over all samples
avg_loss = sum_loss / total_samples;
% Overall accuracy is the total correct predictions divided by total predictions
accuracy = correct_nodes / total_nodes;

end
