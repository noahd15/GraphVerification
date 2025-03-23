%% GCN Model Verification and Analysis
% This script evaluates the trained GCN model on test data and provides
% performance metrics and visualizations

% Check for GPU
if gpuDeviceCount > 0
    g = gpuDevice(1);
    fprintf('GPU detected: %s. Using GPU for verification.\n', g.Name);
    useGPU = true;
else
    fprintf('No GPU detected. Running verification on CPU.\n');
    useGPU = false;
end

useGPU = false;  % Force CPU for consistent verification

%% Load trained model and test data
fprintf('Loading model and test data...\n');
model_path = fullfile('./', 'TrainingFiles', 'trained_model.mat');
data_path = fullfile('./', 'TrainingFiles', 'converted_dataset.mat');
run('TrainingFiles/training_added_graphs.m');  % Load the dataset
% model = load(model_path);
data = load(data_path);

% Get model parameters
% W1 = model.model.W1;
% b1 = model.model.b1;
% W2 = model.model.W2;
% b2 = model.model.b2;
% W3 = model.model.W3;
% b3 = model.model.b3;
% Wlin = model.model.Wlin;
% blin = model.model.blin;

W1 = model.W1;
b1 = model.b1;
W2 = model.W2;
b2 = model.b2;
W3 = model.W3;
b3 = model.b3;
Wlin = model.Wlin;
blin = model.blin;

% Split data into training and test set (using same split logic as training)
num_samples = length(data.edge_indices);
rng(42);  % Set seed for reproducibility
rand_indices = randperm(num_samples);
split_ratio = 0.7;
num_train = round(split_ratio * num_samples);

train_indices = rand_indices(1:num_train);
test_indices = rand_indices(num_train+1:end);

test.edge_indices = data.edge_indices(test_indices);
test.features = data.features(test_indices);
test.labels = data.labels(test_indices);

fprintf('Model loaded. Test set contains %d samples.\n', length(test_indices));

%% Evaluate model on test data

% Make predictions on test data
num_test = length(test.labels);
predictions = zeros(num_test, 1);
probabilities = zeros(num_test, 1);
ground_truth = zeros(num_test, 1);

fprintf('Making predictions on test data...\n');
for i = 1:num_test
    X = test.features{i};
    A = test.edge_indices{i};
    y = test.labels{i};
    
    % Forward pass without dropout
    [output, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
    
    if useGPU
        output = gather(output);
    end
    
    probabilities(i) = double(output);
    predictions(i) = double(output > 0.5);
    ground_truth(i) = double(y);
end

%% Calculate performance metrics

% Accuracy
accuracy = sum(predictions == ground_truth) / num_test;

% Precision, Recall, F1 Score
true_positives = sum((predictions == 1) & (ground_truth == 1));
false_positives = sum((predictions == 1) & (ground_truth == 0));
false_negatives = sum((predictions == 0) & (ground_truth == 1));

precision = true_positives / (true_positives + false_positives);
recall = true_positives / (true_positives + false_negatives);

% Calculate F-scores with different beta values
f_scores = zeros(3, 1);
betas = [0.5, 1, 2];  % Same as in the GAT example

for i = 1:length(betas)
    beta = betas(i);
    f_scores(i) = (1 + beta^2) * precision * recall / ((beta^2 * precision) + recall);
end

% Create results table
metrics_table = table();
metrics_table.Beta = betas';
metrics_table.FScore = f_scores;
metrics_table.Precision = repmat(precision, 3, 1);
metrics_table.Recall = repmat(recall, 3, 1);

% Display metrics
fprintf('\nPerformance Metrics:\n');
fprintf('Accuracy: %.4f\n', accuracy);
disp(metrics_table);

%% Confusion Matrix
figure;
cm = confusionchart(ground_truth, predictions);
cm.Title = 'Confusion Matrix for GCN Model';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(gcf, 'Data/gcn_confusion_matrix.png');

%% ROC Curve
[X, Y, T, AUC] = perfcurve(ground_truth, probabilities, 1);

figure;
plot(X, Y, 'LineWidth', 2);
hold on;
plot([0, 1], [0, 1], '--r', 'LineWidth', 1.5);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(AUC, '%.4f') ')']);
legend('GCN Model', 'Random Classifier', 'Location', 'southeast');
grid on;
saveas(gcf, 'Data/gcn_roc_curve.png');

%% Sample Visualization
% Find a correctly classified and incorrectly classified sample
correct_idx = find((predictions == ground_truth) & (ground_truth == 1), 1, 'first');
incorrect_idx = find(predictions ~= ground_truth, 1, 'first');

sample_indices = [correct_idx, incorrect_idx];
sample_labels = {'Correctly Classified Sample', 'Incorrectly Classified Sample'};

for s = 1:length(sample_indices)
    idx = sample_indices(s);
    if isempty(idx)
        continue;
    end
    
    X = test.features{idx};
    A = test.edge_indices{idx};
    true_label = ground_truth(idx);
    pred_label = predictions(idx);
    prob = probabilities(idx);
    
    % Create graph visualization
    figure;
    G = graph(A);
    p = plot(G, 'Layout', 'force');
    p.NodeColor = 'blue';
    p.MarkerSize = 8;
    
    % Compute node importance by analyzing activation values
    [~, X1, X2, ~, ~, ~, ~, A_norm, ~] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0);
    
    % Use the activations at layer 2 to determine node importance
    node_importance = mean(X2, 2);
    if useGPU
        node_importance = gather(node_importance);
    end
    node_importance = rescale(node_importance);
    
    % Color nodes based on importance
    colormap('jet');
    p.NodeCData = node_importance;
    colorbar;
    
    title(sprintf('%s\nTrue: %d, Predicted: %d (Prob: %.4f)', ...
        sample_labels{s}, true_label, pred_label, prob));
    saveas(gcf, sprintf('Data/gcn_sample_%d_visualization.png', s));
end

%% Save verification results
verification_results = struct();
verification_results.accuracy = accuracy;
verification_results.precision = precision;
verification_results.recall = recall;
verification_results.f_scores = f_scores;
verification_results.AUC = AUC;
verification_results.num_test_samples = num_test;

save('Data/gcn_verification_results.mat', 'verification_results');
fprintf('\nVerification completed. Results saved to gcn_verification_results.mat\n');

%% Helper Functions
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
    output = sigmoid(linear_output);
end