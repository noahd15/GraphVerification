%% GCN Perturbation Analysis
% This script analyzes the sensitivity of the GCN model to input perturbations
% without relying on the NNV toolbox

%% Load trained model and test data
fprintf('Loading model and test data for perturbation analysis...\n');
model_path = fullfile('./', 'TrainingFiles', 'trained_model.mat');
data_path = fullfile('./', 'TrainingFiles', 'converted_dataset.mat');

model = load(model_path);
data = load(data_path);

% Get model parameters
W1 = model.model.W1;
b1 = model.model.b1;
W2 = model.model.W2;
b2 = model.model.b2;
W3 = model.model.W3;
b3 = model.model.b3;
Wlin = model.model.Wlin;
blin = model.model.blin;

% Split data into training and test set
% Uses similar split logic as in training_added_graphs.m
num_samples = length(data.edge_indices);
rng(42);  % Consistent seed for reproducibility
rand_indices = randperm(num_samples);
split_ratio = 0.7;
num_train = round(split_ratio * num_samples);

train_indices = rand_indices(1:num_train);
test_indices = rand_indices(num_train+1:end);

test.edge_indices = data.edge_indices(test_indices);
test.features = data.features(test_indices);
test.labels = data.labels(test_indices);

fprintf('Model loaded. Test set contains %d samples.\n', length(test_indices));

%% 1. Feature Perturbation Analysis
% Select a sample for perturbation analysis
sample_idx = 1;  % Choose first test sample
X_orig = test.features{sample_idx};
A = test.edge_indices{sample_idx};
y_true = test.labels{sample_idx};

% Get original prediction
[output_orig, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X_orig, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
pred_orig = double(output_orig > 0.5);

fprintf('Analyzing sample %d: %d nodes, true label: %d, predicted: %d (prob: %.4f)\n', ...
    sample_idx, size(X_orig, 1), double(y_true), pred_orig, double(output_orig));

% Single Feature Perturbation Analysis
fprintf('Performing single feature perturbation analysis...\n');

perturbation_range = -0.5:0.05:0.5;  % -50% to +50% in 5% increments
num_features = size(X_orig, 2);
feature_outputs = zeros(length(perturbation_range), num_features);
feature_decisions = zeros(length(perturbation_range), num_features);

for f = 1:num_features
    for p = 1:length(perturbation_range)
        % Create perturbed input
        X_perturbed = X_orig;
        perturbation = perturbation_range(p);
        X_perturbed(:, f) = X_orig(:, f) * (1 + perturbation);
        
        % Forward pass with perturbed input
        [output_perturbed, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X_perturbed, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
        
        % Store output and decision
        feature_outputs(p, f) = double(output_perturbed);
        feature_decisions(p, f) = double(output_perturbed > 0.5);
    end
end

% Find most sensitive features (those that caused decision changes)
feature_sensitivity = sum(abs(diff(feature_decisions)), 1);
[sorted_sensitivity, sorted_idx] = sort(feature_sensitivity, 'descend');
top_features = sorted_idx(1:min(5, num_features));

% Create visualization
figure;
subplot(2, 1, 1);
imagesc(feature_decisions);
colormap([0.8 0.8 1; 1 0.8 0.8]);  % Light blue for 0, light red for 1
xlabel('Feature Index');
ylabel('Perturbation Level');
yticks(1:length(perturbation_range));
yticklabels(arrayfun(@(x) sprintf('%.0f%%', x*100), perturbation_range, 'UniformOutput', false));
title('Decision Changes with Feature Perturbation');
colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'Class 0', 'Class 1'});

% Plot output probabilities for top sensitive features
subplot(2, 1, 2);
for i = 1:min(5, length(top_features))
    f = top_features(i);
    plot(perturbation_range*100, feature_outputs(:, f), 'LineWidth', 2);
    hold on;
end
yline(0.5, '--k', 'Decision Boundary', 'LineWidth', 1.5);
xlabel('Perturbation (%)');
ylabel('Output Probability');
title('Output Probability for Most Sensitive Features');
legend(['Feature ' num2str(top_features(1))], ...
       ['Feature ' num2str(top_features(2))], ...
       ['Feature ' num2str(top_features(3))], ...
       ['Feature ' num2str(top_features(4))], ...
       ['Feature ' num2str(top_features(5))], ...
       'Decision Boundary', 'Location', 'best');
grid on;
saveas(gcf, 'gcn_feature_sensitivity.png');

%% 2. Uniform Perturbation Analysis
% Perturb all features uniformly to see overall robustness
fprintf('Performing uniform perturbation analysis...\n');

uniform_range = -0.5:0.02:0.5;  % -50% to +50% in 2% increments
uniform_outputs = zeros(length(uniform_range), 1);

for p = 1:length(uniform_range)
    % Apply uniform perturbation to all features
    perturbation = uniform_range(p);
    X_perturbed = X_orig * (1 + perturbation);
    
    % Forward pass
    [output_perturbed, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X_perturbed, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
    uniform_outputs(p) = double(output_perturbed);
end

% Find perturbation needed to change decision
decision_changes = [false; diff(uniform_outputs > 0.5) ~= 0];
decision_points = uniform_range(decision_changes);

% Create visualization
figure;
plot(uniform_range*100, uniform_outputs, 'b-', 'LineWidth', 2);
hold on;
yline(0.5, '--k', 'Decision Boundary', 'LineWidth', 1.5);
if ~isempty(decision_points)
    for i = 1:length(decision_points)
        xline(decision_points(i)*100, 'r--', 'LineWidth', 1.5);
    end
end
xlabel('Uniform Perturbation (%)');
ylabel('Output Probability');
title('GCN Response to Uniform Feature Perturbation');
grid on;
saveas(gcf, 'gcn_uniform_perturbation.png');

%% 3. Random Perturbation Analysis
% Apply random noise to features to simulate natural variation
fprintf('Performing random perturbation analysis...\n');

noise_levels = 0:0.05:0.5;  % 0% to 50% in 5% increments
num_trials = 100;
random_outputs = zeros(length(noise_levels), num_trials);
decision_flips = zeros(length(noise_levels), 1);

for n = 1:length(noise_levels)
    noise = noise_levels(n);
    
    for t = 1:num_trials
        % Apply random noise to features
        noise_factors = 1 + noise * (2*rand(size(X_orig)) - 1);  % Uniform noise in [-noise, +noise]
        X_noisy = X_orig .* noise_factors;
        
        % Forward pass
        [output_noisy, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X_noisy, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
        random_outputs(n, t) = double(output_noisy);
        
        % Check if decision flipped
        if (output_noisy > 0.5) ~= (output_orig > 0.5)
            decision_flips(n) = decision_flips(n) + 1;
        end
    end
end

% Calculate flip percentage
flip_percentage = decision_flips / num_trials * 100;

% Create visualization
figure;
subplot(2, 1, 1);
boxplot(random_outputs', 'Labels', arrayfun(@(x) sprintf('%.0f%%', x*100), noise_levels, 'UniformOutput', false));
hold on;
yline(0.5, '--r', 'Decision Boundary', 'LineWidth', 1.5);
yline(double(output_orig), '--g', 'Original Output', 'LineWidth', 1.5);
xlabel('Noise Level');
ylabel('Output Probability');
title('Distribution of Outputs with Random Noise');
grid on;

subplot(2, 1, 2);
bar(noise_levels*100, flip_percentage);
xlabel('Noise Level (%)');
ylabel('Decision Flips (%)');
title('Percentage of Decision Changes with Random Noise');
grid on;
saveas(gcf, 'gcn_random_noise.png');

%% 4. Node Removal Analysis (Graph Structure Perturbation)
% Test robustness to missing nodes
fprintf('Performing node removal analysis...\n');

num_nodes = size(X_orig, 1);
removed_nodes = 1:min(5, num_nodes);  % Remove up to 5 nodes
num_trials = 10;  % For each number of removed nodes, try different combinations

node_removal_outputs = zeros(length(removed_nodes), num_trials);
node_removal_decisions = zeros(length(removed_nodes), num_trials);

for r = 1:length(removed_nodes)
    num_remove = removed_nodes(r);
    
    for t = 1:num_trials
        % Randomly select nodes to remove
        available_nodes = 1:num_nodes;
        remove_idx = randperm(num_nodes, num_remove);
        keep_idx = setdiff(available_nodes, remove_idx);
        
        % Create reduced graph
        X_reduced = X_orig(keep_idx, :);
        A_reduced = A(keep_idx, keep_idx);
        
        % Forward pass
        [output_reduced, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X_reduced, A_reduced, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
        
        node_removal_outputs(r, t) = double(output_reduced);
        node_removal_decisions(r, t) = double(output_reduced > 0.5);
    end
end

% Calculate decision change percentage
decision_change_pct = sum(node_removal_decisions ~= pred_orig, 2) / num_trials * 100;

% Create visualization
figure;
boxplot(node_removal_outputs', 'Labels', arrayfun(@(x) sprintf('%d', x), removed_nodes, 'UniformOutput', false));
hold on;
yline(0.5, '--r', 'Decision Boundary', 'LineWidth', 1.5);
yline(double(output_orig), '--g', 'Original Output', 'LineWidth', 1.5);
xlabel('Number of Nodes Removed');
ylabel('Output Probability');
title('Effect of Node Removal on GCN Output');
grid on;
saveas(gcf, 'gcn_node_removal.png');

% Create figure for decision change percentage
figure;
bar(removed_nodes, decision_change_pct);
xlabel('Number of Nodes Removed');
ylabel('Decision Changes (%)');
title('Impact of Node Removal on GCN Decisions');
grid on;
saveas(gcf, 'gcn_node_removal_decisions.png');

%% 5. Two-Dimensional Feature Perturbation Map
% Create a 2D visualization of how perturbing two features affects output
fprintf('Creating 2D feature perturbation map...\n');

% Use the top 2 most sensitive features
feature1 = top_features(1);
feature2 = top_features(2);

% Create perturbation grid
perturbation_grid = -0.3:0.02:0.3;  % -30% to +30%
[X1, X2] = meshgrid(perturbation_grid, perturbation_grid);
output_grid = zeros(size(X1));
decision_grid = zeros(size(X1));

for i = 1:length(perturbation_grid)
    for j = 1:length(perturbation_grid)
        % Create perturbed input
        X_perturbed = X_orig;
        X_perturbed(:, feature1) = X_orig(:, feature1) * (1 + perturbation_grid(i));
        X_perturbed(:, feature2) = X_orig(:, feature2) * (1 + perturbation_grid(j));
        
        % Forward pass
        [output_perturbed, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X_perturbed, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
        
        output_grid(j, i) = double(output_perturbed);
        decision_grid(j, i) = double(output_perturbed > 0.5);
    end
end

% Visualize 2D perturbation map
figure;
subplot(2, 1, 1);
contourf(X1*100, X2*100, output_grid, 20);
hold on;
contour(X1*100, X2*100, decision_grid, [0.5 0.5], 'r', 'LineWidth', 2);
plot(0, 0, 'k*', 'MarkerSize', 10);  % Original point
colorbar;
xlabel(['Feature ' num2str(feature1) ' Perturbation (%)']);
ylabel(['Feature ' num2str(feature2) ' Perturbation (%)']);
title('Output Probability Map for Feature Perturbations');

% Calculate decision boundary distance
subplot(2, 1, 2);
imagesc(perturbation_grid*100, perturbation_grid*100, decision_grid);
colormap([0.8 0.8 1; 1 0.8 0.8]);  % Light blue for 0, light red for 1
hold on;
contour(X1*100, X2*100, output_grid, [0.5 0.5], 'k', 'LineWidth', 2);
plot(0, 0, 'k*', 'MarkerSize', 10);  % Original point
xlabel(['Feature ' num2str(feature1) ' Perturbation (%)']);
ylabel(['Feature ' num2str(feature2) ' Perturbation (%)']);
title('Decision Regions for Feature Perturbations');
colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'Class 0', 'Class 1'});
saveas(gcf, 'gcn_2d_feature_map.png');

%% 6. Multiple Sample Analysis
% Check robustness across multiple test samples
fprintf('Analyzing robustness across multiple test samples...\n');

num_test_samples = min(50, length(test_indices));
epsilon = 0.1;  % 10% perturbation
decision_changes = zeros(num_test_samples, 1);
output_deltas = zeros(num_test_samples, 1);
distances_to_boundary = zeros(num_test_samples, 1);

for s = 1:num_test_samples
    X = test.features{s};
    A = test.edge_indices{s};
    y = test.labels{s};
    
    % Get original prediction
    [output_orig, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
    pred_orig = double(output_orig > 0.5);
    
    % Distance to decision boundary
    distances_to_boundary(s) = abs(double(output_orig) - 0.5);
    
    % Apply uniform perturbation
    X_perturbed = X * (1 + epsilon);
    [output_perturbed, ~, ~, ~, ~, ~, ~, ~, ~] = forward(X_perturbed, A, W1, b1, W2, b2, W3, b3, Wlin, blin, 0.0);
    pred_perturbed = double(output_perturbed > 0.5);
    
    % Record output change and decision change
    output_deltas(s) = abs(double(output_perturbed) - double(output_orig));
    decision_changes(s) = (pred_perturbed ~= pred_orig);
end

% Create visualization
figure;
scatter(distances_to_boundary, output_deltas, 50, decision_changes, 'filled');
xlabel('Distance to Decision Boundary');
ylabel('Output Change with 10% Perturbation');
title('Robustness Analysis Across Multiple Samples');
colormap([0 0.4470 0.7410; 0.8500 0.3250 0.0980]);  % Blue for unchanged, red for changed
colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'Stable', 'Changed'});
grid on;
saveas(gcf, 'gcn_multi_sample_robustness.png');

% Summary statistics
robust_samples = sum(~decision_changes);
vulnerable_samples = sum(decision_changes);
fprintf('\nRobustness Summary (%.0f%% perturbation):\n', epsilon*100);
fprintf('Robust samples: %d (%.1f%%)\n', robust_samples, robust_samples/num_test_samples*100);
fprintf('Vulnerable samples: %d (%.1f%%)\n', vulnerable_samples, vulnerable_samples/num_test_samples*100);
fprintf('Average output change: %.4f\n', mean(output_deltas));
fprintf('Average distance to decision boundary: %.4f\n', mean(distances_to_boundary));

%% Save analysis results
results = struct();
results.feature_sensitivity = sorted_sensitivity;
results.top_sensitive_features = top_features;
results.decision_points = decision_points;
results.flip_percentage = flip_percentage;
results.decision_change_pct = decision_change_pct;
results.multi_sample_robustness = robust_samples/num_test_samples;

save('gcn_perturbation_results.mat', 'results');
fprintf('\nPerturbation analysis completed. Results saved to gcn_perturbation_results.mat\n');

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
    [X3_drop, mask3] = dropout(X3, dropout_prob);

    % Pooling + final linear
    X3_pool = mean(X3_drop, 1);
    linear_output = X3_pool * Wlin + blin;
    output = sigmoid(linear_output);
end