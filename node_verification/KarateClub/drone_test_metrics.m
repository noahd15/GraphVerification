% Initialize
seeds = [0, 1, 2];
numSeeds = numel(seeds);

% Preallocate matrices: rows = metrics, columns = seeds
metrics16 = zeros(4, numSeeds);  % Rows: [TestAcc; Precision; Recall; F1]
metrics32 = zeros(4, numSeeds);

% Helper function to safely average metric
get_metric = @(x) mean(x(:));  % works for scalars or vectors

% Load data for each seed
for i = 1:numSeeds
    s = seeds(i);
    data16 = load(sprintf('models/karate_node_gcn_%d_34.mat', s));
    % data32 = load(sprintf('models/drone_node_gcn_pca_%d_32.mat', s));
    
    metrics16(:, i) = [get_metric(data16.testAcc); ...
                       get_metric(data16.testPrec); ...
                       get_metric(data16.testRec); ...
                       get_metric(data16.testF1)];
                   
    % metrics32(:, i) = [get_metric(data32.testAcc); ...
    %                    get_metric(data32.precision); ...
    %                    get_metric(data32.recall); ...
    %                    get_metric(data32.f1)];
end

% Compute mean across seeds
barData = cat(3, metrics16); % [metric, seed, version]
barMeans = mean(barData, 2);            % Mean across seeds
groupedData = squeeze(barMeans);        % 4x2 [metric x version]

% Create bar plot
metricLabels = {'Test Accuracy', 'Precision', 'Recall', 'F1 Score'};
fig = figure('Visible', 'off');
bar(groupedData);
set(gca, 'XTickLabel', metricLabels, 'FontSize', 12);
% legend({'16 Features', '32 Features'}, 'Location', 'northwest');
ylabel('Score');
title('Drone GCN Test Metrics Comparison (Average over Seeds)', 'FontSize', 14);
grid on;

% Save figure
saveas(fig, 'model_test_metrics_comparison.png');
