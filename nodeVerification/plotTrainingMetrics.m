function plotTrainingMetrics(train_losses, val_losses, test_losses, ...
    train_accs, val_accs, test_accs, ...
    train_precision, train_recall, train_f1, ...
    val_precision, val_recall, val_f1, ...
    test_precision, test_recall, test_f1, ...
    validationFrequency)
% PLOTTRAININGMETRICS Plot training, validation, and test metrics
%   This function creates plots for all metrics (loss, accuracy, precision, recall, F1)
%   showing the training, validation, and test values over epochs.
%
%   Inputs:
%     train_losses, val_losses, test_losses - Loss values for each epoch
%     train_accs, val_accs, test_accs - Accuracy values for each epoch
%     train_precision, val_precision, test_precision - Precision values for each epoch
%     train_recall, val_recall, test_recall - Recall values for each epoch
%     train_f1, val_f1, test_f1 - F1 score values for each epoch
%     validationFrequency - Frequency of validation (for x-axis ticks)
%
%   Note: Any input can be empty ([]) if that metric is not available

% Get project root for saving plots
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    projectRoot = pwd;
end
resultsDir = fullfile(projectRoot, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end
logsDir = fullfile(projectRoot, 'logs');
if ~exist(logsDir, 'dir')
    mkdir(logsDir);
end

% Number of epochs
numEpochs = length(train_losses);
epochs = 1:numEpochs;

% Create figure for Loss
figure('Position', [100, 100, 800, 600]);
hold on;
if ~isempty(train_losses)
    plot(epochs, train_losses, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Loss');
end
if ~isempty(val_losses)
    plot(epochs, val_losses, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Loss');
end
if ~isempty(test_losses) && any(~isnan(test_losses))
    plot(epochs, test_losses, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test Loss');
end
xlabel('Epoch', 'FontSize', 12);
ylabel('Loss', 'FontSize', 12);
title('Training, Validation, and Test Loss', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
saveas(gcf, fullfile(resultsDir, 'loss_curves.png'));
saveas(gcf, fullfile(logsDir, 'loss_curves.png'));

% Create figure for Accuracy
figure('Position', [100, 100, 800, 600]);
hold on;
if ~isempty(train_accs)
    plot(epochs, train_accs, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Accuracy');
end
if ~isempty(val_accs)
    plot(epochs, val_accs, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Accuracy');
end
% Test accuracy is only shown in the final bar plot, not in the line charts
% if ~isempty(test_accs) && any(~isnan(test_accs))
%     plot(epochs, test_accs, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test Accuracy');
% end
xlabel('Epoch', 'FontSize', 12);
ylabel('Accuracy', 'FontSize', 12);
title('Training and Validation Accuracy', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
saveas(gcf, fullfile(resultsDir, 'accuracy_curves.png'));
saveas(gcf, fullfile(logsDir, 'accuracy_curves.png'));

% Create figure for Precision
figure('Position', [100, 100, 800, 600]);
hold on;
if ~isempty(train_precision)
    plot(epochs, train_precision, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Precision');
end
if ~isempty(val_precision)
    plot(epochs, val_precision, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Precision');
end
% Test precision is only shown in the final bar plot, not in the line charts
% if ~isempty(test_precision) && any(~isnan(test_precision))
%     plot(epochs, test_precision, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test Precision');
% end
xlabel('Epoch', 'FontSize', 12);
ylabel('Precision', 'FontSize', 12);
title('Training and Validation Precision', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
saveas(gcf, fullfile(resultsDir, 'precision_curves.png'));
saveas(gcf, fullfile(logsDir, 'precision_curves.png'));

% Create figure for Recall
figure('Position', [100, 100, 800, 600]);
hold on;
if ~isempty(train_recall)
    plot(epochs, train_recall, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Recall');
end
if ~isempty(val_recall)
    plot(epochs, val_recall, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Recall');
end
% Test recall is only shown in the final bar plot, not in the line charts
% if ~isempty(test_recall) && any(~isnan(test_recall))
%     plot(epochs, test_recall, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test Recall');
% end
xlabel('Epoch', 'FontSize', 12);
ylabel('Recall', 'FontSize', 12);
title('Training and Validation Recall', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
saveas(gcf, fullfile(resultsDir, 'recall_curves.png'));
saveas(gcf, fullfile(logsDir, 'recall_curves.png'));

% Create figure for F1 Score
figure('Position', [100, 100, 800, 600]);
hold on;
if ~isempty(train_f1)
    plot(epochs, train_f1, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training F1 Score');
end
if ~isempty(val_f1)
    plot(epochs, val_f1, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation F1 Score');
end
% Test F1 score is only shown in the final bar plot, not in the line charts
% if ~isempty(test_f1) && any(~isnan(test_f1))
%     plot(epochs, test_f1, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test F1 Score');
% end
xlabel('Epoch', 'FontSize', 12);
ylabel('F1 Score', 'FontSize', 12);
title('Training and Validation F1 Score', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
saveas(gcf, fullfile(resultsDir, 'f1_curves.png'));
saveas(gcf, fullfile(logsDir, 'f1_curves.png'));

% Create a combined metrics figure for final epoch
figure('Position', [100, 100, 800, 600]);
metrics = {'Accuracy', 'Precision', 'Recall', 'F1 Score'};
trainValues = [train_accs(end), train_precision(end), train_recall(end), train_f1(end)];
valValues = [val_accs(end), val_precision(end), val_recall(end), val_f1(end)];

% For test values, we need to handle the case where we have a single test value
% rather than a vector of values over epochs
testValues = [];

% Check if we have test metrics as vectors (from multiple epochs)
if ~isempty(test_accs) && ~isempty(test_precision) && ~isempty(test_recall) && ~isempty(test_f1)
    if ~isnan(test_accs) && ~isnan(test_precision) && ~isnan(test_recall) && ~isnan(test_f1)
        if numel(test_accs) > 1
            testValues = [test_accs(end), test_precision(end), test_recall(end), test_f1(end)];
        else
            testValues = [test_accs, test_precision, test_recall, test_f1];
        end
    end
else
    % If we don't have test metrics as vectors, check if we have single test values
    % that might have been passed directly
    testAcc = NaN;
    testPrec = NaN;
    testRec = NaN;
    testF1 = NaN;

    % Try to get these values from the caller's workspace
    try
        evalin('caller', 'testAcc;');
        testAcc = evalin('caller', 'testAcc');
    catch
        % Variable doesn't exist, keep as NaN
    end

    try
        evalin('caller', 'precision(end);');
        testPrec = evalin('caller', 'precision(end)');
    catch
        % Variable doesn't exist, keep as NaN
    end

    try
        evalin('caller', 'recall(end);');
        testRec = evalin('caller', 'recall(end)');
    catch
        % Variable doesn't exist, keep as NaN
    end

    try
        evalin('caller', 'f1(end);');
        testF1 = evalin('caller', 'f1(end)');
    catch
        % Variable doesn't exist, keep as NaN
    end

    % If we have valid values, use them
    if ~isnan(testAcc) && ~isnan(testPrec) && ~isnan(testRec) && ~isnan(testF1)
        testValues = [testAcc, testPrec, testRec, testF1];
    end
end

% Create bar chart
if isempty(testValues)
    bar([trainValues; valValues]');
    legend('Training', 'Validation', 'Location', 'best');
else
    bar([trainValues; valValues; testValues]');
    legend('Training', 'Validation', 'Test', 'Location', 'best');
end
set(gca, 'XTick', 1:length(metrics), 'XTickLabel', metrics);
ylabel('Value', 'FontSize', 12);
title('Final Metrics Comparison', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
saveas(gcf, fullfile(resultsDir, 'final_metrics_comparison.png'));
saveas(gcf, fullfile(logsDir, 'final_metrics_comparison.png'));

end
