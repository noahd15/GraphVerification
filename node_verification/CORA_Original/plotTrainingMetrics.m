function plotTrainingMetrics(train_losses, val_losses, ...
    train_accs, val_accs, ...
    train_precision, train_recall, train_f1, ...
    val_precision, val_recall, val_f1, ...
    validationFrequency, seed, trueLabels, predLabels, ...
    testAcc, testPrecision, testRecall, testF1)

% Get project root for saving plots
projectRoot = getenv('AV_PROJECT_HOME');
if isempty(projectRoot)
    projectRoot = pwd;
end
resultsDir = fullfile(projectRoot, 'node_verification/CORA_Original/train_results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

% Number of epochs
numEpochs = length(train_losses);
epochs = 1:numEpochs;

oldFigVis = get(0,'DefaultFigureVisible');
set(0,'DefaultFigureVisible','off');

%% Loss curve
gcfLoss = figure('Position', [100, 100, 800, 600]); hold on;
plot(epochs, train_losses, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Loss');
plot(epochs, val_losses,   '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Loss');
xlabel('Epoch', 'FontSize', 12);
ylabel('Loss', 'FontSize', 12);
title('Training & Validation Loss', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
lossFile = fullfile(resultsDir, sprintf('loss_curves_seed_%d.png', seed));
saveas(gcfLoss, lossFile);

%% Accuracy curve
gcfAcc = figure('Position', [100, 100, 800, 600]); hold on;
plot(epochs, train_accs, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Accuracy');
plot(epochs, val_accs,   '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Accuracy');
xlabel('Epoch', 'FontSize', 12);
ylabel('Accuracy', 'FontSize', 12);
title('Training & Validation Accuracy', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
accFile = fullfile(resultsDir, sprintf('accuracy_curves_seed_%d.png', seed));
saveas(gcfAcc, accFile);

%% Precision curve
gcfPrec = figure('Position', [100, 100, 800, 600]); hold on;
plot(epochs, train_precision, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Precision');
plot(epochs, val_precision,   '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Precision');
xlabel('Epoch', 'FontSize', 12);
ylabel('Precision', 'FontSize', 12);
title('Training & Validation Precision', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
precFile = fullfile(resultsDir, sprintf('precision_curves_seed_%d.png', seed));
saveas(gcfPrec, precFile);

%% Recall curve
gcfRec = figure('Position', [100, 100, 800, 600]); hold on;
plot(epochs, train_recall, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Recall');
plot(epochs, val_recall,   '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Recall');
xlabel('Epoch', 'FontSize', 12);
ylabel('Recall', 'FontSize', 12);
title('Training & Validation Recall', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
recFile = fullfile(resultsDir, sprintf('recall_curves_seed_%d.png', seed));
saveas(gcfRec, recFile);

%% F1 Score curve
gcfF1 = figure('Position', [100, 100, 800, 600]); hold on;
plot(epochs, train_f1, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training F1 Score');
plot(epochs, val_f1,   '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation F1 Score');
xlabel('Epoch', 'FontSize', 12);
ylabel('F1 Score', 'FontSize', 12);
title('Training & Validation F1 Score', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best', 'FontSize', 10);
grid on;
f1File = fullfile(resultsDir, sprintf('f1_curves_seed_%d.png', seed));
saveas(gcfF1, f1File);

%% Final Metrics Comparison
gcfFinal = figure('Position', [100, 100, 800, 600]);
metrics = {'Accuracy', 'Precision', 'Recall', 'F1 Score'};
trainValues = [train_accs(end), train_precision(end), train_recall(end), train_f1(end)];
valValues   = [val_accs(end),   val_precision(end),   val_recall(end),   val_f1(end)];
testValues  = [testAcc,         testPrecision,        testRecall,        testF1];

bar([trainValues; valValues; testValues]');
legend('Training', 'Validation', 'Test', 'Location', 'best');
set(gca, 'XTick', 1:length(metrics), 'XTickLabel', metrics);
ylabel('Value', 'FontSize', 12);
title('Final Metrics Comparison', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
finalFile = fullfile(resultsDir, sprintf('final_metrics_seed_%d.png', seed));
saveas(gcfFinal, finalFile);

%% Confusion Matrix
gcfCM = figure('Position', [100, 100, 800, 600]);
cm = confusionchart(trueLabels, predLabels, ...
    'ColumnSummary', 'column-normalized', ...
    'RowSummary', 'row-normalized');
title(sprintf('Test Confusion Matrix (seed %d)', seed));
cmFile = fullfile(resultsDir, sprintf('confusion_matrix_seed_%d.png', seed));
saveas(gcfCM, cmFile);

set(0,'DefaultFigureVisible',oldFigVis);


end
