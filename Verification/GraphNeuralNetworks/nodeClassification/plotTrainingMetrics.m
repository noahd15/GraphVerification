function plotTrainingMetrics(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, ...
    train_precision, train_recall, train_f1, val_precision, val_recall, val_f1, ...
    test_precision, test_recall, test_f1, validationFrequency)
    % This function plots training, validation, and test metrics (loss and accuracy)
    % on the same graph

    % Try to get the current directory or use the current directory
    try
        % First try to use the project root if available
        projectRoot = getenv('AV_PROJECT_HOME');
        if isempty(projectRoot)
            % If not set, use the current directory
            projectRoot = pwd;
        end
    catch
        % Fallback to current directory if any error occurs
        projectRoot = pwd;
    end

    % Create results directory path
    resultsDir = fullfile(projectRoot, 'results');

    % Create a directory for results if it doesn't exist
    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end

    % Display where we're saving the plots
    disp(['Saving plots to: ' resultsDir]);

    % Create a figure for losses
    figure('Position', [100, 100, 800, 600]);

    % Plot training loss for all epochs
    plot(1:length(train_losses), train_losses, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Loss');
    hold on;

    % Plot validation loss for all epochs
    plot(1:length(val_losses), val_losses, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Loss');

    % Plot test loss only for epochs where it was calculated
    test_epochs = 1:validationFrequency:length(test_losses);
    test_loss_values = test_losses(test_epochs);
    plot(test_epochs, test_loss_values, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test Loss');

    % Add labels and title
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Loss', 'FontSize', 12, 'FontWeight', 'bold');
    title('Training, Validation, and Test Loss', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;

    % Add explanation text
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
        'String', 'Lower loss values indicate better model performance', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    % Save the figure
    saveas(gcf, fullfile(resultsDir, 'loss_curves.png'));

    % Create a figure for accuracies
    figure('Position', [100, 100, 800, 600]);

    % Plot training accuracy for all epochs
    plot(1:length(train_accs), train_accs, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Accuracy');
    hold on;

    % Plot validation accuracy for all epochs
    plot(1:length(val_accs), val_accs, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Accuracy');

    % Plot test accuracy only for epochs where it was calculated
    test_epochs = 1:validationFrequency:length(test_accs);
    test_acc_values = test_accs(test_epochs);
    plot(test_epochs, test_acc_values, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test Accuracy');

    % Add labels and title
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Accuracy', 'FontSize', 12, 'FontWeight', 'bold');
    title('Training, Validation, and Test Accuracy', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;

    % Add explanation text
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
        'String', 'Higher accuracy values indicate better model performance (range: 0-1)', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    % Save the figure
    saveas(gcf, fullfile(resultsDir, 'accuracy_curves.png'));

    % Create a figure for precision
    figure('Position', [100, 100, 800, 600]);

    % Plot training precision for all epochs
    plot(1:length(train_precision), train_precision, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Precision');
    hold on;

    % Plot validation precision for all epochs
    plot(1:length(val_precision), val_precision, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Precision');

    % Plot test precision only for epochs where it was calculated
    test_epochs = 1:validationFrequency:length(test_precision);
    test_precision_values = test_precision(test_epochs);
    plot(test_epochs, test_precision_values, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test Precision');

    % Add labels and title
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Precision', 'FontSize', 12, 'FontWeight', 'bold');
    title('Training, Validation, and Test Precision', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;

    % Add explanation text
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
        'String', 'Precision = True Positives / (True Positives + False Positives) - Higher is better', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    % Save the figure
    saveas(gcf, fullfile(resultsDir, 'precision_curves.png'));

    % Create a figure for recall
    figure('Position', [100, 100, 800, 600]);

    % Plot training recall for all epochs
    plot(1:length(train_recall), train_recall, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Recall');
    hold on;

    % Plot validation recall for all epochs
    plot(1:length(val_recall), val_recall, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation Recall');

    % Plot test recall only for epochs where it was calculated
    test_epochs = 1:validationFrequency:length(test_recall);
    test_recall_values = test_recall(test_epochs);
    plot(test_epochs, test_recall_values, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test Recall');

    % Add labels and title
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Recall', 'FontSize', 12, 'FontWeight', 'bold');
    title('Training, Validation, and Test Recall', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;

    % Add explanation text
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
        'String', 'Recall = True Positives / (True Positives + False Negatives) - Higher is better', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    % Save the figure
    saveas(gcf, fullfile(resultsDir, 'recall_curves.png'));

    % Create a figure for F1 score
    figure('Position', [100, 100, 800, 600]);

    % Plot training F1 for all epochs
    plot(1:length(train_f1), train_f1, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training F1');
    hold on;

    % Plot validation F1 for all epochs
    plot(1:length(val_f1), val_f1, '-x', 'LineWidth', 1.5, 'DisplayName', 'Validation F1');

    % Plot test F1 only for epochs where it was calculated
    test_epochs = 1:validationFrequency:length(test_f1);
    test_f1_values = test_f1(test_epochs);
    plot(test_epochs, test_f1_values, '-s', 'LineWidth', 1.5, 'DisplayName', 'Test F1');

    % Add labels and title
    xlabel('Epoch', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('F1 Score', 'FontSize', 12, 'FontWeight', 'bold');
    title('Training, Validation, and Test F1 Score', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;

    % Add explanation text
    annotation('textbox', [0.15, 0.01, 0.7, 0.05], ...
        'String', 'F1 Score = 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean of precision and recall', ...
        'EdgeColor', 'none', 'HorizontalAlignment', 'center');

    % Save the figure
    saveas(gcf, fullfile(resultsDir, 'f1_curves.png'));
end
