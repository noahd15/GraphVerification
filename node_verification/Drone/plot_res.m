% Load summary file for dropout
dropout   = load("verification_results\mat_files\summary_all_Linf_dropout_16.mat");

% Number of epsilon values
eN = length(dropout.epsilon);

avgRobust_dropout   = zeros(eN, 1);

for k = 1:eN
    % With Dropout
    robust_do = squeeze(dropout.allAtoms(k, 1, :));
    total_do  = squeeze(dropout.allAtoms(k, 4, :));
    avgRobust_dropout(k) = mean(robust_do ./ total_do);
end

% Plot
figure;
semilogx(dropout.epsilon, avgRobust_dropout, '-s', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('\epsilon', 'FontSize', 12);
ylabel('Average Robustness', 'FontSize', 12);
title('Average Robustness vs Epsilon (16 Features)', 'FontSize', 13);
legend({'With Dropout'}, 'Location', 'southwest');
grid on;
ylim([.2 .4]);

% Save plot
outDir = "verification_results/figures";
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

filename = fullfile(outDir, "avg_robustness_dropout");
saveas(gcf, filename + ".png");
saveas(gcf, filename + ".pdf");

fprintf('Saved plot to:\n- %s.png\n- %s.pdf\n', filename, filename);
