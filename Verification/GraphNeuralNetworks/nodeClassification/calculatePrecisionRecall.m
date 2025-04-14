function [precision, recall, f1] = calculatePrecisionRecall(predictions, targets)
    % Calculate precision, recall, and F1 score for multi-class classification
    % 
    % Inputs:
    %   predictions - categorical array of predicted classes
    %   targets     - categorical array of true classes
    %
    % Outputs:
    %   precision   - precision for each class and macro average
    %   recall      - recall for each class and macro average
    %   f1          - F1 score for each class and macro average
    
    % Get unique classes
    classes = categories(targets);
    numClasses = length(classes);
    
    % Initialize metrics
    precision = zeros(numClasses + 1, 1);
    recall = zeros(numClasses + 1, 1);
    f1 = zeros(numClasses + 1, 1);
    
    % Calculate metrics for each class
    for i = 1:numClasses
        % True positives: predictions that correctly identified the class
        truePositives = sum(predictions == classes{i} & targets == classes{i});
        
        % False positives: predictions that incorrectly identified the class
        falsePositives = sum(predictions == classes{i} & targets ~= classes{i});
        
        % False negatives: instances of the class that were missed
        falseNegatives = sum(predictions ~= classes{i} & targets == classes{i});
        
        % Calculate precision (avoid division by zero)
        if (truePositives + falsePositives) > 0
            precision(i) = truePositives / (truePositives + falsePositives);
        else
            precision(i) = 0;
        end
        
        % Calculate recall (avoid division by zero)
        if (truePositives + falseNegatives) > 0
            recall(i) = truePositives / (truePositives + falseNegatives);
        else
            recall(i) = 0;
        end
        
        % Calculate F1 score (avoid division by zero)
        if (precision(i) + recall(i)) > 0
            f1(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        else
            f1(i) = 0;
        end
    end
    
    % Calculate macro average (average across all classes)
    precision(end) = mean(precision(1:numClasses));
    recall(end) = mean(recall(1:numClasses));
    f1(end) = mean(f1(1:numClasses));
end
