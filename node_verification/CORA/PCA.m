function PCA(adjacency, features, labels, trainIdx, valIdx, testIdx, numComponents, outFile, varThreshold)
% Performs PCA on training data with optional feature selection by variance.
%
%   adjacency:     (N x N x 1) matrix
%   features:      (N x D x 1) or (N x D) matrix
%   labels:        (N x 1)
%   *_Idx:         train/val/test indices
%   numComponents: # principal components to keep
%   outFile:       output .mat file
%   varThreshold:  optional, scalar – remove features with var < threshold

if nargin < 7
    numComponents = 16;
end
if nargin < 8
    outFile = 'reduced_dataset.mat';
end
if nargin < 9
    varThreshold = 0;  % keep all features by default
end

% Reshape features if needed
if ndims(features) == 3
    X = reshape(features, size(features,1), size(features,2));
else
    X = features;
end

N = size(X,1);
X_train = X(trainIdx, :);
X_val   = X(valIdx, :);
X_test  = X(testIdx, :);

% ----- Feature Selection -----
if varThreshold > 0
    featureVars = var(X_train, 0, 1);  % variance per feature
    keepIdx = featureVars > varThreshold;
    X_train = X_train(:, keepIdx);
    X_val   = X_val(:, keepIdx);
    X_test  = X_test(:, keepIdx);
else
    keepIdx = true(1, size(X_train, 2));  % keep all
end

% ----- PCA -----
X_mean = mean(X_train, 1);
X_train_centered = X_train - X_mean;
[coeff, score_train, ~, ~, explained] = pca(X_train_centered);

P = coeff(:, 1:numComponents);

X_val_centered  = X_val - X_mean;
X_test_centered = X_test - X_mean;

X_train_reduced = score_train(:, 1:numComponents);
X_val_reduced   = X_val_centered * P;
X_test_reduced  = X_test_centered * P;

% ----- Combine -----
features = zeros(N, numComponents);
features(trainIdx, :) = X_train_reduced;
features(valIdx, :)   = X_val_reduced;
features(testIdx, :)  = X_test_reduced;

% Save
edge_indices = adjacency;
save(outFile, 'edge_indices', 'features', 'labels');

disp(['Saved PCA-reduced dataset to ', outFile]);
disp(['Explained variance by first ', num2str(numComponents), ' components:']);
disp(sum(explained(1:numComponents)));
disp(['Number of features retained after variance filtering: ', num2str(sum(keepIdx))]);

end
