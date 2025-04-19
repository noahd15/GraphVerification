function PCA(adjacency, features, labels, numComponents, outFile)
% Reduces features using PCA and saves new dataset (adjacency, reduced features, labels) to a .mat file.
%   PCA_reduce_and_save(adjacency, features, labels, numComponents, outFile)
%   adjacency: (N x N x 1) matrix
%   features: (N x D x 1) or (N x D) matrix
%   labels: (N x 1) or (N x 1 x 1) vector
%   numComponents: number of principal components (default: 16)
%   outFile: output .mat file name (string)

if nargin < 4
    numComponents = 16;
end

if nargin < 5
    outFile = 'reduced_dataset.mat';
end

% Reshape features if needed
if ndims(features) == 3
    X = reshape(features, size(features,1), size(features,2));
else
    X = features;
end

% Center the data
X_mean = mean(X, 1);
X_centered = X - X_mean;

% PCA
[~, score, ~, ~, explained] = pca(X_centered);
features = score(:, 1:numComponents);

disp(['Explained variance by first ', num2str(numComponents), ' components:']);
disp(sum(explained(1:numComponents)));

edge_indices = adjacency; % Keep the original adjacency matrix
% Save new dataset
save(outFile, 'edge_indices', 'features', 'labels');
disp(['Saved reduced dataset to ', outFile]);

end