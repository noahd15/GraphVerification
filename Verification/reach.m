% filepath: c:\Users\Noah\OneDrive - Vanderbilt\Spring 2025\CS 6315\Project\AV_Project\Verification\reach.m
%% Load dataset and model parameters
% Load converted_dataset.mat directly
addpath(genpath('/Users/Noah/nnv'))

data = load('./converted_dataset.mat');

% Load training file to get trained weights
run('./TrainingFiles/training_added_graphs.m');

% Select which graph to analyze (for example, the first one)
graph_index = 1;

% Extract adjacency matrix
edge_index = data.edge_indices{graph_index};
% Convert edge_index to adjacency matrix if needed
if size(edge_index, 1) == 2  % If in COO format (source, target)
    num_nodes = max(max(edge_index));
    A = zeros(num_nodes, num_nodes);
    for i = 1:size(edge_index, 2)
        A(edge_index(1,i), edge_index(2,i)) = 1;
    end
else
    A = edge_index;  % Already in adjacency matrix format
end

% Extract node features
X = data.features{graph_index};

% Extract model weights from training
% Note: W1, b1, W2, b2, W3, b3, Wlin, blin should be available 
% from running training_added_graphs.m

% Convert dlarray to double if needed
if isa(W1, 'dlarray')
    W1 = extractdata(W1);
    W2 = extractdata(W2);
    W3 = extractdata(W3);
    Wlin = extractdata(Wlin);
end

% Move from GPU to CPU if needed
if exist('useGPU', 'var') && useGPU
    W1 = gather(W1);
    b1 = gather(b1);
    W2 = gather(W2);
    b2 = gather(b2);
    W3 = gather(W3);
    b3 = gather(b3);
    Wlin = gather(Wlin);
    blin = gather(blin);
    X = gather(X);
    A = gather(A);
end

% Transpose weights if needed (NNV expects weights in different format)
if size(W1, 1) ~= size(X, 2)
    W1 = W1';
    W2 = W2';
    W3 = W3';
    Wlin = Wlin';
end

% Reshape biases if needed
if size(b1, 1) == 1
    b1 = b1';
    b2 = b2';
    b3 = b3';
    blin = reshape(blin, [], 1);
end

% Display extracted components
fprintf('Extracted GNN components for reachability analysis:\n');
fprintf('Adjacency matrix dimensions: %d x %d\n', size(A, 1), size(A, 2));
fprintf('Node features dimensions: %d x %d\n', size(X, 1), size(X, 2));
fprintf('Weights dimensions:\n');
fprintf('  W1: %d x %d\n', size(W1, 1), size(W1, 2));
fprintf('  W2: %d x %d\n', size(W2, 1), size(W2, 2));
fprintf('  W3: %d x %d\n', size(W3, 1), size(W3, 2));
fprintf('  Wlin: %d x %d\n', size(Wlin, 1), size(Wlin, 2));
fprintf('Biases dimensions:\n');
fprintf('  b1: %d x %d\n', size(b1, 1), size(b1, 2));
fprintf('  b2: %d x %d\n', size(b2, 1), size(b2, 2));
fprintf('  b3: %d x %d\n', size(b3, 1), size(b3, 2));
fprintf('  blin: %d x %d\n', size(blin, 1), size(blin, 2));




% Compute the normalized adjacency matrix (same as in your GCN)
function A_norm = computeA_norm(A)
    % Add self-loops to adjacency matrix
    A_ = A + eye(size(A,1)); 
    % Compute degree matrix
    D = diag(sum(A_,1));
    % Compute D^(-1/2)
    D_inv_half = D^(-1/2);
    % Normalize adjacency matrix
    A_norm = D_inv_half * A_ * D_inv_half;
end



% Precompute the normalized adjacency matrix
A_norm = computeA_norm(A);


% Initialize the layers
L = ReluLayer();



IS = ImageStar(X,X); % create input set as a Star set
fprintf('Layer 1:\n');
% Layer 1
IS = IS.affineMap(A_norm,[]);
IS = IS.affineMap(W1,[]);
R1 = L.reach(IS,'exact-star');

R1.getRanges();

% fprintf('Output range: [%f, %f]\n', R1.getRanges(), R1.getRanges());

  