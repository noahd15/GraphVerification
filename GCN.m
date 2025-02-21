useGPU = true;
if useGPU && gpuDeviceCount > 0
    gpuDevice(1);
    disp("Using GPU device");
else
    disp("Using CPU");
end




% Convert the PyTorch tensor to a MATLAB array
data = load('dataset.mat');

disp(data)

% -------------------------------------------------------
% Parameters for a 3-layer GCN
hiddenSize = 64;
W1 = randn(10,hiddenSize);
b1 = zeros(1,hiddenSize);
W2 = randn(hiddenSize,hiddenSize);
b2 = zeros(1,hiddenSize);
W3 = randn(hiddenSize,hiddenSize);
b3 = zeros(1,hiddenSize);
Wlin = randn(hiddenSize,1);
blin = 0;

function Y = relu(X)
    Y = max(X, 0);
end

function X_out = dropout(X, p)
    if p < 0 || p >= 1
        error("Dropout probability must be in [0, 1).");
    end
    mask = rand(size(X)) >= p;
    X_out = (X .* mask) / (1 - p);
end

% -------------------------------------------------------
% Forward pass for GCN
function X_out = graphConv(X_in, A, W, b)
    % Add self-loops to the adjacency matrix

    A_hat = A + eye(size(A));
    
    
    % Calculate degree matrix
    D = diag(sum(A_hat,2));
    
    % Compute D^(-1/2)
    D_inv_sqrt = diag(1 ./ sqrt(diag(D)));
    
    % Compute the normalized adjacency matrix: D^(-1/2) * A_hat * D^(-1/2)
    A_norm = D_inv_sqrt * A_hat * D_inv_sqrt;
    
    % Graph convolution operation: normalize, linear transform, add bias, then non-linearity
    X_out = A_norm * (X_in * W) + b;

end

function output = forward(X, A)
    % Forward pass through the
    global W1 W2 W3 b1 b2 b3 Wlin blin;
    % Compute three convolution layers
    X1 = graphConv(X, A, W1, b1);
    X1 = relu(X1);
    X2 = graphConv(X1, A, W2, b2);
    X2 = relu(X2);
    X3 = graphConv(X2, A, W3, b3);

    % Global mean pool (conceptual: average all node embeddings)
    X_pooled = mean(X3,1);

    % Dropout layer
    X_pooled = dropout(X_pooled, 0.5);

    % Final linear layer
    output = X_pooled * Wlin + blin;
    
    disp("Forward pass complete.");

end


function grads = backward(X, A, y)
    % Declare parameters as global so they can be accessed here.
    global W1 W2 W3 b1 b2 b3 Wlin blin;
    
    % Compute normalized adjacency matrix once for all graph conv layers.
    A_norm = compute_A_norm(A);
    
    %% Forward pass (cache intermediate values for backpropagation)
    % Layer 1
    U1 = X * W1;
    Z1 = A_norm * U1 + repmat(b1, size(X,1), 1);
    A1 = relu(Z1);
    
    % Layer 2
    U2 = A1 * W2;
    Z2 = A_norm * U2 + repmat(b2, size(A1,1), 1);
    A2 = relu(Z2);
    
    % Layer 3 (no activation after graph conv)
    U3 = A2 * W3;
    Z3 = A_norm * U3 + repmat(b3, size(A2,1), 1);
    
    % Global mean pooling (average over nodes)
    X_pool = mean(Z3, 1);
    
    % Dropout (forward with mask caching)
    [X_dp, mask] = dropout_forward(X_pool, 0.5);
    
    % Final linear layer
    output = X_dp * Wlin + blin;
    
    % Compute loss (assuming squared error loss)
    loss = 0.5 * (output - y)^2;
    
    %% Backward pass
    % Derivative of loss w.r.t. final output
    doutput = output - y;
    
    % Gradients for final linear layer
    dWlin = X_dp' * doutput;
    dblin = doutput;
    dX_dp = doutput * Wlin';
    
    % Backprop through dropout
    dX_pool = dropout_backward(dX_dp, mask, 0.5);
    
    % Backprop through global mean pooling:
    % Since X_pool = mean(Z3,1), each node receives an equal share.
    N = size(Z3, 1);
    dZ3 = repmat(dX_pool / N, N, 1);
    
    % Layer 3 backward: Z3 = A_norm*U3 + b3
    % Propagate gradient through A_norm multiplication.
    dU3 = A_norm * dZ3;
    dW3 = A2' * dU3;
    db3 = sum(dZ3, 1);
    dA2 = dU3 * W3';
    
    % Layer 2 backward:
    % ReLU backward on Z2
    dZ2 = dA2 .* (Z2 > 0);
    dU2 = A_norm * dZ2;
    dW2 = A1' * dU2;
    db2 = sum(dZ2, 1);
    dA1 = dU2 * W2';
    
    % Layer 1 backward:
    % ReLU backward on Z1
    dZ1 = dA1 .* (Z1 > 0);
    dU1 = A_norm * dZ1;
    dW1 = X' * dU1;
    db1 = sum(dZ1, 1);
    
    % Pack all gradients into a structure.
    grads = struct('W1', dW1, 'b1', db1, ...
                   'W2', dW2, 'b2', db2, ...
                   'W3', dW3, 'b3', db3, ...
                   'Wlin', dWlin, 'blin', dblin, ...
                   'loss', loss);
end

function A_norm = compute_A_norm(A)
    % Compute the normalized adjacency matrix.
    A_hat = A + eye(size(A));
    D = diag(sum(A_hat, 2));
    D_inv_sqrt = diag(1 ./ sqrt(diag(D)));
    A_norm = D_inv_sqrt * A_hat * D_inv_sqrt;
end

function [X_out, mask] = dropout_forward(X, p)
    % Forward pass for dropout; returns the output and dropout mask.
    mask = rand(size(X)) >= p;
    X_out = (X .* mask) / (1 - p);
end

function dX = dropout_backward(dX_out, mask, p)
    % Backward pass for dropout using the same mask as in the forward pass.
    dX = (dX_out .* mask) / (1 - p);
end

% GCN2 = forward(data.X, data.A);
% disp("Forward pass complete.");
% % -------------------------------------------------------
% % Backward pass for GCN
% grads = backward(data.X, data.A, data.y);
% disp("Backward pass complete.");
% % -------------------------------------------------------
% % Gradient descent update
% learning_rate = 0.01;
% W1 = W1 - learning_rate * grads.W1;
% b1 = b1 - learning_rate * grads.b1;
% W2 = W2 - learning_rate * grads.W2;
% b2 = b2 - learning_rate * grads.b2;
% W3 = W3 - learning_rate * grads.W3;
% b3 = b3 - learning_rate * grads.b3;
% Wlin = Wlin - learning_rate * grads.Wlin;
% blin = blin - learning_rate * grads.blin;
% disp("Gradient descent update complete.");
% % -------------------------------------------------------
% % Save the updated parameters
% save('updated_parameters.mat', 'W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'Wlin', 'blin');
% disp("Updated parameters saved.");
% % -------------------------------------------------------
% % Test the model
% test_output = forward(data.X_test, data.A_test);
% disp("Test output computed.");
% % -------------------------------------------------------
% % Save the test output
% save('test_output.mat', 'test_output');
% disp("Test output saved.");
% % -------------------------------------------------------
% % End of script