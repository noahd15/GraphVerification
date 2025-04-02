function reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest)
    % Verification of a Graph Neural Network
    
    %% Load parameters of gcn
    data = load(modelPath);
    disp(data); % Check the contents of the loaded file
    
    % Ensure required variables are loaded
    if isfield(data, 'muX') && isfield(data, 'sigsqX') && isfield(data, 'parameters')
        muX = data.muX;
        sigsqX = data.sigsqX;
        parameters = data.parameters;
    else
        error("The loaded file does not contain 'muX', 'sigsqX', or 'parameters'.");
    end
    
    w1 = gather(parameters.mult1.Weights); % Ensure this field exists
    w2 = gather(parameters.mult2.Weights);
    w3 = gather(parameters.mult3.Weights);
    
    % model function
    %     ANorm => adjacency matrix of A
    %     Z1 => input
    % 
    %     Z2 = ANorm * Z1 * w1;
    %     Z2 = relu(Z2) + Z1; (layer 1)
    % 
    %     Z3 = ANorm * Z2 * w2;
    %     Z3 = relu(Z3) + Z2; (layer 2)
    % 
    %     Z4 = ANorm * Z3 * w3;
    %     Y = softmax(Z4,DataFormat="BC"); (output layer)

    
    %% Start for loop for verification here, preprocess one molecule at a time
    
    N = size(featureDataTest, 3);
    
    % L_inf size
    % epsilon = [0.005; 0.01; 0.02; 0.05];
    
    % Store resuts
    targets = {};
    outputSets = {};
    rT = {};
    
    for k = 1:length(epsilon)
    
        for i = 1:N

            % Get molecule data
            [ATest,XTest,labelsTest] = preprocessData(adjacencyDataTest(:,:,N),featureDataTest(:,:,N),labelDataTest(N,:));
            
            % normalize data
            XTest = (XTest - muX)./sqrt(sigsqX);
            XTest = dlarray(XTest);
                    
            % adjacency matrix represent connections, so keep it as is
            Averify = normalizeAdjacency(ATest);
            
            % Get input set: input values for each node is X
            lb = extractdata(XTest-epsilon(k));
            ub = extractdata(XTest+epsilon(k));
            Xverify = ImageStar(lb,ub);
            
            % Compute reachability
            t = tic;
            
            reachMethod = 'approx-star';
            L = ReluLayer(); % Create relu layer;
            
            Y = computeReachability({w1,w2,w3}, L, reachMethod, Xverify, Averify);

            % store results
            outputSets{i} = Y;
            targets{i} = labelsTest;
            rT{i} = toc(t);
        
        end
        
        % Save verification results
        save("results/verified_nodes_"+modelPath+"_eps"+string(epsilon(k))+".mat", "outputSets", "targets", "rT");
    
    end

end

%% Helper functions

function [adjacency,features,labels] = preprocessData(adjacencyData,featureData,labelData)

    [adjacency, features] = preprocessPredictors(adjacencyData,featureData);
    labels = [];
    
    % Convert labels to categorical.
    for i = 1:size(adjacencyData,3)
        % Extract and append unpadded data.
        T = nonzeros(labelData(i,:));
        labels = [labels; T];
    end
    
    labels2 = nonzeros(labelData);
    assert(isequal(labels2,labels2))
    
    atomicNumbers = unique(labels);
    atomNames =  atomicSymbol(atomicNumbers);
    labels = categorical(labels, atomicNumbers, atomNames);

end

function [adjacency,features] = preprocessPredictors(adjacencyData,featureData)

    adjacency = sparse([]);
    features = [];
    
    for i = 1:size(adjacencyData, 3)
        % Extract unpadded data.
        numNodes = find(any(adjacencyData(:,:,i)),1,"last");
    
        A = adjacencyData(1:numNodes,1:numNodes,i);
        X = featureData(1:numNodes,1:numNodes,i);
    
        % Extract feature vector from diagonal of feature matrix.
        X = diag(X);
    
        % Append extracted data.
        adjacency = blkdiag(adjacency,A);
        features = [features; X];
    end

end

function ANorm = normalizeAdjacency(A)

    % Add self connections to adjacency matrix.
    A = A + speye(size(A));
    
    % Compute inverse square root of degree.
    degree = sum(A, 2);
    degreeInvSqrt = sparse(sqrt(1./degree));
    
    % Normalize adjacency matrix.
    ANorm = diag(degreeInvSqrt) * A * diag(degreeInvSqrt);

    % Convert to a full matrix to avoid the sparse-input error:
    ANorm = full(ANorm);

end

function Y = computeReachability(weights, L, reachMethod, input, adjMat)
    % weights = weights of GNN ({w1, w2, w3}
    % L = Layer type (ReLU)
    % reachMethod = reachability method for all layers('approx-star is default)
    % input = pertubed input features (ImageStar)
    % adjMat = adjacency matric of corresonding input features
    % Y = computed output of GNN (ImageStar)

    Xverify = input;
    Averify = adjMat;
    n = size(adjMat,1);
    
    %%%%%%%%  LAYER 1  %%%%%%%%
    
    % part 1
    newV = Xverify.V;
    
    % Debug the size of newV and expected dimensions
    disp("Size of newV before reshape: " + mat2str(size(newV)));
    disp("Expected size: [" + n + ", " + (n+1) + "]");

    % Ensure the number of elements matches
    if numel(newV) ~= n * (n+1)
        error("Dimension mismatch: Cannot reshape newV to [" + n + ", " + (n+1) + "]");
    end

    % Reshape newV
    newV = reshape(newV, [n, n+1]);
    
    newV = Averify * newV;
    newV = tensorprod(newV, extractdata(weights{1}));
    newV = permute(newV, [1 4 3 2]);
    X2 = ImageStar(newV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    
    % part 2
    X2b = L.reach(X2, reachMethod);
    repV = repmat(Xverify.V,[1,32,1,1]);
    Xrep = ImageStar(repV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    X2b_ = X2b.MinkowskiSum(Xrep);
    
    %%%%%%%%  LAYER 2  %%%%%%%%
    
    % part 1
    newV = X2b_.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{2}),2,1);
    newV = permute(newV, [1 4 2 3]);
    X3 = ImageStar(newV, X2b_.C, X2b_.d, X2b_.pred_lb, X2b_.pred_ub);
    % part 2
    X3b = L.reach(X3, reachMethod); 
    X3b_ = X3b.MinkowskiSum(X2b_);
    
    %%%%%%%%  LAYER 3  %%%%%%%%
    
    newV = X3b_.V;
    newV = tensorprod(full(Averify), newV, 2, 1);
    newV = tensorprod(newV, extractdata(weights{3}), 2, 1);
    newV = permute(newV, [1 4 2 3]);
    Y = ImageStar(newV, X3b_.C, X3b_.d, X3b_.pred_lb, X3b_.pred_ub);

end


