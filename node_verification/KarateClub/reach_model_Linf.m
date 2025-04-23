function reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest, num_features)
    % Setup

    % fprintf('adjacencyDataTest size: %d x %d\n', size(adjacencyDataTest));
    adjacencyDataTest = reshape(adjacencyDataTest, [size(adjacencyDataTest, 1), size(adjacencyDataTest, 2), 1]);
    featureDataTest   = reshape(featureDataTest, [size(featureDataTest, 1), size(featureDataTest, 2), 1]);
    adjacencyDataTest = double(adjacencyDataTest);
    featureDataTest   = double(featureDataTest);

    load("models/"+modelPath+".mat");

    w1 = gather(parameters.mult1.Weights);

    % w2 = gather(parameters.mult2.Weights);
    % w3 = gather(parameters.mult3.Weights);

    N = size(featureDataTest, 3);
    reachMethod = 'approx-star';
    L = ReluLayer();

    for k = 1:length(epsilon)
        results    = {};
        outputSets = {};
        targets    = {};
        rT         = {};

        for i = 1:N
            [ATest, XTest, labelsTest] = preprocessData(adjacencyDataTest(:,:,i), featureDataTest(:,:,i), labelDataTest(i,:));
            


            Averify = normalizeAdjacency(ATest);
            XTest = dlarray(XTest);


            % Construct input set
            lb = extractdata(XTest - epsilon(k));
            ub = extractdata(XTest + epsilon(k));

            
            Xverify = ImageStar(lb, ub);

            % Reachability
            t = tic;
            Y_all = computeReachability({w1}, L, reachMethod, Xverify, Averify);
            y = Y_all.V;
            % whos y
            elapsedTime = toc(t);

            numNodes = size(Y_all.V, 1);
            % fprintf('Num nodes: %d\n', numNodes);

            for j = 1:numNodes
                matIdx = zeros(1, numNodes); matIdx(j) = 1;
                Y_j = Y_all.affineMap(matIdx, []);

                outputSets{(i-1)*numNodes + j} = Y_j;
                targets{(i-1)*numNodes + j}    = labelDataTest(j);
                rT{(i-1)*numNodes + j}         = elapsedTime;

                % Verify immediately
                results{(i-1)*numNodes + j} = verifyAtom(Y_j, labelDataTest(j));
            end
        end

        % Save everything at once (smaller since verification done)
        save("verification_results/mat_files/verified_nodes_"+modelPath+"_eps_"+string(epsilon(k))+ "_" + string(num_features) + ".mat", ...
    "results", "rT", "targets", "-v7.3");
        disp("DONE: "+modelPath+", eps="+epsilon(k));
    end
end


function [adjacency, features, labels] = preprocessData(adjacencyData, featureData, labelData)
    [adjacency, features] = preprocessPredictors(adjacencyData, featureData);
    labels = labelData(:);
end

function [adjacency, features] = preprocessPredictors(adjacencyData, featureData)
    adjacency = sparse([]);
    features = [];

    for i = 1:size(adjacencyData, 3)
        % numNodes = find(any(adjacencyData(:,:,i)), 1, "last");
        % fprintf('Num nodes: %d\n', numNodes);
        % if isempty(numNodes) || numNodes==0
        %     continue
        % end
        numNodes = size(adjacencyData, 1);

        A = adjacencyData(1:numNodes, 1:numNodes, i);
        X = featureData(1:numNodes, :, i);

        adjacency = blkdiag(adjacency, A);

        % Concatenate feature rows
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

end

function Y = computeReachability(weights, L, reachMethod, input, adjMat)
    Xverify = input;
    Averify = adjMat; 
    n = size(adjMat,1); 

    %%%%%%%%  LAYER 1  %%%%%%%%
    newV = Xverify.V; %272x16x1x4353
    newV = squeeze(Xverify.V); %272x16x4353  
    Averify_full = full(Averify);
    newV = tensorprod(Averify_full, newV, 2, 1); %272x16x4353 
    w = extractdata(weights{1}); %32x64 
    newV = tensorprod(newV, extractdata(weights{1}), 2, 1);
    newV = reshape(newV, [size(newV,1), size(newV,2), 1, size(newV,3)]); 
    newV = permute(newV, [1 4 3 2]); 
    
    X2 = ImageStar(newV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub); 
    % part 2 %
    X2b = L.reach(X2, reachMethod);
    repV = repmat(Xverify.V,[1,1,1,1]); 
    Xrep = ImageStar(repV, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    % X2b_ = X2b.MinkowskiSum(Xrep);

    %%%%%%%% COMMENTED OUT MULTI-LAYER CODE %%%%%%%%
    % %%%%%%%%  LAYER 2  %%%%%%%%
    % newV = X2b_.V;
    % newV = tensorprod(full(Averify), newV, 2, 1);
    % newV = tensorprod(newV, extractdata(weights{2}),2,1);
    % newV = permute(newV, [1 4 2 3]);
    % X3 = ImageStar(newV, X2b_.C, X2b_.d, X2b_.pred_lb, X2b_.pred_ub);
    % X3b = L.reach(X3, reachMethod);
    % X3b_ = X3b.MinkowskiSum(X2b_);

    % %%%%%%%%  LAYER 3  %%%%%%%%
    % newV = X3b_.V;
    % newV = tensorprod(full(Averify), newV, 2, 1);
    % newV = tensorprod(newV, extractdata(weights{3}), 2, 1);
    % newV = permute(newV, [1 4 2 3]);
    % Y = ImageStar(newV, X3b_.C, X3b_.d, X3b_.pred_lb, X3b_.pred_ub);

    %%%%%%%% SINGLE-LAYER OUTPUT %%%%%%%%
    Y = Xrep;
end

function result = verifyAtom(X, target)
    % X is a 7D Star set (output scores for one node)
    outDim = size(X.V, 2);  % Use the number of columns (i.e. output dimension)
    atomHs = label2Hs(target, outDim);

    res = verify_specification(X, atomHs);

    if res == 2
        res = checkViolated(X, target);
    end

    result = res;
end

function res = checkViolated(Set, label)
    res = 5; % assume unknown (property is not unsat, try to sat)
    target = label;
    % Get bounds for every index
    [lb,ub] = Set.getRanges;
    maxTarget = ub(target);
    % max value of the target index smaller than any other lower bound?
    if any(lb > maxTarget)
        res = 0; % falsified
    end
end

function Hs = label2Hs(label, outDim)
    % Convert output target to halfspace for verification.
    % outDim: dimensionality of the network output.
    if nargin < 2
        error('Please supply the output dimension as the second argument.');
    end

    target = label;

    % Build the halfspace constraints assuming each output dimension is a score.
    % Remove the constraint for the target and define the others accordingly.
    G = eye(outDim);
    G(target,:) = [];
    % For each non-target entry, require: output(target) - output(i) >= 0 
    % which can be written as -1*output(i) + 1*output(target) >= 0;
    G = -G;  % multiply by -1 to get the correct signs
    G(:, target) = 1;

    g = zeros(size(G,1),1);

    Hs = [];
    for i = 1:length(g)
        Hs = [Hs; HalfSpace(G(i,:), g(i))];
    end
end
