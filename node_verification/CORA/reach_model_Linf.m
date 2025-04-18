function reach_model_Linf(modelPath, epsilon, adjacencyDataTest, featureDataTest, labelDataTest)
    % Verification of a full‐batch GCN on the Cora test subgraph
    if nargin==0
        % fall back to defaults if called with no args
        projectRoot = getenv('AV_PROJECT_HOME');
        data = load(fullfile(projectRoot,'data','cora_node.mat'));
        A_full = data.edge_indices(:,:,1);
        X_full = data.features(:,:,1);
        y_full = double(data.labels(:)) + 1;
        numNodes = size(X_full,1);
        rng(2024);
        [~,~,idxTest] = trainingPartitions(numNodes, [0.8 0.1 0.1]);
        adjacencyDataTest = A_full(idxTest, idxTest);
        featureDataTest   = X_full(idxTest, :);
        labelDataTest     = y_full(idxTest);
        modelPath = "cora_node_gcn_1";
        epsilon   = 0.005;
    end

    % Load trained GCN parameters
    S = load("models/"+modelPath+".mat","parameters");
    params = S.parameters;

    % Gather weights
    W = { gather(params.mult1.Weights), ...
          gather(params.mult2.Weights), ...
          gather(params.mult3.Weights) };

    % Build the test subgraph exactly as in training
    Atest = sparse(double(adjacencyDataTest));   % N×N sparse adjacency
    
    % == CHANGE HERE: no transpose, so Xtest is N×F ==
    Xtest = dlarray(featureDataTest);            % N×F
    ytest = labelDataTest;                       % N×1 labels

    % Normalize adjacency once
    Averify = normalizeAdjacency(Atest);

    % Create output directory if needed
    outDir = fullfile("verification_results","mat_files");
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    % Verify under each ε
    for k = 1:numel(epsilon)
        epsi = epsilon(k);

        % Build a single ImageStar over all node features
        lb = extractdata(Xtest - epsi);    % N×F
        ub = extractdata(Xtest + epsi);    % N×F
        Xverify = ImageStar(lb, ub);       % V is N×F×1×nDirs

        tStart = tic;
        Rstars = computeReachability(W, ReluLayer(), 'approx-star', Xverify, Averify);
        elapsed = toc(tStart);

        % Save results
        save(fullfile(outDir, ...
            "verified_nodes_" + modelPath + "_eps_" + string(epsi) + ".mat"), ...
            'Rstars', 'ytest', 'elapsed', '-v7.3');
        fprintf("ε=%.4f verified in %.2fs\n", epsi, elapsed);
    end
end

%% ------------------------------------------------------------------------
function ANorm = normalizeAdjacency(A)
    % Symmetric normalization of sparse adjacency
    A = A + speye(size(A));
    d = sum(A,2);
    D = spdiags(d.^(-0.5), 0, size(A,1), size(A,1));
    ANorm = D * A * D;
end

%% ------------------------------------------------------------------------
function Ynodes = computeReachability(weights, L, reachMethod, Xverify, adjMat)
    % Performs 3 GCN layers on ImageStar Xverify and slices per‐node stars.

    V = Xverify.V;                          % now N×F×1×nDirs
    [nNodes, inF, ~, nDirs] = size(V);

    % --- Layer 1 ---
    V1 = reshape(V, [nNodes, inF, nDirs]);                   
    V1 = tensorprod(full(adjMat), V1, 2, 1);                 
    V1 = tensorprod(V1,            weights{1},  2, 1);      
    V1 = permute(V1, [1 3 4 2]);                             
    S1 = ImageStar(V1, Xverify.C, Xverify.d, Xverify.pred_lb, Xverify.pred_ub);
    R1 = L.reach(S1, reachMethod);

    % --- Layer 2 ---
    hid1 = size(R1.V,2);
    V2 = reshape(R1.V, [nNodes, hid1, nDirs]);
    V2 = tensorprod(full(adjMat), V2, 2, 1);
    V2 = tensorprod(V2,               weights{2},  2, 1);
    V2 = permute(V2, [1 3 4 2]);
    S2 = ImageStar(V2, R1.C, R1.d, R1.pred_lb, R1.pred_ub);
    R2 = L.reach(S2, reachMethod);

    % --- Layer 3 ---
    hid2 = size(R2.V,2);
    V3 = reshape(R2.V, [nNodes, hid2, nDirs]);
    V3 = tensorprod(full(adjMat), V3, 2, 1);
    V3 = tensorprod(V3,               weights{3},  2, 1);
    V3 = permute(V3, [1 3 4 2]);
    S3 = ImageStar(V3, R2.C, R2.d, R2.pred_lb, R2.pred_ub);

    % Slice off each node’s output star
    Ynodes = cell(nNodes,1);
    for u = 1:nNodes
        Ynodes{u} = S3.slice(1, u);
    end
end
