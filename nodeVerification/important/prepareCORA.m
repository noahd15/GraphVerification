function prepareCoraForGCN(outputFile)
    % Step 1: Download and extract CORA
    url = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz';

    avProjectHome = getenv('AV_PROJECT_HOME');
    if isempty(avProjectHome)
        error('Environment variable AV_PROJECT_HOME is not set.');
    end

    tmpDir = fullfile(avProjectHome, 'cora_tmp');
    if ~exist(tmpDir, 'dir')
        mkdir(tmpDir);
    end

    tgzFile = fullfile(tmpDir, 'cora.tgz');
    if ~isfile(tgzFile)
        fprintf('Downloading CORA dataset...\n');
        websave(tgzFile, url);
    end

    % Extract if files are missing
    contentFile = fullfile(tmpDir, 'cora.content');
    citesFile = fullfile(tmpDir, 'cora.cites');
    if ~isfile(contentFile) || ~isfile(citesFile)
        fprintf('Extracting CORA dataset...\n');
        untar(tgzFile, tmpDir);
    end

    % Step 2: Load the content and citation files
    content = readtable(contentFile, 'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', false);
    cites = readtable(citesFile, 'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', false);

    node_ids = table2array(content(:, 1));
    features = table2array(content(:, 2:end-1));
    rawLabels = table2cell(content(:, end));

    % Encode labels to integers
    [labelSet, ~, labelInts] = unique(rawLabels);
    labels = categorical(labelInts, 1:numel(labelSet), labelSet);

    % Build node ID to index map
    [~, ~, nodeIdxMap] = unique(node_ids);
    numNodes = length(nodeIdxMap);

    % Build adjacency matrix
    edge_i = nodeIdxMap(cites{:, 1});
    edge_j = nodeIdxMap(cites{:, 2});
    A = sparse(edge_i, edge_j, 1, numNodes, numNodes);
    A = A + A'; A(A > 1) = 1;
    A = double(A); % ensure numeric

    % Step 3: PCA reduce features
    reducedDim = 64;
    if size(features, 2) < reducedDim
        error('Feature dimensionality (%d) is less than the reduced dimension (%d).', size(features, 2), reducedDim);
    end
    [coeff, Xpca] = pca(features);
    reducedFeatures = Xpca(:, 1:reducedDim);

    % Step 4: Split indices
    rng(42);
    num = size(features, 1);
    idx = randperm(num);
    trainCut = round(0.6 * num);
    valCut = round(0.8 * num);
    idxTrain = idx(1:trainCut);
    idxValidation = idx(trainCut+1:valCut);
    idxTest = idx(valCut+1:end);

    % Step 5: Wrap in 3D arrays (1 graph)
    featureData_reduced = zeros(numNodes, reducedDim, 1);
    featureData_reduced(:, :, 1) = reducedFeatures;

    labelData = zeros(1, numNodes);
    labelData(1, :) = double(labels);

    adjacencyData = zeros(numNodes, numNodes, 1);
    adjacencyData(:, :, 1) = A;

    % Step 6: Save
    save(outputFile, 'featureData_reduced', 'labelData', 'adjacencyData', 'idxTrain', 'idxValidation', 'idxTest');
    fprintf('CORA dataset prepared and saved to %s\n', outputFile);
end
