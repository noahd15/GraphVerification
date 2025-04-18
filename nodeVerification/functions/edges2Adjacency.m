function adjacency_tensor = edges2Adjacency(dataset)
    % Convert edge indices to adjacency matrices for each data point,
    % and return a 3D numeric array of size [num_nodes, num_nodes, num_graphs].
    %
    % Input:
    %   dataset - structure with fields:
    %     - edge_indices: {1×n cell} of edge indices for each graph
    %     - features: either a numeric array of size [num_nodes, num_features, num_graphs]
    %                 or a 1×n cell array where each cell contains a numeric matrix
    %                 of size [num_nodes, num_features] for each graph.
    %     - labels: (not used here)
    %
    % Output:
    %   adjacency_tensor - numeric array of size [num_nodes, num_nodes, num_graphs]
    
    num_graphs = length(dataset.edge_indices);
    
    % Use the features field to determine the number of nodes.
    if ~isempty(dataset.features)
        if iscell(dataset.features)
            % When features are stored as a cell array.
            num_nodes = size(dataset.features{1}, 1);
        else
            % When features are stored as a numeric 3D array (e.g. [num_nodes, num_features, num_graphs]).
            num_nodes = size(dataset.features, 1);
        end
    else
        % Fallback: infer node count from the edge indices of the first graph.
        num_nodes = max(dataset.edge_indices{1}(:));
    end
    
    % Preallocate the output 3D array.
    adjacency_tensor = zeros(num_nodes, num_nodes, num_graphs);
    disp(num_graphs);
    % Process each graph.
    for i = 1:num_graphs
 %       fprintf('Edge indices shape: %s\n', dataset.edge_indices);
        edges = dataset.edge_indices(:, :, i);

        % Assume that the graph has fixed num_nodes.
        n = num_nodes;
        
        % Initialize the adjacency matrix for the current graph.
        adjacency = zeros(n, n);
        
        if ~isempty(edges)
            if size(edges,1) == size(edges,2)
                % If edges are already given as an NxN matrix.
                adjacency = full(double(edges));
            else
                % Otherwise, treat 'edges' as an edge list.
                if size(edges,1) > size(edges,2)
                    edges = edges';
                end
                valid_edges = (edges(1,:) > 0 & edges(1,:) <= n) & ...
                              (edges(2,:) > 0 & edges(2,:) <= n);
                sources = edges(1, valid_edges);
                targets = edges(2, valid_edges);
                for j = 1:length(sources)
                    adjacency(sources(j), targets(j)) = 1;
                    % Uncomment the next line if the graph is undirected:
                    % adjacency(targets(j), sources(j)) = 1;
                end
            end
        end
        
        % Assign the computed adjacency matrix into the 3D array.
        adjacency_tensor(:, :, i) = adjacency;
    end
end