function adjacency_tensor = edges2Adjacency(dataset)
    % Convert edge indices to adjacency matrices for each data point,
    % and return a 3D numeric array of size [num_nodes, num_nodes, num_graphs].
    %
    % Input:
    %   dataset - structure with fields:
    %     - edge_indices: {1×n cell} of edge indices for each graph
    %     - features: {1×n cell} of node features
    %     - labels: {1×n cell} of labels
    %
    % Output:
    %   adjacency_tensor - numeric array of size [num_nodes, num_nodes, num_graphs]
    
    num_graphs = length(dataset.edge_indices);
    
    % Assume all graphs have the same number of nodes. Use the first graph's feature data.
    if ~isempty(dataset.features{1})
        num_nodes = size(dataset.features{1}, 1);
    else
        % If features of the first graph are empty, infer from the first edge index list.
        num_nodes = max(dataset.edge_indices{1}(:));
    end
    
    % Preallocate a 3D numeric array
    adjacency_tensor = zeros(num_nodes, num_nodes, num_graphs);
    
    % Process each graph
    for i = 1:num_graphs
        edges = dataset.edge_indices{i};
        % Determine the number of nodes for this graph.
        % (We assume all graphs are fixed to num_nodes; if not, one could use:
        %  n = size(dataset.features{i}, 1); )
        n = num_nodes;
        
        % Initialize a numeric adjacency matrix for the current graph.
        adjacency = zeros(n, n);
        
        if ~isempty(edges)
            if size(edges,1) == size(edges,2)
                % If the provided edges are already in an NxN format,
                % then just copy the relevant block.
                adjacency = full(double(edges));
            else
                % Otherwise, treat edges as an edge list.
                % Ensure that edges is 2-by-num_edges.
                if size(edges,1) > size(edges,2)
                    edges = edges';
                end
                valid_edges = (edges(1,:) > 0 & edges(1,:) <= n) & ...
                              (edges(2,:) > 0 & edges(2,:) <= n);
                sources = edges(1, valid_edges);
                targets = edges(2, valid_edges);
                % Populate the adjacency matrix.
                for j = 1:length(sources)
                    adjacency(sources(j), targets(j)) = 1;
                    % Optionally, add symmetry if the graph is undirected:
                    % adjacency(targets(j), sources(j)) = 1;
                end
            end
        end
        
        % Store the computed adjacency in the 3D array.
        adjacency_tensor(:, :, i) = adjacency;
    end
end
