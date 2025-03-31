% edges2Adjacency.m
function adjacency_matrices = edges2Adjacency(dataset)
    % Convert edge indices to adjacency matrices for each data point
    % 
    % Input:
    %   dataset - structure with fields:
    %     - edge_indices: {1×n cell} of edge indices for each graph
    %     - features: {1×n cell} of node features
    %     - labels: {1×n cell} of labels
    %
    % Output:
    %   adjacency_matrices - {1×n cell} of adjacency matrices

    num_graphs = length(dataset.edge_indices);
    adjacency_matrices = cell(1, num_graphs);
    
    for i = 1:num_graphs
        % Get edge indices for this graph
        edges = dataset.edge_indices{i};
        
        % Get number of nodes from features
        num_nodes = size(dataset.features{i}, 1);
        if num_nodes == 0
            % If features don't provide node count, try inferring from edges
            if ~isempty(edges)
                num_nodes = max(edges(:));
            else
                num_nodes = 0;
            end
        end
        
        % Skip if no nodes
        if num_nodes == 0
            adjacency_matrices{i} = sparse(0, 0);
            continue;
        end
        
        % Initialize adjacency matrix
        adjacency = sparse(num_nodes, num_nodes);
        
        % Fill adjacency matrix using edge indices
        if ~isempty(edges)
            % Ensure edges are valid indices
            valid_edges = edges > 0 & edges <= num_nodes & round(edges) == edges;
            if size(valid_edges, 1) >= 2
                valid_idx = all(valid_edges(1:2,:), 1);
                
                % Extract source and target nodes for valid edges
                sources = edges(1, valid_idx);
                targets = edges(2, valid_idx);
                
                % Create sparse adjacency matrix directly
                adjacency = sparse(sources, targets, ones(1, length(sources)), num_nodes, num_nodes);
            end
        end
        
        adjacency_matrices{i} = adjacency;
    end
end