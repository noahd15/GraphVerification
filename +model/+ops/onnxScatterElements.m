function [Y, numDimsY] = onnxScatterElements(X, ONNXIndices, updates, ONNXAxis, reduction, numDimsX)
% Implements the ONNX ScatterElements operator

% Copyright 2020-2024 The MathWorks, Inc.

if ONNXAxis<0
    ONNXAxis = ONNXAxis + numDimsX;                                 % Axis can be negative. Convert it to its positive equivalent.
end
% Convert axis to DLT axis. ONNXAxis is origin 0 and we index dimensions in
% reverse ONNX ordering
mlAxis = numDimsX - ONNXAxis;
% Convert ONNXIndices to DLT. 
ONNXIndices(ONNXIndices<0) = ONNXIndices(ONNXIndices<0) + size(X,mlAxis);  % Make negative ONNXIndices nonnegative.
mlIndices = ONNXIndices + 1;
% Find the linear indices of X into which we scatter the updates
mlLinearIndices = scatterElementsLinearIndices(X, mlAxis, mlIndices);
Y = X;
switch reduction
    case "none"
        Y(mlLinearIndices) = updates;
    case "add"
        Y(mlLinearIndices) = Y(mlLinearIndices)+reshape(updates,[],1);
    case "mul"
        Y(mlLinearIndices) = Y(mlLinearIndices).*reshape(updates,[],1);
    case "max"
        Y(mlLinearIndices) = max(Y(mlLinearIndices),reshape(updates,[],1));
    case "min"
        Y(mlLinearIndices) = min(Y(mlLinearIndices),reshape(updates,[],1));
end
numDimsY = numDimsX;

    function L = scatterElementsLinearIndices(X,dim,Indices)
        % Reduce to 0-based.
        L = Indices;
        L = L - 1;
        if dim > 1
            % Shift to be the component from the relevant dimension.
            L = L*prod(size(X, 1:(dim-1)));
        end
        sz = 1;
        % Go through the dims of X.
        for d = 1:ndims(X)
            if d ~= dim
                % Add in the component from this dimension.
                idx = ((1:size(Indices,d))-1)*sz;                   % Note we take the size of Indices here, not X.
                szvec = [ones(1, d-1),  numel(idx), 1];
                L = L + reshape(idx, szvec);
            end
            % Increase the cumulative size.
            sz = sz * size(X,d);
        end
        % Shift back to being 1-based.
        L = L(:) + 1;
    end
end
