function [Y, numDimsY] = onnxConstantOfShape(value, ONNXShape)
% Returns a DLT tensor with the reverse of the ONNXShape. 

% Copyright 2020 The MathWorks, Inc.

DLTShape = fliplr(extractdata(ONNXShape(:)'));
numDimsY = numel(DLTShape);
switch numDimsY
    case 0
        % If shape is empty, output is a scalar
        Y = value;
    case 1
        Y = ones(DLTShape,1) .* value;
    otherwise
        Y = ones(DLTShape) .* value;
end
end
