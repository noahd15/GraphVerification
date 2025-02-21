function [sz, numDimsY] = prepareTileArgs(ONNXRepeats)
% Prepares arguments for implementing the ONNX Tile operator.  The
% generated code looks like this: 
% 
%   Copyright 2020-2022 The MathWorks, Inc.    
%
% [sz, NumDims.Y] = prepareTileArgs(Vars.repeats);
% Vars.Y = repmat(Vars.X, sz)

ONNXRepeats = extractdata(ONNXRepeats(:)');     % Make repeats a row.
if isscalar(ONNXRepeats)
    % If scalar, just repmat a 1D vector into a longer 1D vector
    sz = [ONNXRepeats 1];
else
    sz = fliplr(ONNXRepeats);
end
numDimsY = numel(ONNXRepeats);
end
