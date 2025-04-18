function [symbol, count] = labelSymbol(labelNum)
% LABELSYMBOL Convert label number to label string
%   symbol = labelSymbol(labelNum) returns the label string of the
%   specified label number.
%
%   [symbol, count] = labelSymbol(labelNum) also returns the count for
%   each label.
%
%   The function supports labels:
%   0 -> "Not Compromised"
%   1 -> "Low Privilege Compromised"
%   2 -> "High Privilege Compromised"

numLabels = numel(labelNum);
symbol = strings(numLabels, 1);
count = strings(numLabels, 1);

notCompCount = 0;
lowPrivCount = 0;
highPrivCount = 0;

for i = 1:numLabels
    switch labelNum(i)
        case 0
            symbol(i) = "Not Compromised";
            notCompCount = notCompCount + 1;
            count(i) = "NotComp" + notCompCount;
        case 1
            symbol(i) = "Low Privilege Compromised";
            lowPrivCount = lowPrivCount + 1;
            count(i) = "LowPriv" + lowPrivCount;
        case 2
            symbol(i) = "High Privilege Compromised";
            highPrivCount = highPrivCount + 1;
            count(i) = "HighPriv" + highPrivCount;
        otherwise
            error("Invalid label number: %d. Supported labels are 0, 1, and 2.", labelNum(i));
    end
end

end