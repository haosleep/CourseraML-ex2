function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

% 讓特徵映射到6次方
% 會得到
% 1
% x1, x2
% x1^2, x1*x2, x2^2
% x1^3, x1^2*x2, x1*x2^2, x2^3
% ...下略
% 總計1+2+3+4+5+6+7 = 28個特徵結果
degree = 6;
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        % 矩陣(:, end+1),會在最右邊再新加一列 等號右邊的數值
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end