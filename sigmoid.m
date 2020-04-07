function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% 因為這次處理的是分類問題(結果只有0跟1)
% 因此要利用邏輯回歸的方式處理
% 邏輯回歸用上的邏輯函數公式 g(z) = 1 / (1 + e^-z)
% g(z)結果會限縮在0~1之間
% 當z > 0時,g(z) > 0.5; z < 0時,g(z) < 0.5
% 剛好對應於分類問題0,1的結果

% 為了讓這函式不論傳進來的z是純量,向量還是矩陣都能正常運作
% 這邊的除要使用./
g = 1 ./ (1 + exp(-z));


% =============================================================

end
