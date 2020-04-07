function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% 邏輯回歸下的損失函數公式(其實類似於線性插值的處理方式)
% J(θ) = (1/m) * sigma(i=1:m)(-yi*log(h(Xi)) - (1-yi)log(1-h(Xi)))
% h(X)一樣是將X資料代入theta的計算公式後的預估結果值
% 不過在分類問題的情況下,要利用sigmoid.m把結果轉為0~1

hx = sigmoid(X * theta);
J = (-y' * log(hx) - (1 - y)' * log(1 - hx)) / m;

% 偏導數
% (1/m) * sigma(i=1:m)(h(Xi) - yi)*xj
grad = ((hx - y)' * X)' ./ m;


% =============================================================

end
