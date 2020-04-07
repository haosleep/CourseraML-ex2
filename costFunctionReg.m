function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

% 正規化的損失函數就是在原本的損失函數後面再加上 λ/(2m) * sigma(j=1:n)θj^2
% λ/(2m)是恆正,θ^2也是恆正的
% 損失函數的根本就是值愈小愈好,機器學習求的theta也是為了要讓損失函數的值達到最小
% 因此,在損失函數要加上一項λ/(2m) * sigma(j=1:n)θj^2時
% 為了要讓損失函數的值變小,θ自然也得因此變小
% θ變小就不會讓特徵值的影響太大,可以有效解決過擬合的問題

% λ(拉格朗日乘數)是用來調整正規化幅度的
% λ給太大的話,每個θ會因此變得極小反而導致欠擬合的結果(特徵值的影響變得微乎其微)
% λ給太小的話卻也有可能還是一樣是過擬合的情況
% 至於λ取多少比較恰當也只能靠實驗和經驗得知(就像前次作業的學習率一樣)

hx = sigmoid(X * theta);
% 正規化主要是為了降低各個θ的
% 但一般來說,θ0(對應的X0固定是1)基本上是不用降的
% 雖然其實要降也是沒有關係(θ0X0 = θ0,本身就是一個常數),不過意義不大
% 所以通常正規化的θ不會包含θ0

% 因此,先設一個新的參數cF_theta來保存原本的theta
% 在讓cF_theta的第一項,也就是θ0的部分改為0
% 由cF_theta來處理正規化的部分,就可以使θ0不受正規化影響
cF_theta = theta;
cF_theta(1) = 0;
J = (-y' * log(hx) - (1 - y)' * log(1 - hx)) / m + (cF_theta' * cF_theta) * lambda / (2 * m);

% 加上了正規化的偏導數公式就是原本的偏導數後面再加上(λ/m)*θj
% θ自然也是用對應正規化的cF_theta
grad = ((hx - y)' * X)' ./ m + cF_theta * lambda / m;

% =============================================================

end
