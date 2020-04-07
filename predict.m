function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%


% 經過機器學習後的theta和資料X進行計算後的結果(也就是h(x))
% 其值為正時,通過邏輯函數的結果會大於0.5,預測判斷結果會是1
% 其值為負時,通過邏輯函數的結果會小於0.5,預測判斷結果會是0
% 因此,使用sigmoid(X * theta) >= 0.5的判斷式
% 剛好就能得到 資料數量*1 的矩陣,其結果都是非0即1的邏輯值
p = sigmoid(X * theta) >= 0.5;




% =========================================================================


end
