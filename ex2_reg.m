%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

% ex2data2.txt 含118*3筆資料
% 前兩列分別是對微芯片的第一次測試和第二次測試的結果
% 第三列是微芯片結果是否合格
% 以下將利用機器學習的方式處理這份資料,分析出在這份資料下的決策邊界
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

% 先利用plotData.m將資料用二維圖表示出來
plotData(X, y);

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled

% 從前面產生的二維圖中已可看出這次的資料沒辦法僅靠一條線作為決策邊界(欠擬合)
% 但資料所提供的特徵只有兩個
% 因此這裡將利用mapFeature.m進行特徵映射
% 增加特徵值的維度讓兩個特徵值不會單純就只靠兩個theta值進行預測
% 可求得更適當的決策邊界

% mapFeature.m中設定是將特徵映射到6次方,會取得28個特徵結果
% 因此原本是118*2矩陣的X資料
% 會轉變為118*28矩陣
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
% 雖然利用特徵映射可以讓求得的更相應於資料的決策邊界
% 但也可能發生過擬合的問題
% (太過於要迎合給予的資料的決策邊界,其結果反而過於極端以至於要預測資料以外的新資訊時準確性極低)
% 所以要再加上正規化來處理
% 要在costFunctionReg.m求得加上正規化後的損失函數和梯度(part1作業)
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
% 因為特徵映射後的特徵值數量增為28個
% 這裡只先print前五個梯度來檢查計算是否正確
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% 前面的部分都設置好後,就進行和ex2時一樣的機器學習來求得最終的theta值
% 可以嘗試改變拉格朗日參數來觀察看看其他結果

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
% 這邊比起ex2.m呼叫fminunc時,多回傳了一個exit_flag
% 根據help fminunc的說明表示exit_flag可能回傳的結果有六種數值之一,說明中有描述各數值對應的意思
% 順帶一提,執行ex2_reg後回傳的exit_flag結果是3
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% 照著part2說明的指示
% 實際來嘗試看看當拉格朗日參數設定不同值時,決策邊界會有什麼結果
% 測試用的拉格朗日參數也根據part2說明給的值來進行測試

% 雖然也想像前次作業ex1_multi.m時一樣將所有結果都繪製在同一張圖中
% 但在plotDecisionBoundary.m中會呼叫plotData.m
% 而plotData.m裡會執行figure重開一張新圖
% 要想把所有結果都繪製在同一張圖中,就得從plotData.m中的指令開始改寫
% 擔心改動幅度太多影響到基本的作業內容,且這也不算本次作業的重點,故就算了
test_lambda = [0 10 100];
for cL = 1:length(test_lambda)
  tLiTheta = initial_theta;
  [theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, test_lambda(cL))), initial_theta, options);
  plotDecisionBoundary(theta, X, y);
  % 姑且在圖片標題裡標註使用的拉格朗日參數值來區分圖片內容
  hold on;
  title(sprintf('lambda = %g', test_lambda(cL)))
  hold off;
endfor


% Compute accuracy on our training set
% 最後計算機器學習預測的準確度
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');

