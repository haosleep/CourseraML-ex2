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

% ex2data2.txt �t118*3�����
% �e��C���O�O��L������Ĥ@�����թM�ĤG�����ժ����G
% �ĤT�C�O�L������G�O�_�X��
% �H�U�N�Q�ξ����ǲߪ��覡�B�z�o�����,���R�X�b�o����ƤU���M�����
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

% ���Q��plotData.m�N��ƥΤG���Ϫ�ܥX��
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

% �q�e�����ͪ��G���Ϥ��w�i�ݥX�o������ƨS��k�Ⱦa�@���u�@���M�����(�����X)
% ����ƩҴ��Ѫ��S�x�u�����
% �]���o�̱N�Q��mapFeature.m�i��S�x�M�g
% �W�[�S�x�Ȫ���������ӯS�x�Ȥ��|��´N�u�a���theta�ȶi��w��
% �i�D�o��A���M�����

% mapFeature.m���]�w�O�N�S�x�M�g��6����,�|���o28�ӯS�x���G
% �]���쥻�O118*2�x�}��X���
% �|���ܬ�118*28�x�}
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
% ���M�Q�ίS�x�M�g�i�H���D�o����������ƪ��M�����
% ���]�i��o�͹L���X�����D
% (�ӹL��n��X��������ƪ��M�����,�䵲�G�ϦӹL�󷥺ݥH�ܩ�n�w����ƥH�~���s��T�ɷǽT�ʷ��C)
% �ҥH�n�A�[�W���W�ƨӳB�z
% �n�bcostFunctionReg.m�D�o�[�W���W�ƫ᪺�l����ƩM���(part1�@�~)
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
% �]���S�x�M�g�᪺�S�x�ȼƶq�W��28��
% �o�̥u��print�e���ӱ�ר��ˬd�p��O�_���T
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

% �e�����������]�m�n��,�N�i��Mex2�ɤ@�˪������ǲߨӨD�o�̲ת�theta��
% �i�H���է��ܩԮ�Ԥ�Ѽƨ��[��ݬݨ�L���G

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
% �o���_ex2.m�I�sfminunc��,�h�^�ǤF�@��exit_flag
% �ھ�help fminunc���������exit_flag�i��^�Ǫ����G�����ؼƭȤ��@,���������y�z�U�ƭȹ������N��
% ���a�@��,����ex2_reg��^�Ǫ�exit_flag���G�O3
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

% �ӵ�part2����������
% ��ڨӹ��լݬݷ�Ԯ�Ԥ�ѼƳ]�w���P�Ȯ�,�M����ɷ|�����򵲪G
% ���եΪ��Ԯ�Ԥ�ѼƤ]�ھ�part2���������ȨӶi�����

% ���M�]�Q���e���@�~ex1_multi.m�ɤ@�˱N�Ҧ����G��ø�s�b�P�@�i�Ϥ�
% ���bplotDecisionBoundary.m���|�I�splotData.m
% ��plotData.m�̷|����figure���}�@�i�s��
% �n�Q��Ҧ����G��ø�s�b�P�@�i�Ϥ�,�N�o�qplotData.m�������O�}�l��g
% ��ߧ�ʴT�פӦh�v�T��򥻪��@�~���e,�B�o�]���⥻���@�~�����I,�G�N��F
test_lambda = [0 10 100];
for cL = 1:length(test_lambda)
  tLiTheta = initial_theta;
  [theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, test_lambda(cL))), initial_theta, options);
  plotDecisionBoundary(theta, X, y);
  % �h�B�b�Ϥ����D�̼е��ϥΪ��Ԯ�Ԥ�ѼƭȨӰϤ��Ϥ����e
  hold on;
  title(sprintf('lambda = %g', test_lambda(cL)))
  hold off;
endfor


% Compute accuracy on our training set
% �̫�p������ǲ߹w�����ǽT��
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');

