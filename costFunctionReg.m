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

% ���W�ƪ��l����ƴN�O�b�쥻���l����ƫ᭱�A�[�W �f/(2m) * sigma(j=1:n)�cj^2
% �f/(2m)�O��,�c^2�]�O����
% �l����ƪ��ڥ��N�O�ȷU�p�U�n,�����ǲߨD��theta�]�O���F�n���l����ƪ��ȹF��̤p
% �]��,�b�l����ƭn�[�W�@���f/(2m) * sigma(j=1:n)�cj^2��
% ���F�n���l����ƪ����ܤp,�c�۵M�]�o�]���ܤp
% �c�ܤp�N���|���S�x�Ȫ��v�T�Ӥj,�i�H���ĸѨM�L���X�����D

% �f(�Ԯ�Ԥ魼��)�O�Ψӽվ㥿�W�ƴT�ת�
% �f���Ӥj����,�C�ӣc�|�]���ܱo���p�ϦӾɭP�����X�����G(�S�x�Ȫ��v�T�ܱo�L�G��L)
% �f���Ӥp���ܫo�]���i���٬O�@�ˬO�L���X�����p
% �ܩ�f���h�֤�����]�u��a����M�g��o��(�N���e���@�~���ǲ߲v�@��)

hx = sigmoid(X * theta);
% ���W�ƥD�n�O���F���C�U�ӣc��
% ���@��ӻ�,�c0(������X0�T�w�O1)�򥻤W�O���έ���
% ���M���n���]�O�S�����Y(�c0X0 = �c0,�����N�O�@�ӱ`��),���L�N�q���j
% �ҥH�q�`���W�ƪ��c���|�]�t�c0

% �]��,���]�@�ӷs���Ѽ�cF_theta�ӫO�s�쥻��theta
% �b��cF_theta���Ĥ@��,�]�N�O�c0�������אּ0
% ��cF_theta�ӳB�z���W�ƪ�����,�N�i�H�ϣc0�������W�Ƽv�T
cF_theta = theta;
cF_theta(1) = 0;
J = (-y' * log(hx) - (1 - y)' * log(1 - hx)) / m + (cF_theta' * cF_theta) * lambda / (2 * m);

% �[�W�F���W�ƪ����ɼƤ����N�O�쥻�����ɼƫ᭱�A�[�W(�f/m)*�cj
% �c�۵M�]�O�ι������W�ƪ�cF_theta
grad = ((hx - y)' * X)' ./ m + cF_theta * lambda / m;

% =============================================================

end
