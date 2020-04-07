function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% �]���o���B�z���O�������D(���G�u��0��1)
% �]���n�Q���޿�^�k���覡�B�z
% �޿�^�k�ΤW���޿��Ƥ��� g(z) = 1 / (1 + e^-z)
% g(z)���G�|���Y�b0~1����
% ��z > 0��,g(z) > 0.5; z < 0��,g(z) < 0.5
% ��n������������D0,1�����G

% ���F���o�禡���׶Ƕi�Ӫ�z�O�¶q,�V�q�٬O�x�}���ॿ�`�B�@
% �o�䪺���n�ϥ�./
g = 1 ./ (1 + exp(-z));


% =============================================================

end
