function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

% �P�_�S�x�ȭӼ�
% size(�x�}, 2)��ܯx�}�C��(size(�x�}, 1)�h�^�ǯx�}���)
% �YX�C��<=3, ��ܯS�x�Ȥ��W�L���(��ӯS�x��+X0)
% �M����ɴN�u�n�H�@���u��ܧY�i
if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    % ���I�Y���@�u,�G�n���ܨM����ɥΪ����I
    % �� ���X�Ĥ@�ӯS�x�� �� �̤p�ȴ�@�T�w�� �M �̤j�ȥ[�@�T�w��
    % �o��ȥΨӧ@���M����ɥΪ�����I��x�b��
    % �N�i�H�N�Ҧ����X���t�A�b�u���d��
    
    % �ɥR����:
    % �b�W����plotData�e�������D���G���Ϫ�ܸ�Ʈ�
    % x�b���Ȫ�ܸ�ƪ��Ĥ@�ӯS�x��X1,y�b���Ȫ�ܸ�ƲĤG�ӯS�x��X2
    % �]���o�˨�plot_x,�ҥΪ����X��X1����|����o���I��
    % �G���ϤW��ܨM����ɷ|������[  
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    % �ھڤW���M�w��X1,�b�w����theta�U�������X2
    % �M����ɧY��h(x) = 0 (��sigmoid��,���G�|�O0.5)
    % �S�x�Ȥ��W�L��Ӫ�h(x) = �c0 + �c1 * X1 + �c2 * X2 = 0
    % ��z��i�o
    % X2 = (-1/�c2) * (�c0 + �c1 * X1)
    % �N���oplot_x������plot_y
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    % plot_x��1*2�x�},���(�Ĥ@���I��x,�ĤG���I��x)
    % plot_y��1*2�x�},���(�Ĥ@���I��y,�ĤG���I��y)
    % plot(plot_x, plot_y)�Y���N���������I�νu�s��
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    % �[�W����
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    % ��x�b�d��]��30~100,y�b�d��]�]��30~100
    % ���F�N�Ҧ��I���]�t�b��(��M�o�ƭȬO�ھڤFex2data1����Ʃҳ]�p�L��)
    axis([30, 100, 30, 100])
else
    % �U���O�S�x�ȭӼƶW�L��Ӫ����p(�S��k�γ�ª����u�B�z)
    % Here is the grid range
    % u,v�q-1~1.5(-1,1.5�o�ƭ����ӬO�ھ�ex2data2����Ʃҳ]�p�L��)������50���I�@��X1,X2��
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % ��u,v�i��@�˪��S�x�M�g��A���W�g�L�����ǲ߫�o�쪺theta
    % �o����Ƶ����Ou,v�զX���ƭȮɪ��w�����G���x�}
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    % ���F���T������X1,X2,����m�x�}(��e�@���@�~ex1.m����J_vals�@��)
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    % �e�����u,�bx�b�Ou,y�b�Ov�ɵe�Xz,�u�e�Xz = 0�ɪ������u
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
