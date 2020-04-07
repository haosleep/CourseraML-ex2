function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


% Find Indices of Positive and Negative Examples
% 傳進來的y是100x1的向量
% y==1,y==0同樣會是100x1的向量(不過裡面的內容是logical)
% find(向量)的功用是回傳向量中非0的索引(向量形式)
% pos,neg即為y資料中,y屬於1和y屬於0的部分
pos = find(y==1); neg = find(y == 0);
% Plot Examples
% 利用pos,neg區分X資料(對應y == 1和y == 0的)
% y == 1的資料在二維圖上以黑色十字標示
% y == 0的資料在二維圖上以黑色圓圈,內部塗黃標示
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);




% =========================================================================



hold off;

end
