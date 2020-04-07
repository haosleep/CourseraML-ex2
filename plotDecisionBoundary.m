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

% 判斷特徵值個數
% size(矩陣, 2)表示矩陣列數(size(矩陣, 1)則回傳矩陣行數)
% 若X列數<=3, 表示特徵值不超過兩個(兩個特徵值+X0)
% 決策邊界就只要以一條線表示即可
if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    % 兩點即成一線,故要找表示決策邊界用的兩點
    % 取 資料X第一個特徵值 的 最小值減一固定值 和 最大值加一固定值
    % 這兩值用來作為決策邊界用的兩個點的x軸值
    % 就可以將所有資料X都含括在線的範圍內
    
    % 補充說明:
    % 在上面的plotData畫分類問題的二維圖表示資料時
    % x軸的值表示資料的第一個特徵值X1,y軸的值表示資料第二個特徵值X2
    % 因此這樣取plot_x,所用的資料X的X1絕對會介於這兩點內
    % 二維圖上表示決策邊界會比較美觀  
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    % 根據上面決定的X1,在已知的theta下找對應的X2
    % 決策邊界即為h(x) = 0 (取sigmoid時,結果會是0.5)
    % 特徵值不超過兩個的h(x) = θ0 + θ1 * X1 + θ2 * X2 = 0
    % 整理後可得
    % X2 = (-1/θ2) * (θ0 + θ1 * X1)
    % 就取得plot_x對應的plot_y
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    % plot_x為1*2矩陣,表示(第一個點的x,第二個點的x)
    % plot_y為1*2矩陣,表示(第一個點的y,第二個點的y)
    % plot(plot_x, plot_y)即為將對應的兩點用線連接
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    % 加上說明
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    % 把x軸範圍設為30~100,y軸範圍也設為30~100
    % 為了將所有點都包含在內(當然這數值是根據了ex2data1的資料所設計過的)
    axis([30, 100, 30, 100])
else
    % 下面是特徵值個數超過兩個的情況(沒辦法用單純的直線處理)
    % Here is the grid range
    % u,v從-1~1.5(-1,1.5這數值應該是根據ex2data2的資料所設計過的)之間取50個點作為X1,X2值
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % 讓u,v進行一樣的特徵映射後再乘上經過機器學習後得到的theta
    % 得到當資料給的是u,v組合的數值時的預測結果的矩陣
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    % 為了正確的對應X1,X2,改轉置矩陣(跟前一次作業ex1.m中的J_vals一樣)
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    % 畫等高線,在x軸是u,y軸是v時畫出z,只畫出z = 0時的等高線
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
