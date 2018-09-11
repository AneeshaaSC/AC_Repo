function plotsurvive(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.
data=csvread('train.csv');
X=data(:,1:6);
y=data(:,7);
% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
%data = load ("ex2data1.txt"); 
%y=data(:,3);
pos = find(y==1);
neg = find(y == 0);
Xpas=data(:,1);

%plot(Xpas(pos, 1),'k+','MarkerFaceColor', 'y','LineWidth', 2,'MarkerSize', 7);
plot(Xpas(neg, 1),'ko','MarkerFaceColor', 'b','LineWidth', 2,'MarkerSize', 7);
%plot(Xpas(neg, 1), Xpas(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

%


