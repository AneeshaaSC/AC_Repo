function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);%28000
num_labels = size(Theta2, 1);
%Theta1 is 50x785
%Theta2 is 10x51

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');%(28000x785)(785x50)=28000x50
h2 = sigmoid([ones(m, 1) h1] * Theta2');%(28000x51)(51x10) = 28000x10
[dummy, p] = max(h2, [], 2);

% =========================================================================


end
