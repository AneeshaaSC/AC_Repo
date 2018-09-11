function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Adding bias unit to X matrix

X=[ones(m, 1) X]; 
a_1=X;

% X is size 5000x401
% Theta1 is size 25x401
% Theta2 is size 10x26

% Get required 
%activiation layer in second unit
%(5000x401)(401x25)
%z_2 size is (5000x25)
z_2=a_1*Theta1';
a_2=sigmoid(z_2);

 
% add ones column to include bias unit
a_2=[ones(size(a_2,1),1) a_2];
% a_2 is size 5000x26
 
% activation unit in third layer
z_3=a_2*Theta2';
a_3=sigmoid(z_3);

%h is a (5000x10) matrix 
h=a_3;

%regularization
T1=Theta1(:,2:end);
T2=Theta2(:,2:end);
SQT1=T1.^2;
SQT2=T2.^2;
reg=(lambda/(2*m))*[sum(SQT1(:))+sum(SQT2(:))];

y = eye(num_labels)(y,:);
J = ((1/m) * sum(sum((-y .* log(h))-((1-y) .* log(1-h))))) + reg;

%calculate required values for backpropagation

delta_3=a_3-y;

% g_z_2 size =5000x25
g_z_2=sigmoidGradient(z_2);


% ((5000x10) * (10x25)) .* (5000x25)
delta_2=(delta_3 * Theta2(:,2:end)).*g_z_2 ;
% delta_2 size= 5000x25
 
%size of Big_delta_2  is 10x26
Big_delta_2=delta_3'*a_2;

% size of Big_delta_1 is 25x401
Big_delta_1=delta_2'*a_1;


Theta1_grad=Big_delta_1/m;
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda*Theta1(:,2:end))/m; % size of D_1 is 25x401
Theta2_grad=Big_delta_2/m;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda*Theta2(:,2:end))/m;  %size of D_2  is 10x26

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end;
