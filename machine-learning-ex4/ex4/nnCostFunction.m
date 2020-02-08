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

% initialize y vector
y_all = eye(num_labels);
DELTA_one = zeros(size(Theta1));
DELTA_two = zeros(size(Theta2));

for i=1:m
  x_vec = X(i,:);
  a_one = [1 x_vec]; % Adding bias input
  a_one = a_one'; % Make it a column vector
  y_vec = y_all(:, y(i));
  
  z_two = Theta1 * a_one;
  a_two = sigmoid(z_two); % 25x401 * 401*1 = 25x1    
  a_two = [1; a_two]; % adding bias - 26 x 1    
  z_three = Theta2 * a_two;
  a_three = sigmoid(z_three); % 10 x 26 * 26 x 1 = 10x1  
  % Calculating the cost for the given Theta1 and Theta2
  J = J + sum(((-1 .* y_vec) .* log(a_three)) - ((1 - y_vec) .* log(1 - a_three)));  
  
  %
  % Backpropagation
  %  
  delta_three = a_three - y_vec; % 10 x 1  
  g_prime_z2 = [1; sigmoidGradient(z_two)]; % 26 x 1
  delta_two = (Theta2' * delta_three) .* g_prime_z2;  % (26x10 * 10x1) .* 26x1
  % Remove the bias from delta_two
  delta_two = delta_two(2:end); % 25 x 1
  % Calculate accumulated error  
  DELTA_two = DELTA_two + (delta_three * a_two'); % 10x1 * 1X26 = 10x26
  DELTA_one = DELTA_one + (delta_two * a_one'); % 26x1 * 1x401 = 26x401
endfor

J = J / m;

% Adding the regularization cost
Theta1_without_bias = Theta1(:,2:size(Theta1, 2));
Theta2_without_bias = Theta2(:,2:size(Theta2, 2));
Theta1_without_bias_sqr = Theta1_without_bias .^ 2;
Theta2_without_bias_sqr = Theta2_without_bias .^ 2;

Theta1_sum = sum(sum(Theta1_without_bias_sqr));
Theta2_sum = sum(sum(Theta2_without_bias_sqr));

regularized_cost = (lambda/(2 * m)) * (Theta1_sum + Theta2_sum);

J = J + regularized_cost;

% Calculating Theta_gradient without regularization
Theta1_grad_unregularized = (DELTA_one ./ m);
Theta2_grad_unregularized = (DELTA_two ./ m);


% Calculating Theta_gradient with regularization
Theta1_reg = (lambda / m) .* Theta1;
Theta1_reg(:,1) = 0;
Theta1_grad = Theta1_grad_unregularized + Theta1_reg;

Theta2_reg = (lambda / m) .* Theta2;
Theta2_reg(:,1) = 0;
Theta2_grad = Theta2_grad_unregularized + Theta2_reg;
%Theta1_grad = Theta1_grad_unregularized .+ [zeros(size(Theta1_grad,1),1) ((lambda / m) .* Theta1_grad(:,2:end))];
%Theta2_grad = Theta2_grad_unregularized .+ [zeros(size(Theta2_grad,1),1) ((lambda / m) .* Theta2_grad(:,2:end))];
%Theta1_grad = [((1/m) .* DELTA_one(:,1)) Theta1_grad(:,2:end)];

%Theta2_grad = (DELTA_two ./ m) + (lambda .* Theta2_grad));
%Theta2_grad = [((1/m) .* DELTA_two(:,1)) Theta2_grad(:,2:end)];










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
