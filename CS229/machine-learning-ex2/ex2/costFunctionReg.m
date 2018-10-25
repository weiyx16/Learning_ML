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

z = X * theta;
y_tem = y;
y = [y, 1-y];
g = sigmoid(z);
g_tem = g;
g = [log(g) , log(1-g)];

J = -1 / m * sum(sum(y .* g)) + lambda / (2*m) * sum(theta.^(2));
for iter = 1:size(theta)
	grad(iter) = 1/m * sum((g_tem - y_tem) .* X(:,iter)) + lambda/m * theta(iter);
	end
grad(1) = 1/m * sum((g_tem - y_tem) .* X(:,1));




% =============================================================

end
