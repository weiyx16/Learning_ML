function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    theta_new = zeros(3, 1);
    theta_new(1, 1) = theta(1, 1) - alpha / m * sum(X*theta - y);
    total = zeros(1, 2);
    for m_iter = 1:m
        total(1, 1) += (X(m_iter,:)*theta - y(m_iter,:)) * X(m_iter, 2);
        total(1, 2) += (X(m_iter,:)*theta - y(m_iter,:)) * X(m_iter, 3);
    end

    theta_new(2, 1) = theta(2, 1) - alpha / m * total(1,1);
    theta_new(3, 1) = theta(3, 1) - alpha / m * total(1,2);

    theta = theta_new





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
