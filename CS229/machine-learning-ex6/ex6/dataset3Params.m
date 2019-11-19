function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

err = 1;
for C_cur = 0.6:0.2:1.4
    for sigma_cur = 0.1:0.2:0.5
        model= svmTrain(X, y, C_cur, @(x1, x2) gaussianKernel(x1, x2, sigma_cur));
        pred = svmPredict(model, Xval);
        err_cur = mean(double(pred ~= yval));
        if err_cur < err
            C=C_cur;
            sigma = sigma_cur;
            err = err_cur;
        end
        fprintf(['\n-------------------------\n'...
                'Is testing model with sigma = %f, C = %f, and error = %f \n' ...
                'For now, the best model with sigma = %f, C = %f, and error = %f \n'], 
                sigma_cur, C_cur, err_cur, sigma, C, err);
    end
end 


% =========================================================================

end
