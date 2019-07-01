function [error] = dataset3Paramsprueba(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
error=[];
C=[0.01, 0.03, 0.1, 0.3, 1, 3, 10,30];
sigma=C;
for i=1:numel(C)
    for j=1:numel(sigma)
model = svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
pred = svmPredict(model, Xval);
error=[error mean(double(pred ~= yval))];
    end    
end
error
end