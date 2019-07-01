function [J, X_grad,Theta_grad] = cofiCostFuncprueba(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
C=X*Theta';
J=(1/2)*sum((C(R == 1)- Y(R == 1)).^2);

for i=1:num_movies
    idx = find(R(i, :)==1);
    Thetatemp = Theta(idx, :);
    Ytemp = Y(i,idx);
    X_grad(i,:)=(X(i,:)*Thetatemp'-Ytemp)*Thetatemp;   
end

X_grad =((X* Theta'-Y).*R) * Theta;


% for j=1:num_users
% idx = find(R(:,j)==1);
% Xt = X(idx,:);
% Yt = Y(idx,j);
% Theta_grad(j, :) = (Xt *Theta-Yt);
% end
Theta_grad = ((Theta * X'-Y').*R')*X;
end