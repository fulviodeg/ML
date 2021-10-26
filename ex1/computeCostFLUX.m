function J = computeCostFLUX(X, y, theta)

fprintf('inside the computecostFunction\n');


J = computecostVectorized(X, y, theta);
%J = computecostElementWise(X, y, theta);

fprintf('computeCostFunction END\n');



end
% =========================================================================

function J = computecostVectorized(X, y, theta)
m = length(y);

J = ( sum(((X * theta) - y ).^2) )/(2*m);

%'h' containing all of the hypothesis values - one for each training example (i.e. for each row of X)
%J1 = X * theta; 
%{
J2 = J1 - y;
J3 = J2.^2;
J4 = sum(J3);
J = J4 / (2*m);
%}
end

% =========================================================================

function J = computecostElementWise(X, y, theta)
m = length(y);
prediction = 0;
delta = 0;
theta0 = theta(1);
theta1 = theta(2);
for j = 1:m
  delta = (theta0 * X(j,1) + theta1*X(j,2) - y(j))^2;
  prediction = prediction + delta;
endfor
J = prediction/(2*m)
end