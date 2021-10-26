function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);
    m = length(y);
    n = length(theta);
    h = 0;
    error_vector = 0;
    change = 0;
    
    for iter = 1:num_iters
        h = X * theta;
        error_vector = h - y;
        change = (alpha/m) * (X' * error_vector);
        theta = theta - change;
        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);
    end

end

%{*********** element-wise implementation ******
  m = length(y);
  n = length(theta);
  iter = 0;
  delta = 0;
  delta0 = 0; 
  delta1 = 0; 
  temp0=0;
  temp1=0;
  change = 0;
  change1 = 0;
  J_history = 0;

  for iter = 1:num_iters
      % computing for theta0
      for j = 1:m
        change0 = (theta(1)*X(j,1) + theta(2)*X(j,2) - y(j)) * X(j,1);
        delta0 = delta0 + change0;
      end 
      temp0 = theta(1) - alpha * 1/m * delta0;
      % simultaneously computing for theta1
      for j = 1:m
        change1 = (theta(1)* X(j,1) + theta(2)*X(j,2) - y(j)) * X(j,2);
        delta1 = delta1 + change1;
      end
      temp1 = theta(2) - alpha * 1/m * delta1;
      %simultaneously updating theta
      theta(1) = temp0;
      theta(2) = temp1;
      % Save the cost J in every iteration    
      J_history(iter) = computeCost(X, y, theta);
      change0 = change1 = delta0 = delta1 = 0;
  end 
%}