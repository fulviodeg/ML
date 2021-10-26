
clear ; close all; clc


%% ======================= Part 2: Plotting =======================
fprintf('carico db ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

%plotData(X, y);

fprintf('adding X2 = X1^2 ...\n');

fprintf('Plotting training set ...\n');
plot(data(:, 1), y, 'rx', 'MarkerSize', 10); % Plot the data
fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);
fprintf('adding ones ...\n');
X = [ones(m, 1), X]; % Add a column of ones to x
theta = zeros(size(X,2), 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1000;
alpha = 0.01;


fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, iterations);

theta
h1 = X*theta
% Plot the linear fit
hold on; % keep previous plot visible
%plot(data(:, 1), h1, '-');
legend('Training data', 'Linear regression')
%hold off % don't overlay any more plots on this figure

fprintf('Program paused. Press enter to continue.\n');
pause;

%=================================more features====================
fprintf('\nAdding a feature to see if it gets better ...\n')
X = data(:, 1);
X = [X, X.^2];
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1), X]; % Add a column of ones to x
theta = zeros(size(X,2), 1); % initialize fitting parameters
fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, iterations);
theta
%plot(data(:, 1), y, 'rx', 'MarkerSize', 10); % Plot the data
h2 = X*theta
figure(2);
plot(h2) %BHOOOOOOOO!!!
%plot(data(:, 1), h2', '-')
hold off % don't overlay any more plots on this figure



%{
% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
%}