% =========================================================================
% Title:       Student Performance - Artificial Neural Network (ANN) Regression
% Course:      Numerical Methods for Data Mining
% Author:      Chiara Giavalisco
% Description: Predictive modeling of student performance using a shallow
%               feedforward neural network, feature normalization,
%               and regression performance evaluation.
% =========================================================================

clear; close all; clc;

%% =========================================================================
% 1. Data Loading and Preparation
% =========================================================================

% Load dataset
data = readtable('Student_Performance.csv');

% Extract features and target variable
% Features: HoursStudied, PreviousScores, SleepHours, SampleQuestionPapersPracticed
% Target:   PerformanceIndex
Xd = [data.HoursStudied, ...
      data.PreviousScores, ...
      data.SleepHours, ...
      data.SampleQuestionPapersPracticed, ...
      data.PerformanceIndex];

% Z-score normalization
mu = mean(Xd);
sigma = std(Xd);
dataNormalized = (Xd - mu) ./ sigma;

% Prepare inputs (X) and targets (Y) for Neural Network toolbox
% Dimensions: Features/Targets as rows, samples as columns
X = dataNormalized(:, 1:4)';
Y = dataNormalized(:, 5)';

%% =========================================================================
% 2. Neural Network Architecture and Data Splitting
% =========================================================================

% Define hidden layer configuration
hiddenLayerSize = 20;
net = feedforwardnet(hiddenLayerSize);

% Split dataset: 70% Training, 15% Validation, 15% Testing
[trainInd, valInd, testInd] = dividerand(length(X), 0.7, 0.15, 0.15);

net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd   = valInd;
net.divideParam.testInd  = testInd;

%% =========================================================================
% 3. Model Training and Prediction
% =========================================================================

% Train the neural network
[net, tr] = train(net, X, Y);

% Predict targets on the full dataset
YPred = net(X);

%% =========================================================================
% 4. Performance Evaluation
% =========================================================================

% Root Mean Squared Error (RMSE)
rmse = sqrt(mean((YPred - Y).^2));
fprintf('\n--- Performance Metrics ---\n');
fprintf('Overall RMSE: %.4f\n', rmse);

% Mean Squared Error (MSE) via built-in perform function
performance = perform(net, Y, YPred);
disp(['Mean Squared Error (MSE): ', num2str(performance)]);

%% =========================================================================
% 5. Visualization
% =========================================================================

figure(1);
plot(X, Y, 'bo'); 
hold on;
plot(X, YPred, 'r*');
hold off;

legend('Actual', 'Predicted', 'Location', 'best');
xlabel('Predictors (Normalized)');
ylabel('Student Performance (Normalized)');
title('Neural Network Regression Results');
grid on;
