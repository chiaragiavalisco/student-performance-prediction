% =========================================================================
% Title:       Student Performance - Linear Regression Analysis
% Course:      Numerical Methods for Data Mining
% Author:      Chiara Giavalisco
% Description: Predictive modeling of student performance using multiple
%               linear regression, residual analysis, and test evaluation.
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

% % Normalization (optional)
% mu = mean(Xd); 
% sigma = std(Xd); % Standard deviation
% Xd = (Xd - mu) ./ sigma;
% Xd = Xd ./ max(Xd);

% Exploratory scatter plot matrix
figure(1);
plotmatrix(Xd);

% % Remove outliers in PerformanceIndex and create the histogram (optional)
% idx = isoutlier(Xd(:, 5));
% Xd(idx, :) = [];

% Distribution of the target variable (normally distributed)
figure(2);
histogram(Xd(:, 5));
title('Target Distribution: Performance Index');
xlabel('Performance Index');
ylabel('Count');

%% =========================================================================
% 2. Dataset Partitioning (Train / Test Split)
% =========================================================================

n = size(Xd, 1);

% Fix seed for reproducibility
rng('default');

% Partition dataset: 70% Training set, 30% Test set
c = cvpartition(height(Xd), "HoldOut", 0.3);
trainData = Xd(training(c), :);
testData  = Xd(test(c), :);

%% =========================================================================
% 3. Model Training and Diagnostic
% =========================================================================

% Fit Multiple Linear Regression Model
mdl = fitlm(trainData(:, 1:4), trainData(:, 5))

% Analysis of Variance (ANOVA)
disp('Analysis of Variance (ANOVA):');
disp(anova(mdl));

% Model diagnostic plots
figure(3);
plot(mdl);

figure(4);
plotResiduals(mdl); % Histogram of the model training residuals

% % Stepwise regression adjustments (optional)
% % Remove variables with p-value above 0.05 (none in this specific case)
% newMdl1 = removeTerms(mdl, "x1"); 
% % Improve the model by adding or removing variables
% newMdl2 = step(newMdl1, 'NSteps', 30); 
% plotResiduals(newMdl2);

%% =========================================================================
% 4. Model Testing and Evaluation
% =========================================================================

% Predict responses for the test dataset
ypred = predict(mdl, testData(:, 1:4));

% Calculate residuals on test data
errs = ypred - testData(:, 5);
% errs(isoutlier(errs, 'grubbs')); % Outlier filter on residual errors

% Plot the residual histogram of the test dataset
figure(5);
histogram(errs);
title('Histogram of Residuals - Test Data');
xlabel('Residual Error (Predicted - Actual)');
ylabel('Frequency');

%% =========================================================================
% 5. Single-Sample Prediction Verification
% =========================================================================

sampleIdx = 2350;

performance_tested    = testData(sampleIdx, 1:4);
predicted_performance = predict(mdl, performance_tested);
actual_performance    = testData(sampleIdx, 5);

fprintf('\n--- Single Sample Prediction Test (Sample #%d) ---\n', sampleIdx);
disp(['Predicted Performance: ', num2str(predicted_performance)]);
disp(['Actual Performance:    ', num2str(actual_performance)]);
