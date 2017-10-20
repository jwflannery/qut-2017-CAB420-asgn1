clear; close all; clc
mTrain=load('data\mcycleTrain.txt'); %Training data
mTest=load('data\mcycleTest.txt');
whos

ytr=mTrain(:,1); xtr=mTrain(:,2); %First column is feature, second is target value.
ytest = mTest(:, 1); xtest=mTest(:, 2);
whos

%% ======== Plot the training data in a scatter plot =====%

plot(xtr,ytr,'bo'); hold on; %Plot xtr along x and ytr along y, marking with Blue O's.
%plot(xtr(1:20,:), ytr(1:20,:), 'ro'); %Plot the first 20 data of features to values, marking with Red O's.

%% ===========Create a linear predictor and plot======== %%

Xtr = [ones(size(xtr, 1),1), xtr]; %Create a matrix Xtr with three columns: All ones, feature data, and feature data squared.

learner = linearReg(Xtr, ytr);
yhat = predict(learner, Xtr);

xline = [0:.01:2]'; %transpose:make a column vector (like training) into x value for plotting
yline = predict(learner, polyx(xline, 1) ); %assuming quadratic features

%figure;
plot(xline, yline, 'r');
legend('Training data', 'Linear regression');


%%=================  Fifth Degree Polynomial Plot ============ %%

Xtr5 = [ones(size(xtr, 1),1), xtr, xtr.^2, xtr.^3, xtr.^4, xtr.^5]; %Create a matrix Xtr with three columns: All ones, feature data, and feature data squared.

learner5 = linearReg(Xtr5, ytr);
yhat5 = predict(learner5, Xtr5);

xline5 = [0:.01:2]'; %transpose:make a column vector (like training) into x value for plotting
yline5 = predict(learner5, polyx(xline5, 5) ); %assuming quadratic features

figure;
plot(xtr,ytr,'bo'); hold on; %Plot xtr along x and ytr along y, marking with Blue O's.
plot(xline5, yline5, 'r');
ylim([-150, 100]);
legend('Training data', 'Linear regression');

%% ======== Mean Squared Error - Training ======== %%

%model = svmTrain(X, y, range(c), @(x1, x2) gaussianKernel(x1, x2, range(s)));
%predictions = svmPredict(model, Xval);

error = mean((yhat-ytr).^2 );
error5 = mean((yhat5-ytr).^2 );

%% ======== Mean Squared Error - Test ======== %%

%model = svmTrain(X, y, range(c), @(x1, x2) gaussianKernel(x1, x2, range(s)));
%predictions = svmPredict(model, Xval);
Xtest = [ones(size(xtest, 1),1), xtest];
Xtest5 = [ones(size(xtest, 1),1), xtest, xtest.^2, xtest.^3, xtest.^4, xtest.^5]; %Create a matrix Xtr with three columns: All ones, feature data, and feature data squared.

testError = mse(learner, Xtest, ytest);
testError5 = mse(learner5, Xtest5, ytest);