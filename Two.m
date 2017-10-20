clear; close all; clc
mTrain=load('data\mcycleTrain.txt'); %Training data
whos
mTest=load('data\mcycleTest.txt'); %Test data

ytr=mTrain(:,1); xtr=mTrain(:,2); %First column is feature, second is target value.
ytest=mTest(:,1); xtest=mTest(:,2);
whos


%% ======== Plot the training data in a scatter plot =====%

 %Plot xtr along x and ytr along y, marking with Blue O's.
%plot(xtr(1:20,:), ytr(1:20,:), 'ro'); hold on; %Plot the first 20 data of features to values, marking with Red O's.

%% =======  KNN Predictor ====== %%

    %%Create a KNN predictor and plot

kvalues = [1, 2, 3, 5, 10, 50];
figure('position', [0, 0, 1280, 720]);
for i=1:length(kvalues)
    subplot(length(kvalues)/2, length(kvalues)/3, i)
    plot(xtr,ytr,'bo'); hold on;
    learner = knnRegress(kvalues(i), xtr, ytr);
    xline = [0:0.01:2]';
    yhat = predict(learner, xline);
    plot(xline, yhat, 'r');
    legend('Training data', 'kNN regression');
    title(sprintf('Plot of %iNN regression', i));
end
%yline = predict(learner, polyx(xline, 2) ); %assuming quadratic features

%figure;
%plot(xtr, ytr,'bo'); hold on; %Plot xtr along x and ytr along y, marking with Blue O's.

