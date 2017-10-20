clear; close all; clc
mTrain=load('data\mcycleTrain.txt'); %Training data
whos
mTest=load('data\mcycleTest.txt'); %Test data

ytr=mTrain(:,1); xtr=mTrain(:,2); %First column is feature, second is target value.
ytest=mTest(:,1); xtest=mTest(:,2);
whos


%% =======  A: MSE First 20  ====== %%

kvalues = [1:100];
errors = []
Xhead = xtr(1:20);
figure;
for i=1:length(kvalues)
    learner = knnRegress(kvalues(i), Xhead, ytr);
    xline = [0:0.01:0.i]';
    yhat = predict(learner, xline);
    errors(i) = mean((yhat-ytr).^2 );
end
loglog(kvalues, errors);

%% ======= B: MSE All data ========= %%
kvalues = [1:100];
errors = []
figure;
for i=1:length(kvalues)
    learner = knnRegress(kvalues(i), xtr, ytr);
    xline = [0:0.01:0.i]';
    yhat = predict(learner, xline);
    errors(i) = mean((yhat-ytr).^2 );
end
loglog(kvalues, errors);

%% ======= C: Cross-Validation ====== %%
mse = [];
for k=1:100
    for xval=1:4,
        xline = [0:0.01:0.i]';
        iTest = randperm(80, 20);
        iTrain = setdiff(1:80, iTest);
        learner = knnRegress(k, xtr((iTrain)), ytr);
        yhat = predict(learner, xline);
        mse(k, xval) = mean((yhat-ytr).^2 );
    end;
end;
figure; plot(mean(mse));    % loglog([1:100], mse);