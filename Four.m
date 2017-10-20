clear; close all; clc
%%
iris=load('data\iris.txt'); %Training data
pi = randperm(size(iris, 1));
Y=iris(pi,5); X=iris(pi, 1:2);

X = iris(:, [1, 2]); y = iris(:, 5);

%% ============A: Plot Iris Data==========%%
one = find(y==0);
two = find(y==1);
thr = find(y==2);

scatter(X(one, 1), X(one,2), 'g'); hold on
scatter(X(two, 1), X(two, 2), '+b' );
scatter(X(thr, 1), X(thr, 2), 'or' );

xlabel('Feature 1')
ylabel('Feature 2')

legend('Class 1', 'Class 2', 'Class 3' ); hold off;

%% ===============B: Learn 1nn===============%%
learner = knnClassify(1, X, y);
Yhat = predict(learner, X);

class2DPlot(learner, X, y)

%% =================C: Multiple K value comparisons=====================%%
kvalues = [1, 3, 10, 30];
figure;
for i = 1:length(kvalues)
    learner = knnClassify(kvalues(i), X, y);
    Yhat = predict(learner, X);
    class2DPlot(learner, X, y);
end;

%%  =============D: 80/20======================%%
kvalues = [1,2,5,10,50,100,200];
error = zeros(1, length(kvalues));
for i = 1:length(kvalues)
    %[x, y] = shuffleData(X, y);
    
    XTrain = X(1:118, :);
    YTrain = Y(1:118, :);
    learner = knnClassify(kvalues(i), XTrain, YTrain);
    
    XVal = X(119:148, :);
    YVal = Y(119:148, :);
    
    Yhat = predict(learner, XVal);
    err = Yhat ~= YVal;
    error(i) = sum(err);
end
figure; 
scatter(kvalues, error); hold on;
plot(kvalues, error);
ylim([0, 30]);
%% 