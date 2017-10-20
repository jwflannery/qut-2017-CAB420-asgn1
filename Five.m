clear; close all; clc;
iris = load('data/iris.txt');
X = iris(:, 1:2); Y=iris(:,end);
[X Y] = shuffleData(X, Y);
X = rescale(X);
XA = X(Y<2,:); YA=Y(Y<2);
XB = X(Y>0,:); YB=Y(Y>0);

%% ============= A: Scatter Plot ============= %%
%figure;
scatter(XA(:,1),XA(:,2), 'r.'); hold on;
scatter(XB(:,1),XB(:,2),'go'); hold off;
legend('Class 1', 'Class 2'); hold off;

%% ============= B: Plot2DLinear Demo ======== %%

figure;
learner = logisticClassify2();
learner = setClasses(learner, unique(YA));
wts = [0.5, 1, -.25]
learner = setWeights(learner, wts);
plot2DLinear(learner, XA, YA);

figure;
learnerB = logisticClassify2()
learnerB = setClasses(learnerB, unique(YB));
wts = [0.5, 1, -.25]
learnerB = setWeights(learnerB, wts);
plot2DLinear(learnerB, XB, YB);



%% ============= C: Predict Comparison ======= %%
figure;
yhat = predict(learner, XA)
err = yhat ~= YA;
mse =  mean((err).^2 );

plotClassify2D(learner, XA, YA);


%% ============= D: Gradient Derivation ====== %%







%% ============= E: Gradient Descent ========= %%






%% ============= F: Logistic Classifier ======= %%




%% ============================================ %%