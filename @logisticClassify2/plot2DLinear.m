function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%

  [n,d] = size(X);
  weights = obj.wts

  if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;
  
  X1 = X(Y==obj.classes(1), 1);
  Y1 = X(Y==obj.classes(1), 2);
  X2 = X(Y==obj.classes(2), 1);
  Y2 = X(Y==obj.classes(2), 2);
    
plot(X1, Y1, 'bx'); hold on;
plot(X2, Y2, 'ro');

ylin = linspace(min(X(:)), max(X(:)), 99)';
yhatweights = ylin.*weights;
yhat = sum(yhatweights, 2);
plot(yhat, ylin); hold off;