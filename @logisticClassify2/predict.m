function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );

for i=1:size(Xte,1);
    x = sign(obj.wts(1) + obj.wts(2)*Xte(i, 1) + obj.wts(3)*Xte(i, 2));
    yhat(i) = obj.classes(ceil((x+3)/2));
end;

Yte = yhat';