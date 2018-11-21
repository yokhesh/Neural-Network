function [ grad,del1,gradt2,del2 ] = backprop( a3,a2,y,theta1,theta2,lambda,del1,del2,a1,m )
%error value
thou2 = a3-y;
si = a2(:,2:end).*(1-a2(:,2:end));
thou1 = (thou2*theta2(:,2:end)).*si;
%Delta value
del2 = del2+(a2'*thou2);
del1 = del1+(a1'*thou1);
%Gradient value
%gradient for theta1
grad1 = del1(1,:)/m;
grad2 = del1(2:end,:)/m;
grad3 = (lambda/m)*theta1(:,2:end);
grad4 = grad2'+grad3;
grad = [grad1' grad4];
%gradient for theta2
[ma2,na2] = size(a2);
grad1t2 = del2(1,:)/ma2;
grad2t2 = del2(2:end,:)/ma2;
grad3t2 = (lambda/ma2)*theta2(:,2:end);
grad4t2 = grad2t2'+grad3t2;
gradt2 = [grad1t2' grad4t2];
end

