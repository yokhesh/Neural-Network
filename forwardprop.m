function [ a2,a3,pred ] = forwardprop( a1,theta1,theta2 )
a2 = a1*theta1';
a2 = 1./(1+(exp(-1.*a2)));
[ma2,na2] = size(a2);
da2 = ones(ma2,1);
a2 = [da2 a2];
a3 = a2*theta2';
a3 = 1./(1+(exp(-1.*a3)));
pred = zeros(length(a3),1);
for i = 1:length(a3)
    if a3(i,1) < 0.5
        pred(i,1) = 0;
    else
        pred(i,1) = 1;
    end
end
end

