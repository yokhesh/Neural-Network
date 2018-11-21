function [ J,r ] = cost( a3,theta2,y,m,lambda )
h = log(a3);
j1 = y.*h;
j2 = sum(j1); j3 = sum(j2');
j4 = (1-y).*log(1-a3);
j5 = sum(j4); j6 = sum(j5');j6 = j3+j6;
Jwr = j6/(-1*m);
t_r = theta2(:,2:end);
t_r = t_r.^2;
t_r1 = sum(t_r);t_r2 = sum(t_r1');
reg = (lambda*t_r2)/(2*m);
J = Jwr+reg;

end

