%train correct
lambda = 1;
train_correct = load('training_correct.txt');
test_correct = load('testing_correct.txt');
train_faulty = load ('training_faulty.txt');
test_faulty = load('testing_faulty.txt');
test_faulty1 = (test_faulty(:,4)*180)/pi;test_correct1 = (test_correct(:,4)*180)/pi;
train_faulty1 = (train_faulty(:,4)*180)/pi;train_correct1 = (train_correct(:,4)*180)/pi;

train_faulty = [train_faulty(:,2:3) train_faulty1 train_faulty(:,5:6)];
train_correct = [train_correct(:,2:3) train_correct1 train_correct(:,5:6)];
test_faulty = [test_faulty(:,2:3) test_faulty1 test_faulty(:,5:6)];
test_correct = [test_correct(:,2:3) test_correct1 test_correct(:,5:6)];
training = [train_correct;train_faulty];[mtr,ntr] = size(train_correct);[ptr,qtr] = size(train_faulty);

testing = [test_correct;test_faulty];[mte,nte] = size(test_correct);[pte,qte] = size(test_faulty);

%training data preprocess
mean_train = mean(training');
var_train = var(training');
st_train = std(training');
rms_train = rms(training');
p2p_train = peak2peak(training');
rsq_train = rssq(training');
max_train = max(training');
sum_train = sum(training');
training = [mean_train;st_train;var_train;rms_train;p2p_train;rsq_train;max_train;sum_train]';
m = size(training, 1);
n = size(training, 2);
d = ones(m,1);
training = [d training];

%testing data preprocess
mean_test = mean(testing');
st_test = std(testing');
var_test = var(testing');
rms_test = rms(testing');
p2p_test = peak2peak(testing');
rsq_test = rssq(testing');
sum_test = sum(testing');
max_test = max(testing');
testing = [mean_test;st_test;var_test;rms_test;p2p_test;rsq_test;max_test;sum_test]';
mt = size(testing, 1);
dt = ones(mt,1);
testing = [dt testing];

training_labels = [ones((size(train_correct,1)),1);zeros((size(train_faulty,1)),1)];
testing_labels = [ones((size(test_correct,1)),1);zeros((size(test_faulty,1)),1)];
a1 = training;
%theta initialization
in_layer = n; 
hid_layer = 4;
% hid_layer = (2/5)*in_layer; hid_layer = fix(hid_layer);
out_layer = 1;
epsilon_init = 0.000001;
% theta1 = rand(hid_layer, 1 + in_layer) * 2 * epsilon_init - epsilon_init;
% theta2 = rand(out_layer, 1 + hid_layer) * 2 * epsilon_init - epsilon_init;
theta1 = zeros(hid_layer, 1 + in_layer);theta2 = zeros(out_layer, 1 + hid_layer);
del1 = 0;del2 = 0;

J = 100;
y = training_labels;
ini = theta1;count = 0;
while J>0.0001  && ~isnan(J)
%cost function
[a2,a3,pred] = forwardprop(a1,theta1,theta2);
J = cost(a3,theta2,y,m,lambda);
count = count+1;
%backpropagation
% alpha = 0.0001;
alpha = 0.0001;
[grad,del1,gradt2,del2] = backprop(a3,a2,y,theta1,theta2,lambda,del1,del2,a1,m);
if ~isnan(J)
theta1 = theta1 - (alpha.*(grad));
theta2 = theta2 - (alpha.*(gradt2));
end
end
%testing using the test set
ytest = [ones((size(test_correct,1)),1);zeros((size(test_faulty,1)),1)];
[a2,a3,pred] = forwardprop(testing,theta1,theta2);
fo = confusionmat(pred,ytest);
accuracy = (fo(1,1)+fo(2,2))/mt