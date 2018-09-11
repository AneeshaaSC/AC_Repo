data=csvread('train_modified.csv');
X=data(:,2:8);
y=data(:,1);
[m,n]=size(X);
Xnorm=featurenormalize(X);

X=[ones(m,1) Xnorm];
initial_theta=zeros(n+1,1);
size(X)
size(initial_theta)
[cost, grad] = costf(initial_theta, X, y);

fprintf('\nCost at initial_theta : %f\n', cost);
fprintf('Gradient at initial_theta: \n');
fprintf(' %f \n', grad);

test_theta=[1 0.4 0.89 -3 -0.22 0.7 1.2 5]';
[cost, grad] = costf(test_theta, X, y);

fprintf('\nCost at test theta: %f\n', cost);
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);

options = optimset('GradObj', 'on', 'MaxIter', 500);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costf(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);

fprintf('theta: \n');
fprintf(' %f \n', theta);

%------------------------------------------------

% operate on the test set now
fprintf('Loading test data \n');
testdata=csvread('test_modified.csv');
Xtest=testdata(:,1:7);




fprintf('normalize test data\n');
X_test=featurenormalize(Xtest);
[p,q]=size(Xtest);


Xtest=[ones(p,1) Xtest];

fprintf('predicting\n');
z=Xtest*theta;
prd=sig(z);
ytest=zeros(size(prd));
for i=1:length(prd)
  if prd(i)>=0.5
    ytest(i)=1;
  elseif prd(i)<0.5
    ytest(i)=0;
  endif;
end;

fprintf('output\n');
length_of_test=length(testdata(:,1));
passid=[892:1:(892+length_of_test)-1];
output=[passid' ytest] ;
csvwrite ('survivors.csv', output);

