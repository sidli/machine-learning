function p4_svm()
EPS =0.0001;
sup_vecs = zeros(10,4);counter=1;
samples = importdata('../mystery.data', ',');
par_vec = solver(samples);
w = par_vec(1:end-1);
b = par_vec(end);
[m,n]=size(samples);
% verify and get the support vectors
for i =1:m
    x_i = samples(i,1:(n-1));
    y_i = samples(i,n);
    f_vec = phi(x_i);
    result  = f_vec*w+b;
    if abs(result * y_i -1)<EPS
        sup_vecs(counter,:)=x_i;
        counter = counter +1;
    end
    if result*y_i<0.999
        warn('error classification found!')
    end
end
fprintf('Final weight is\n');
disp(w);
fprintf('Final bias is %f \n\n',b);
fprintf('Margin is %f \n\n',1.0/sqrt(w'*w));
fprintf('The support vectors\n');
% output as latex input
[m,n]=size(sup_vecs);
for i = 1:m
    fprintf('&(%f',sup_vecs(i,1));
    for j=2:n
        fprintf(',%f',sup_vecs(i,j));
    end
    fprintf(') \\\\ \n');
end
end


function ret = solver(samples)
[m,~]=size(samples);
n = 15;
nf=14;
N_Ftr =4;
t = ones(1,n);
t(end)=0;
H = diag(t);
f=zeros(1,n);
A = zeros(m,n);
A(:,end) =1;
for i=1:m
    A(i,1:nf)=phi(samples(i,1:N_Ftr)); %A[i] = [x^(i),1]
    A(i,:)=A(i,:)*samples(i,N_Ftr+1); % multiply y^(i)
end
b = ones(m,1);
A=-A;b=-b;

options = optimset('Algorithm','active-set');
x = quadprog(H,f,A,b,[],[],[],[],[],options);
ret=x;
end

function f_vec = phi(x)
f_vec = zeros(1,14);
f_vec(1) = x(1)^2;
f_vec(2) = x(2)^2;
f_vec(3) = x(3)^2;
f_vec(4) = x(4)^2;
f_vec(5) = x(1)*x(2);
f_vec(6) = x(1)*x(3);
f_vec(7) = x(1)*x(4);
f_vec(8) = x(2)*x(3);
f_vec(9) = x(2)*x(4);
f_vec(10) = x(3)*x(4);
f_vec(11) = x(1);
f_vec(12) = x(2);
f_vec(13) = x(3);
f_vec(14) = x(4);
end