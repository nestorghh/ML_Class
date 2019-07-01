function [theta,cont]=regresion_lineal_multiple(X,y,theta,k)
format long
tol=1.D-8;
cont=1;
parada = 100;
[m,n]=size(X);
intercept=ones(m,1);
X=[intercept,X];
A=X'*X;
b=X'*y;
c=1/2*y'*y;
J=@(theta,A,b,c)1/2*(theta'*A*theta)-b'*theta+c;
GJ=@(theta,A,b) A*theta-b;
while cont<=k & parada>tol
   g=GJ(theta,A,b);
   alpha=(g'*g)/(g'*A*g);
   theta=theta-alpha*g;
   g=GJ(theta,A,b);
   parada=norm(g);
   cont=cont+1; 
end

