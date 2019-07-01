function [theta,cont,parada,J_history]=multiple_linear_regression_Newton(X,y,theta,k)
%--------------------------------------------------------------------------
% La funci�n multiple_linear_regression_Newton es la implementaci�n de la regresi�n
% lineal multiple haciendo uso del m�todo de Newton para minimizar la funci�n 
% de costo J.
%--------------------------------------------------------------------------
% Inputs
% X: matriz de dise�o. El n�mero de filas de la matriz es el n�mero de 
% registros del conjunto de datos y el n�mero de columnas es igual al 
% n�mero de atributos o variables explicativas (m*n).
% y: Vector de variable predictora (m*1).
% theta: punto inicial ((n+1)*1).
% k: n�mero m�ximo de iteraciones permitidas.
%--------------------------------------------------------------------------
% Outputs
% theta: vector que contiene los coeficientes de la regresi�n lineal.
% cont: N�mero de iteraciones realizadas por el algoritmo.
% parada: Norma del gradiente en el �ptimo
% J_history: Vector que contiene el falor de la funci�n objetivo J por
% iteraci�n.
%--------------------------------------------------------------------------
% Observaciones: 
% Siempre converge en dos iteraiones como se puede demostrar en la teor�a
% para el caso de funciones cuadr�ticas. La regresi�n lineal entra en estos
% casos, ya que la funci�n de costo es cuadr�tica convexa.
%--------------------------------------------------------------------------
format long
tol=1.D-5;
cont=0;
parada = 100;
m=size(X,1);
intercept=ones(m,1);
X=[intercept,X];
A=X'*X;
b=X'*y;
c=(1/2)*(y'*y);
J=@(theta,A,b,c)1/2*(theta'*A*theta)-b'*theta+c;
GJ=@(theta,A,b) A*theta-b;
H=A;
J_history = [];
while cont <= k & parada > tol
    g=GJ(theta,A,b);
    theta=theta-H\g;
    parada=norm(g);
    cont=cont+1;
    J_history(cont)=J(theta,A,b,c);
end
figure
plot(1:numel(J_history),J_history,'--b*','LineWidth',1)
grid on
xlabel('Number of Iterations')
ylabel('J(\theta)')
title('Cost Function Value per Iteration')
