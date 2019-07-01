function [theta,cont,parada,J_history]=multiple_linear_regression_Newton(X,y,theta,k)
%--------------------------------------------------------------------------
% La función multiple_linear_regression_Newton es la implementación de la regresión
% lineal multiple haciendo uso del método de Newton para minimizar la función 
% de costo J.
%--------------------------------------------------------------------------
% Inputs
% X: matriz de diseño. El número de filas de la matriz es el número de 
% registros del conjunto de datos y el número de columnas es igual al 
% número de atributos o variables explicativas (m*n).
% y: Vector de variable predictora (m*1).
% theta: punto inicial ((n+1)*1).
% k: número máximo de iteraciones permitidas.
%--------------------------------------------------------------------------
% Outputs
% theta: vector que contiene los coeficientes de la regresión lineal.
% cont: Número de iteraciones realizadas por el algoritmo.
% parada: Norma del gradiente en el óptimo
% J_history: Vector que contiene el falor de la función objetivo J por
% iteración.
%--------------------------------------------------------------------------
% Observaciones: 
% Siempre converge en dos iteraiones como se puede demostrar en la teoría
% para el caso de funciones cuadráticas. La regresión lineal entra en estos
% casos, ya que la función de costo es cuadrática convexa.
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
