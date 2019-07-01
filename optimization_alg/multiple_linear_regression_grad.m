function [theta,cont,parada,J_history]=multiple_linear_regression_grad(X,y,theta,k)
%--------------------------------------------------------------------------
% La funci�n multiple_linear_regression es la implementaci�n de la regresi�n
% lineal multiple haciendo uso del m�todo de steepest descent (m�todo de
% Cauchy) para minimizar la funci�n objetivo (suma al cuadrado de los
% errores). 
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
% 1) La longitud del paso se tom� usando b�squeda lineal exacta, esto es, 
% alpha=argmin(J(teta-alpha*g)). En el caso cuadr�tico es f�cil determinar 
% este valor. 
% 2) El criterio de parada tomado ac� consiste en que el gradiente se haga
% cercano a cero por iteraci�n, ya que en el caso cuadr�tico, es una
% condici�n suficiente de convergencia.
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
J_history = [];
while parada>tol & cont<=k
g=GJ(theta,A,b);
alpha=(g'*g)/(g'*A*g);
%alpha=7.233959339857889e-008;
theta=theta-alpha*g;
g=GJ(theta,A,b);
parada=norm(g);
cont=cont+1;
J_history(cont)=J(theta,A,b,c);
end
figure
plot(1:numel(J_history),J_history,'--r*','LineWidth',2)
grid on
xlabel('Number of Iterations')
ylabel('J(\theta)')
title('Cost Function Value per Iteration')
