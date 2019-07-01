function [theta,cont,parada,J_history]=multiple_linear_regression_spect(X,y,theta,k)
%--------------------------------------------------------------------------
% La funci�n multiple_linear_regression_spect es la implementaci�n de la regresi�n
% lineal multiple haciendo uso del m�todo del gradiente espectral (longitud del paso
% de Barzilai & Borwein) para minimizar la funci�n de costo J.
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
% 1) La longitud del paso es la longitud espectral propuesta por Barzilai & 
% Borwein.
% 2) El criterio de parada tomado ac� consiste en que el gradiente se haga
% cercano a cero por iteraci�n, ya que en el caso cuadr�tico, es una
% condici�n suficiente de convergencia.
%--------------------------------------------------------------------------
format long
tol=1.D-5;
cont=0;
lp=1;
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
while cont <= k & parada > tol
    gk=GJ(theta,A,b);
    h=A*gk;
    thetas=theta-lp*gk;
    lp =(gk'*gk)/(gk'*h);
    parada=norm(gk);
    cont=cont+1;
    theta=thetas;
    J_history(cont)=J(theta,A,b,c);
end
figure
plot(1:numel(J_history),J_history,'--b*','LineWidth',1)
grid on
xlabel('Number of Iterations')
ylabel('J(\theta)')
title('Cost Function Value per Iteration')
