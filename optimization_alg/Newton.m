function [Xk, k] = Newton(Xo, tol, Miter)
%==========================================================================
% Newton -- Busca el cero de una funcion con el metodo de Newton.
% SINOPSIS:
%   [Xk k] = Newton(Xo, tol, Miter, nombre)
%
% DESCRIPCION:
%   Aproxima el cero de una funcion mediante el metodo de Newton para
%   funciones de R^n->R^n. El metodo parte de un punto inicial Xo con una 
%   tolerancia tol y un maximo de iteraciones Miter. 
%
% PARAMETROS:
%   Xo    - Aproximacion inicial del minimo.
%   tol      - Tolerancia de convergencia.
%   Miter      - Maximo de iteraciones antes de abortar en algoritmo.
%
% RETORNOS:
%   XK - Aproximaxion del minimo de la funcion.
%   k    - Numero de iteraciones realizadas. 
%
%   EJEMPLO:
%   Busqueda del cero de la funcion F con punto inicial Xo=[-1;3].
%   con una tolerancia de 10e-14 y un maximo de 100 iteraciones.
%   newton.txt es el archivo de salida. Se debe declarar previamente en un
%   ejecutable la funcion y su Jacobiano.
%
%       [Xk k] = Newton([-1;3],10e-14,100,'newton.txt');
%==========================================================================
X=[rand(10,1),rand(10,1),rand(10,1),rand(10,1),rand(10,1)];
y=5*rand(10,1);
e = 1;
k = 0;
while(e > tol & k < Miter)
    G=GJ(X,y,Xo);
    He=H(X);
    dk=He\-G;
    Xk = Xo + dk;
    e = norm(Xo-Xk);
    k = k+1;
    Xo = Xk;
end