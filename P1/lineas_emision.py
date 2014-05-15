import numpy as np
import matplotlib.pyplot as plt
from math import *

#Se cargan los datos de las observaciones
datos=np.loadtxt("energy_counts.dat")
sh=np.shape(datos)
n_ener_datos=datos[:,0]
n_cuen_datos=datos[:,1]

#Funcion Likelihood sin exponencial. En su forma original da siempre cero
def Likelihood(n_cuen_datos, n_cuen_modelo):
    chi_cuadrado = sum((n_cuen_datos-n_cuen_modelo)**2)
    return chi_cuadrado

#Modelo fisico
def mi_modelo(n_ener_datos, A, B, Eo, sigma, alpha):
    return ( A * (n_ener_datos**alpha) ) + ( B * np.exp(-( (n_ener_datos-Eo) / (np.sqrt(2)*sigma) )**2) )

#Listas vacias que guardan los pasos
paso_A = np.empty((0)) 
paso_B = np.empty((0))
paso_Eo = np.empty((0))
paso_sigma = np.empty((0))
paso_alpha = np.empty((0))
paso_L = np.empty((0))

#Inicializacion de pasos. Los valores se estimaron tras correr varias veces el programa
A_o = 10**16
paso_A = np.append(paso_A, A_o)
B_o = 10**3
paso_B = np.append(paso_B, B_o)
Eo_o = 1.387*10**3
paso_Eo = np.append(paso_Eo, Eo_o)
sigma_o = 0
paso_sigma = np.append(paso_sigma, sigma_o)
alpha_o = 10**2
paso_alpha = np.append(paso_alpha, alpha_o)

n_cuen_inicial = mi_modelo(n_ener_datos, paso_A[0], paso_B[0], paso_Eo[0], paso_sigma[0], paso_alpha[0])
paso_L = np.append(paso_L, Likelihood(n_cuen_datos, n_cuen_inicial))

n_iteraciones = 100000 

for i in range(n_iteraciones-1):
    #Se define el paso de cada variable como 3 ordenes de magnitud menos que el valor de inicializacion
    A_prima = np.random.normal(paso_A[i], 10**13) 
    B_prima = np.random.normal(paso_B[i], 10**0)
    Eo_prima = np.random.normal(paso_Eo[i], 10**0)
    sigma_prima = np.random.normal(paso_sigma[i], 10**-2)
    alpha_prima = np.random.normal(paso_alpha[i], 10**-1)
    
    n_cuen_inicial = mi_modelo(n_ener_datos, paso_A[i], paso_B[i], paso_Eo[i], paso_sigma[i], paso_alpha[i])
    n_cuen_prima = mi_modelo(n_ener_datos, A_prima, B_prima, Eo_prima, sigma_prima, alpha_prima)
    
    L_prima = Likelihood(n_cuen_datos, n_cuen_prima)
    L_inicial = Likelihood(n_cuen_datos, n_cuen_inicial)
    
    Alpha = L_inicial-L_prima
    
    if(Alpha>0.0):
        paso_A  = np.append(paso_A,A_prima)
        paso_B  = np.append(paso_B,B_prima)
        paso_Eo  = np.append(paso_Eo,Eo_prima)
        paso_sigma  = np.append(paso_sigma,sigma_prima)
        paso_alpha  = np.append(paso_alpha,alpha_prima)
        paso_L  = np.append(paso_L,L_prima)

    else:
        Beta = np.random.random()
        if(np.log(Beta)<=Alpha):
            paso_A  = np.append(paso_A,A_prima)
            paso_B  = np.append(paso_B,B_prima)
            paso_Eo  = np.append(paso_Eo,Eo_prima)
            paso_sigma  = np.append(paso_sigma,sigma_prima)
            paso_alpha  = np.append(paso_alpha,alpha_prima)
            paso_L  = np.append(paso_L,L_prima)
            
        else:
            paso_A  = np.append(paso_A,paso_A[i])
            paso_B  = np.append(paso_B,paso_B[i])
            paso_Eo  = np.append(paso_Eo,paso_Eo[i])
            paso_sigma  = np.append(paso_sigma,paso_sigma[i])
            paso_alpha  = np.append(paso_alpha,paso_alpha[i])
            paso_L  = np.append(paso_L,L_inicial)

max_Likelihood_id = np.argmin(paso_L)
mejor_A = paso_A[max_Likelihood_id]
mejor_B = paso_B[max_Likelihood_id]
mejor_Eo = paso_Eo[max_Likelihood_id]
mejor_sigma = paso_sigma[max_Likelihood_id]
mejor_alpha = paso_alpha[max_Likelihood_id]

print "El valor de A es", mejor_A
print "El valor de B es", mejor_B
print "El valor de E_0 es", mejor_Eo
print "El valor de sigma es", mejor_sigma
print "El valor de alpha es", mejor_alpha

plt.scatter(n_ener_datos,n_cuen_datos)
plt.plot(n_ener_datos,mi_modelo(n_ener_datos, mejor_A, mejor_B, mejor_Eo, mejor_sigma, mejor_alpha))
plt.show()

count, bins, ignored =plt.hist(paso_A, 20, normed=True)
plt.show()
count, bins, ignored =plt.hist(paso_B, 20, normed=True)
plt.show()
count, bins, ignored =plt.hist(paso_Eo, 20, normed=True)
plt.show()
count, bins, ignored =plt.hist(paso_alpha, 20, normed=True)
plt.show()
count, bins, ignored =plt.hist(paso_sigma, 20, normed=True)
plt.show()
