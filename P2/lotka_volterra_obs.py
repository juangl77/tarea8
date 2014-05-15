
# In[930]:

#Importando librerias
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


# In[931]:

#importando datos y creando funcion interpolada
data = np.loadtxt("lotka_volterra_obs.dat")
tiempo = data[:,0]
tiempo[0] = 0
presa = data[:,1]
predador = data[:,2]
fpresa = interp1d(tiempo,presa)
fpredador = interp1d(tiempo,predador)
tiempo = np.linspace(0,0.799,1000)
dt = tiempo[1]-tiempo[0]


# In[932]:

#Definiendo funcion likelihood
def likelihood(y_obs,y_model,x_obs,x_model):
    chi1 = 0.5*sum((y_obs-y_model)**2.0)
    chi2 = 0.5*sum((x_obs-x_model)**2.0)
    lh = [-chi1,-chi2]
    return lh
    


# In[933]:

#funcion para ecuacion diferencial
def integrar(x0,y0,dt,alpha,beta,delta,gamma):
    x = x0 + dt*x0*(alpha-beta*y0)
    y = y0 - dt*y0*(gamma-delta*x0)
    return [x,y]


# In[934]:

#funcion para el modelo
def my_model(x0,y0,dt,alpha,beta,delta,gamma):
    x = [x0]
    y = [y0]
    for i in range(1,len(tiempo)):
        valores = integrar(x[i-1],y[i-1],dt,alpha,beta,delta,gamma)
        x.append(valores[0])
        y.append(valores[1])
    x = np.array(x)
    y = np.array(y)
    return x,y


# In[935]:

#Inicializando listas
alpha_w = np.empty((0))
beta_w = np.empty((0))
delta_w = np.empty((0))
gamma_w = np.empty((0))
lx_w = np.empty((0))
ly_w = np.empty((0))


#Inicializar valores aleatorios
alpha_w = np.append(alpha_w,30)
beta_w = np.append(beta_w,4)
delta_w = np.append(delta_w,2)
gamma_w = np.append(gamma_w,50)


# In[936]:

#Primera iteracion
x0 = fpresa(0)
y0 = fpredador(0)
datos = my_model(x0,y0,dt,alpha_w[0],beta_w[0],delta_w[0],gamma_w[0])
l = likelihood(fpredador(tiempo),datos[1],fpresa(tiempo),datos[0])
lx_w = np.append(lx_w, l[1])
ly_w = np.append(ly_w, l[0])


# In[937]:

#MCMC 
n_it = 1000
for i in range(n_it):
    alpha_prime = np.random.normal(alpha_w[i],0.05)
    beta_prime = np.random.normal(beta_w[i],0.05)
    delta_prime = np.random.normal(delta_w[i],0.05)
    gamma_prime = np.random.normal(gamma_w[i],0.05)
    
    datos = my_model(x0,y0,dt,alpha_w[i],beta_w[i],delta_w[i],gamma_w[i])
    datos_prime = my_model(x0,y0,dt,alpha_prime,beta_prime,delta_prime,gamma_prime)
    
    l = likelihood(fpredador(tiempo),datos[1],fpresa(tiempo),datos[0])
    l_prime = likelihood(fpredador(tiempo),datos_prime[1],fpresa(tiempo),datos_prime[0])

    
    rx = l_prime[1]/l[1]
    ry = l_prime[0]/l[0]
    if(rx<= 1.0 and ry<=1.0):
        alpha_w = np.append(alpha_w,alpha_prime)
        beta_w = np.append(beta_w,beta_prime)
        lx_w = np.append(lx_w, l_prime[1])
        delta_w = np.append(delta_w,delta_prime)
        gamma_w = np.append(gamma_w,gamma_prime)
        ly_w = np.append(ly_w, l_prime[0])
    else:
        r2 = np.random.random()
        if(r2<=np.exp(-rx) and r2<=np.exp(-ry)):
            alpha_w = np.append(alpha_w,alpha_prime)
            beta_w = np.append(beta_w,beta_prime)
            lx_w = np.append(lx_w, l_prime[1])
            delta_w = np.append(delta_w,delta_prime)
            gamma_w = np.append(gamma_w,gamma_prime)
            ly_w = np.append(ly_w, l_prime[0])
        else:
            alpha_w = np.append(alpha_w,alpha_w[i])
            beta_w = np.append(beta_w,beta_w[i])
            lx_w = np.append(lx_w, lx_w[i]) 
            delta_w = np.append(delta_w,delta_w[i])
            gamma_w = np.append(gamma_w,gamma_w[i])
            ly_w = np.append(ly_w, ly_w[i])
    
            
                


# In[938]:

#Probando
print "los parametros del modelo son: alpha = " + str(alpha_w[-1]) + " beta = " + str(beta_w[-1]) + " delta = "+str(delta_w[-1])+ " gamma = " + str(gamma_w[-1])
l = lx_w + ly_w


# In[939]:

#Encontrando incertidumbres
dbeta = beta_w[-1]/10
dalpha =alpha_w[-1]/10
ddelta = delta_w[-1]/10
dgamma = gamma_w[-1]/10
indexalpha = np.where(abs(alpha_w-alpha_w[-1])<=dalpha  )
indexbeta = np.where(abs(beta_w-beta_w[-1])<=dbeta  )
indexdelta = np.where(abs(delta_w-delta_w[-1])<=ddelta)
indexgamma = np.where(abs(gamma_w-gamma_w[-1])<=dgamma)
ialpha =  np.intersect1d(np.intersect1d(np.array(indexbeta),np.array(indexdelta)),np.array(indexgamma))
ibeta = np.intersect1d(np.intersect1d(np.array(indexalpha),np.array(indexdelta)),np.array(indexgamma))
idelta = np.intersect1d(np.intersect1d(np.array(indexbeta),np.array(indexalpha)),np.array(indexgamma))
igamma = np.intersect1d(np.intersect1d(np.array(indexbeta),np.array(indexdelta)),np.array(indexalpha))
sigma_alpha = np.mean(np.sqrt((alpha_w[ialpha]-alpha_w[-1])**2/(-2*l[ialpha]-l[-1])))
sigma_beta = np.mean(np.sqrt((beta_w[ibeta]-beta_w[-1])**2/(-2*l[ibeta]-l[-1])))
sigma_delta = np.mean(np.sqrt((delta_w[idelta]-delta_w[-1])**2/(-2*l[idelta]-l[-1])))
sigma_gamma = np.mean(np.sqrt((gamma_w[igamma]-gamma_w[-1])**2/(-2*l[igamma]-l[-1])))
print "las incertidumbres para cada parametro del modelo son: alpha " + str(sigma_alpha) + " beta " + str(sigma_beta) + " delta "+str(sigma_delta)+ " gamma = " + str(sigma_gamma)


# In[940]:

#Graficas 
from scipy.interpolate import griddata 
num = 1001
min_alpha = np.amin(alpha_w)
min_beta = np.amin(beta_w)
min_gamma = np.amin(gamma_w)
min_delta = np.amin(delta_w)

max_alpha = np.amax(alpha_w)
max_beta = np.amax(beta_w)
max_gamma = np.amax(gamma_w)
max_delta = np.amax(delta_w)

#1. Alfa - Beta
grid_alpha, grid_beta = np.mgrid[min_alpha:max_alpha:200j, min_beta:max_beta:200j]

points = np.ones((num,2))
points[:,0] = alpha_w
points[:,1] = beta_w
grid_l = griddata(points, l, (grid_alpha, grid_beta), method='cubic')

fig = plt.figure()
plt.imshow(grid_l.T, extent=(min_alpha, max_alpha, min_beta, max_beta), aspect='auto',origin='lower')
plt.savefig('alpha_beta.png')
plt.close()

#2. Alfa - Gama
grid_alpha, grid_gamma = np.mgrid[min_alpha:max_alpha:200j, min_gamma:max_gamma:200j]

points = np.ones((num,2))
points[:,0] = alpha_w
points[:,1] = gamma_w
grid_l = griddata(points, l, (grid_alpha, grid_gamma), method='cubic')

fig = plt.figure()
plt.imshow(grid_l.T, extent=(min_alpha, max_alpha, min_gamma, max_gamma), aspect='auto',origin='lower')
plt.savefig('alpha_gamma.png')
plt.close()

#3. Alfa - Delta
grid_alpha, grid_delta = np.mgrid[min_alpha:max_alpha:200j, min_delta:max_delta:200j]

points = np.ones((num,2))
points[:,0] = alpha_w
points[:,1] = delta_w
grid_l = griddata(points, l, (grid_alpha, grid_delta), method='cubic')

fig = plt.figure()
plt.imshow(grid_l.T, extent=(min_alpha, max_alpha, min_delta, max_delta), aspect='auto',origin='lower')
plt.savefig('alpha_delta.png')
plt.close()

#4. Beta - Gamma
grid_beta, grid_gamma = np.mgrid[min_beta:max_beta:200j, min_gamma:max_gamma:200j]

points = np.ones((num,2))
points[:,0] = beta_w
points[:,1] = gamma_w
grid_l = griddata(points, l, (grid_beta, grid_gamma), method='cubic')

fig = plt.figure()
plt.imshow(grid_l.T, extent=(min_beta, max_beta, min_gamma, max_gamma), aspect='auto',origin='lower')
plt.savefig('beta_gamma.png')
plt.close()

#5. Beta - Delta
grid_beta, grid_delta = np.mgrid[min_beta:max_beta:200j, min_delta:max_delta:200j]

points = np.ones((num,2))
points[:,0] = beta_w
points[:,1] = delta_w
grid_l = griddata(points, l, (grid_beta, grid_delta), method='cubic')

fig = plt.figure()
plt.imshow(grid_l.T, extent=(min_beta, max_beta, min_delta, max_delta), aspect='auto',origin='lower')
plt.savefig('beta_delta.png')
plt.close()

#6. Gamma - Delta
grid_gamma, grid_delta = np.mgrid[min_gamma:max_gamma:200j, min_delta:max_delta:200j]

points = np.ones((num,2))
points[:,0] = gamma_w
points[:,1] = delta_w
grid_l = griddata(points, l, (grid_gamma, grid_delta), method='cubic')

fig = plt.figure()
plt.imshow(grid_l.T, extent=(min_gamma, max_gamma, min_delta, max_delta), aspect='auto',origin='lower')
plt.savefig('gamma_delta.png')
plt.close()


# In[ ]:



