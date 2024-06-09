

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
from scipy import signal
import matplotlib.animation as animation
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot



fig=plt.figure()
fig.set_dpi(100)
#fig.set_size_inches(7,6.5)
plt.close("all")

#%% Datos a modificar en la simulacion 
saveImg = False
# Pendulo doble

L1 = 1.      #longitud del péndulo 1 en m
L2 = 1.      #longitud del pendulo 2
m1 = 1.0      #masa del pendulo 1
m2 = 1.0      #masa del pendulo 2
g = 9.81      #aceleracion de la gravedad
tf = 10.0     #tiempo de simulacion
m12 = m1 + m2
nt = 1000  #numero de intervalos de tiempo
dt = tf/nt
# Valores iniciales del péndulo doble
# Angulos iniciales en grados
thetas = np.arange(0,180.5,0.5)
thetas10 = np.zeros(len(thetas))
simInfo = "EvolucionCaotico"
# Velocidades iniciales en grados
w1_0_deg = 0
w2_0_deg = 0

# Velocidades iniciales en gradianes
w1_0 = w1_0_deg*np.pi/180.0
w2_0 = w2_0_deg*np.pi/180.0

for i in range(len(thetas)):
    theta1_0_deg = thetas[i]
    theta2_0_deg = theta1_0_deg
    
    theta1_0 = theta1_0_deg*np.pi/180.0
    theta2_0 = theta2_0_deg*np.pi/180.0
    
    
    
    z0 = [theta1_0,theta2_0,w1_0,w2_0] #Valores iniciales   
    
    par=[L1,L2,m1,m2,m12,g]
    
    
    
    # Definicion de las ecuaciones de movimiento del pendulo doble
    def double_pendulum(z,t,par):
        z1,z2,z3,z4=z  
    
        sinz = np.sin(z1-z2)
        cosz = np.cos(z1-z2)
        sinz1=np.sin(z1)
        sinz2=np.sin(z2)
        z42 = z4 * z4
        z32 = z3 * z3
        coszsq = cosz*cosz
    
        dzdt=[z3,z4,
             (-m2*L1*z32*sinz*cosz+g*m2*sinz2*cosz-m2*L2*z42*sinz-m12*g*sinz1)/(L1*m12-m2*L1*coszsq),
             (m2*L2*z42*sinz*cosz+g*sinz1*cosz*m12+L1*z32*sinz*m12-g*sinz2*m12)/(L2*m12-m2*L2*coszsq)]
        return dzdt
    
    
    # Llamada a odeint que resuelve las ecuaciones de movimiento
    
    
    t=np.linspace(0,tf,nt)
    abserr = 1.0e-8
    relerr = 1.0e-6
    z=odeint(double_pendulum,z0,t,args=(par,),atol=abserr, rtol=relerr)

    thetas10[i] = z[-1,0] # Escojo el ultimo theta (el de 10 sec)
#%% Definicion del grafico
def GuardarImagen(fig, title):
    if saveImg: fig.savefig("Img1/" + title + ".jpg")
    return
# Aqui se grafica la evolucion de los angulos theta1 y theta 2 con el tiempo

fig, ax1 = plt.subplots()
fig.suptitle(r"$\Theta_1$ al pasar 10 segundos")
ax1.set_xlabel(r'$\Theta_{(1,0)}\ (grados)$')
ax1.set_ylabel('$\\theta_1(10)$ (grados)', color='b', fontsize=10)
line1 = ax1.scatter(thetas,thetas10*180/np.pi, s = 5)
line2, = ax1.plot(thetas[:], np.zeros(len(thetas)), linewidth = 1, color = "green", linestyle='dashed')
GuardarImagen(fig,simInfo)


plt.show()





