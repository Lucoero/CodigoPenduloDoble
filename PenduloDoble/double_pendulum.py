

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


#%% Datos a modificar en la simulacion 
saveImg = False
# Pendulo doble

L1 = 1      #longitud del péndulo 1 en m
L2 = 1      #longitud del pendulo 2
m1 = 1.0      #masa del pendulo 1
m2 = 1.0      #masa del pendulo 2
g = 9.8     #aceleracion de la gravedad
tf = 5.0     #tiempo de simulacion
m12 = m1 + m2
nt = 1000  #numero de intervalos de tiempo
dt = tf/nt
# Valores iniciales del péndulo doble
# Angulos iniciales en grados
theta1_0_deg = 5*np.sqrt(2)/2
theta2_0_deg = 5
# Velocidades iniciales en grados
w1_0_deg = 0
w2_0_deg = 0
"""
Cuando aumento mucho los angulos las aproximaciones lineales se caen. Hay mucha info en:
    Double pendulum: an experiment in caos 61. 1038(1993)
Al aumentar los angulos la energia cinetica se vuelve del mismo orden que la potencial. 
Tendra algo que ver con el caos?
"""
simInfo = "{}_{}_{}__{}_{}__{}_{}_{}_{}".format(L1,L2,m1,m2,tf,
                                                theta1_0_deg,theta2_0_deg,w1_0_deg,w2_0_deg)

# Angulos iniciales en radianes
theta1_0 = theta1_0_deg*np.pi/180.0
theta2_0 = theta2_0_deg*np.pi/180.0

# Velocidades iniciales en gradianes

w1_0 = w1_0_deg*np.pi/180.0
w2_0 = w2_0_deg*np.pi/180.0

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

#%% CÁLCULO DE LAS ENERGÍAS
"""
La energía mecánica será la suma de las amplitudes al cuadrado de los modos normales de oscilación
La cinética y la potencial se pueden ver con:
    1. Coordenadas normales: defino nor1 y nor2, y así las matrices T y U son diagonales
    2. Por solución sinusoidal: Por definición T = 1/2 m1*w_1**2 + ... y U....
Voy a tirar por la forma 2.
Recuerda que:
    z[:,0]: Theta1
    z[:,1]: Theta2
    z[:,3]: w1
    z[:,4]: w2
"""

T = 1/2 * m1 * z[:,2]**2 * L1**2 + 1/2 * m2 * (z[:,2]**2 * L1**2 + z[:,3]**2 * L2**2 + 2*z[:,2]*L1*z[:,3]*L2*np.cos(z[:,0]-z[:,1]))
U = -m1*g*L1*np.cos(z[:,0]) - m2*g*(L1*np.cos(z[:,0]) + L2*np.cos(z[:,1])) 
E = T + U
plt.close('all')




#%% Definicion del grafico
def GuardarImagen(fig, title):
    if saveImg: fig.savefig("Img1/" + title + ".jpg")
    return
# Aqui se grafica la evolucion de los angulos theta1 y theta 2 con el tiempo

fig, ax1 = plt.subplots()
fig.suptitle("Espacio real", fontsize = 25)
ax2 = ax1.twinx()
ax1.set_xlabel('time (s)', fontsize = 25)
ax1.set_ylabel('$\\theta_1$(rad)', color='b', fontsize=25)
ax2.set_ylabel('$\\theta_2$(rad)', color='r', fontsize=25)
ax1.tick_params('x', labelsize=25)
ax1.tick_params('y', colors='b', labelsize=25)
ax2.tick_params('y', colors='r',labelsize=25)
ax1.set_xlim(xmin=0.,xmax=tf) #limites del eje x
line1, = ax1.plot(t[:],z[:,0], linewidth=2, color='b')
line2, = ax2.plot(t[:],z[:,1], linewidth=2, color='r')
GuardarImagen(fig,"EspReal_" + simInfo)

fig, ax3 = plt.subplots()
fig.suptitle("Espacio de fases")
ax4 = ax3.twinx()
ax3.set_xlabel('$\\theta$ (rad)')
ax3.set_ylabel('$\\omega_1$(rad/s)', color='b', fontsize=10)
ax4.set_ylabel('$\\omega_2$(rad/s)', color='r', fontsize=10)
ax3.tick_params('y', colors='b')
ax4.tick_params('y', colors='r')
line1, = ax3.plot(z[:,0],z[:,2], linewidth=2, color='b')
line2, = ax4.plot(z[:,1],z[:,3], linewidth=2, color='r')

GuardarImagen(fig,"EspFases_" + simInfo)

# ENERGIAS
fig, ax3 = plt.subplots()
fig.suptitle("Energias", fontsize = 25)
ax4 = ax3.twinx()
ax3.set_xlabel('$t$ (s)',fontsize=25)
ax3.set_ylabel('$T (J)$', color = "b",fontsize=25)
ax4.set_ylabel("U (J)", color = "r",fontsize=25)
ax1.tick_params('x', labelsize=25)
ax3.tick_params('y', colors='b',labelsize=25)
ax4.tick_params("y", colors="r",labelsize=25) 
line1, = ax3.plot(t,T, linewidth=1.5, color='b')
line2, = ax4.plot(t,U, linewidth=1.5, color='r')
line3, = ax4.plot(t,E, linewidth=2, color='green', label = r"$E\simeq {}\quad J$".format(round(E[0],3)))
fig.legend(loc = (0.7,0.9), fontsize = 25)
GuardarImagen(fig,"Energias_" + simInfo)


# Aqui se hace una animacion del movimiento del péndulo doble en
# el espacio real (x,y)

Llong=(L1+L2)*1.1

fig, ax3 = plt.subplots()
fig.suptitle("Simulación del péndulo")
ax3 = plt.axes(xlim=(-Llong,Llong), ylim=(-Llong,Llong))
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')

line1,=ax3.plot([],[],lw=1, color = "black")
line2,=ax3.plot([],[],lw=1, color = "black")
line3,=ax3.plot([],[],lw=1,color = "orange",label = r"Trayectoria de $m_2$" )
line4, = ax3.plot([],[],lw=1,label = r"Trayectoria de $m_1$" )
bob1 = plt.Circle((1, 1),Llong*0.02, fc='b')
bob2 = plt.Circle((1, 1),Llong*0.02, fc='r')
fig.legend(handles = [line3,line4], loc = (0.6,0.75))
time_template = 'time = %.1fs'
time_text = ax3.text(0.05, 0.9, '', transform=ax3.transAxes)

def init():
    bob1.center = (1, 1)
    ax3.add_artist(bob1)
    bob2.center = (0,0)
    ax3.add_artist(bob2)
    line1.set_data([],[])
    line2.set_data([],[]) 
    line3.set_data([],[])
    line4.set_data([],[])
    time_text.set_text('')
    return bob1,bob2,line1,line2,line3,line4,time_text


def animate(i):
    x1, y1 = bob1.center
    x1 = L1*np.sin(z[i,0])
    y1 = -L1*np.cos(z[i,0])
    line1.set_data((0,x1),(0,y1))
    bob1.center = (x1, y1)
    x2, y2 = bob2.center
    x2 = x1+L2*np.sin(z[i,1])
    y2 = y1-L2*np.cos(z[i,1])
    line2.set_data((x1,x2),(y1,y2))
    line3.set_data(L1*np.sin(z[0:i,0])+L2*np.sin(z[0:i,1]),-L1*np.cos(z[0:i,0])-L2*np.cos(z[0:i,1]))
    line4.set_data(L1*np.sin(z[0:i,0]),-L1*np.cos(z[0:i,0]))

    bob2.center = (x2, y2)

    time_text.set_text(time_template%(i*dt))

    return bob1,bob2,time_text

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=10000,
                               interval=5)


plt.show()





