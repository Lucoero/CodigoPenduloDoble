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

L = 0.5      # longitud de cada hilo en m
m = 1.0    # masa
g = 9.8     # aceleracion de la gravedad
tf = 15.0    # tiempo de simulacion
nt = 1000  #numero de intervalos de tiempo
dt = tf/nt
# Valores iniciales del péndulo doble
# Pivote en metros
x0 = 0
v0 = 0
# Angulos iniciales en grados
theta1_0_deg = 0.0001
theta2_0_deg = 0.0001
# Velocidades iniciales en grados
w1_0_deg = 0
w2_0_deg = 0

simInfo = "PropuestaPivote"
# Angulos iniciales en radianes
theta1_0 = theta1_0_deg*np.pi/180.0
theta2_0 = theta2_0_deg*np.pi/180.0

# Velocidades iniciales en gradianes

w1_0 = w1_0_deg*np.pi/180.0
w2_0 = w2_0_deg*np.pi/180.0

z0 = [theta1_0,theta2_0,w1_0,w2_0,x0,v0] #Valores iniciales   

par=[L,g]



# Definicion de las ecuaciones de movimiento del pendulo doble
def double_pendulum(z,t,par):
    z1,z2,z3,z4,z5,z6=z 
    c1 = 2-z1**2
    c2 = 2-z2**2
    c3 = z3**2*z1 + z4**2*z2
    b = g/L
    z3p = (
          (24 + c1**2*(24 + c2**2))
          /
          ((24 + c1**2)*((24 +c1**2)*(24 + c2**2)-(12+c1*c2)**2))
          * 
          (
          24*b*z1 
          -
          (12 +c1*c2)/(24+c2**2)*(12*b*z2 + 2*c2*c3)
          +
          2*c1*c3
          )
          )
    z4p = (
          (12*b*z2-(12+c2*c1)*z3p + 2*c2*c3)
          /
          (24 + c2**2)
          )
    dzdt=[z3,
          z4,
          z3p,
          z4p,
          z6,
          L/6 * (z4p*c2 + z3p*c1 -2*c3) # Por alguna razon esta acelerando constantemente con signo negativo
          ]
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
    z[:,5]: x
    z[:,6]: v
"""
"""
T = 1/2 * m1 * z[:,2]**2 * L1**2 + 1/2 * m2 * (z[:,2]**2 * L1**2 + z[:,3]**2 * L2**2 + 2*z[:,2]*L1*z[:,3]*L2*np.cos(z[:,0]-z[:,1]))
U = -m1*g*L1*np.cos(z[:,0]) - m2*g*(L1*np.cos(z[:,0]) + L2*np.cos(z[:,1])) 
E = T + U
"""

#%% Definicion del grafico
plt.close('all')
def GuardarImagen(fig, title):
    if saveImg: fig.savefig("Img1/" + title + ".jpg")
    return
# Aqui se grafica la evolucion de los angulos theta1 y theta 2 con el tiempo
"""
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

"""
# Aqui se hace una animacion del movimiento del péndulo doble en
# el espacio real (x,y)

Llong=L*1.1

fig, ax3 = plt.subplots()
fig.suptitle("Simulación del péndulo")
ax3 = plt.axes(xlim=(-Llong,Llong), ylim=(-Llong,Llong))
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')
ax3.set_ylim(ymin = -2*L - 0.1, ymax = 2*L + 0.1)
ax3.set_xlim(xmin = -2*L - 0.1, xmax = 2*L + 0.1)
line1,=ax3.plot([],[],lw=1, color = "black")
line2,=ax3.plot([],[],lw=1, color = "black")
line3,=ax3.plot([],[],lw=1,color = "orange",label = r"Trayectoria de $m_2$" )
line4, = ax3.plot([],[],lw=1,label = r"Trayectoria de $m_1$" )
line5, = ax3.plot([],[],lw = 1, color = "green",label = "Trayectoria del pivote")
bob1 = plt.Circle((1, 1),Llong*0.02, fc='b')
bob2 = plt.Circle((1, 1),Llong*0.02, fc='r')
bob3 = plt.Circle((1,1),Llong*0.02,fc = "green")
fig.legend(handles = [line3,line4,line5], loc = (0.6,0.75))
time_template = 'time = %.1fs'
time_text = ax3.text(0.05, 0.9, '', transform=ax3.transAxes)

def init():
    bob1.center = (1, 1)
    ax3.add_artist(bob1)
    bob2.center = (0,0)
    bob3.center = (0,0)
    ax3.add_artist(bob2)
    ax3.add_artist(bob3)
    line1.set_data([],[])
    line2.set_data([],[]) 
    line3.set_data([],[])
    line4.set_data([],[])
    line5.set_data([],[])
    time_text.set_text('')
    return bob1,bob2,line1,line2,line3,line4,time_text


def animate(i):
    x1, y1 = bob2.center
    x1 = L*np.sin(z[i,0])
    y1 = -L*np.cos(z[i,0])
    line1.set_data((0,x1),(0,y1))
    bob1.center = (x1, y1)
    x2, y2 = bob2.center
    x2 = x1+L*np.sin(z[i,1])
    y2 = y1-L*np.cos(z[i,1])
    line2.set_data((x1,x2),(y1,y2))
    line3.set_data(L*np.sin(z[0:i,0])+L*np.sin(z[0:i,1]),-L*np.cos(z[0:i,0])-L*np.cos(z[0:i,1]))
    line4.set_data(L*np.sin(z[0:i,0]),-L*np.cos(z[0:i,0]))
    line5.set_data(z[0:i,5],np.zeros(i))

    bob2.center = (x2, y2)

    time_text.set_text(time_template%(i*dt))

    return bob1,bob2,time_text

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=10000,
                               interval=5)


plt.show()







