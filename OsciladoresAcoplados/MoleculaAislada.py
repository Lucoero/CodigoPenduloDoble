#%% Librerias
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
guardarImg = False
batido = False
limitarEjes = False
energias_indiv = False
# Muelles acoplados

m = 1.0      # Valor de las masas
M = 2.0      # Valor del atomo central
k = 1    # Constante elástica de los muelles externos
kp = 0.05   # Constante elástica del muelle que une los atomos
tf = 450.0   # tiempo de simulacion
mi = 1.0/m   # Inversa de la masa

# Posiciones de equilibrio de las dos masas (del origen)
d1 = 1.0
d2 = 3.0
d3 = 2.0 

nt = 10000  #numero de intervalos de tiempo
dt = tf/nt

# Valores iniciales
z1_0 = -0.1  # Desplazamiento inicial masa 1
z2_0 = 0.0  # Desplazamiento inicila masa 2
z3_0 = 0.0  # Velocidad inicial masa 1
z4_0 = 0.0  # Velocidad inicial masa 2
z5_0 = 0.0 # Desplazamiento inicial de la masa 3
z6_0 = 0.0 # Velocidad inicial de la masa 0

simInfo = "{}_{}_{}__{}_{}__{}_{}_{}_{}".format(m,k,kp,d1,d2,z1_0,z2_0,z3_0,z4_0)

z0=[z1_0,z2_0,z3_0,z4_0,z5_0,z6_0] # Valores iniciales   

par=[mi,M,k,kp]
#%% Definicion de las ecuaciones de movimiento de los muelles acoplados
def double_pendulum(z,t,par):
    z1,z2,z3,z4,z5,z6 = z  
    dzdt=[z3,z4,
         (kp*(z1-z5))*mi,
         (kp*(z2-z5))*mi,
         z6,
         -kp/M * (2*z5 -z1-z2)
         ]
    return dzdt


#%% Llamada a odeint que resuelve las ecuaciones de movimiento
t=np.linspace(0,tf,nt)
abserr = 1.0e-8
relerr = 1.0e-6
z = odeint(double_pendulum,z0,t,args=(par,),atol=abserr, rtol=relerr)
"""
Siendo d la distancia al punto de equilibrio
z[:,0]: d de la masa 1
z[:,1]: d de la masa 2
z[:,2]: v de la masa 1
z[:,3]: v de la masa 2
z[:,4]: d de la masa 3
z[:,5]: v de la masa 3
"""
#######################
#Generate sine wave
Fs = 10000/tf

# generate frequency axis
n = np.size(t)
fr = (Fs/2)*np.linspace(0,1,n//2)



#%% Obtengamos la energia del sistema
Ec1 = 1/2 * m * z[:, 2]**2
Ec2 =  1/2 * m *z[:, 3]**2
Ec3 = 1/2 * M*z[:,5]**2
Ec = Ec1 + Ec2 + Ec3

Ep1 =  1/2 * k * z[:, 0]**2 # Energia potencial de m1 considerando solo el muelle de la pared
Ep2 =  1/2 * k * z[:, 1]**2 # Idem m2.
Ep = (1/2*kp*(z[:,0] - z[:,4])**2 + 1/2*kp*(z[:,1] - z[:,4])**2
      +1/2*kp*(2*z[:,4] - z[:,0]-z[:,1])**2)  

E1 = Ec1 + Ep1
E2 = Ec2 + Ep2
E = Ec + Ep

#%% Buscador de periodo de batido (descartado)
def BuscarBatido():
    """
    Vamos a intentar buscar las frecuencias de oscilacion a partir 
    de la determinacion de dos maximos sucesivos en la grafica de las energias.
    No va bien. Voy a hacerlo a mano. 
    """
    # Buscamos un maximo de la energia potencial
    max1 = np.max(Ep)
    global posMax1 
    posMax1 = np.where(Ep == max1)[0][0]
    
    # A partir de el, buscamos otro
    Ep2 = Ep[posMax1 + 5:] # El 5 es para asegurarme que me alejo del maximo
    max2 = np.max(Ep2)
    global posMax2 
    posMax2 = np.where(Ep2 == max2)[0][0] - 5
    # Hago la diferencia de tiempos entre uno y otro (vendra dado pos2 respecto a pos1)
    periodo = t[posMax2]
    frecPropia = 2*np.pi/periodo
    print(r"Frecuencia angular del modo simétrico:",frecPropia, "rad/s")
    return frecPropia

if batido:
    frecPropia = BuscarBatido()
#%% Definicion del grafico
plt.close("all")
def GuardarImagen(fig, title):
    if guardarImg: fig.savefig("Img2/" + title + ".jpg")
    return
#%%% Compute FFT
x1 = z[:,0]
X = fft(x1)
X_m = (2/n)*(abs(X[0:np.size(fr)]))

fig,ax2 = plt.subplots()
ax2.set_title('Magnitude Spectrum', fontsize = 25);
ax2.set_xlabel('Frequency(Hz)', fontsize=25)
ax2.set_ylabel('Magnitude', fontsize=25)
ax2.set_xlim(0.0,0.6)
ax2.tick_params(labelsize = 25)
line1, = ax2.plot(fr,X_m)
#%%% Aqui se grafica la evolucion de los desplazamientos x1 y x2 con el tiempo
fig, ax1 = plt.subplots()
fig.suptitle("Desplazamientos respecto al tiempo x1 y x2", fontsize = 25)
ax2 = ax1.twinx()
ax1.set_xlabel('time (s)',fontsize=15)
ax1.set_ylabel('x1(m)', color='b', fontsize=15)
ax2.set_ylabel('x2(m)', color='r', fontsize=15)
ax1.tick_params('y', colors='b', labelsize = 25)
ax2.tick_params('y', colors='r', labelsize = 25)
ax1.tick_params("x", labelsize = 25)
if limitarEjes:
    ax1.set_xlim(xmin=0.,xmax=270.0) #limites del eje x
line1, = ax1.plot(t[:],z[:,0], linewidth=2, color='b')
line2, = ax2.plot(t[:],z[:,1], linewidth=2, color='r')
GuardarImagen(fig, "Pos_"+ simInfo)

fig, ax1 = plt.subplots()
fig.suptitle("Desplazamientos respecto al tiempo x1 y x3", fontsize = 25)
ax2 = ax1.twinx()
ax1.set_xlabel('time (s)',fontsize=15)
ax1.set_ylabel('x1(m)', color='b', fontsize=15)
ax2.set_ylabel('x3(m)', color='r', fontsize=15)
ax1.tick_params('y', colors='b', labelsize = 25)
ax2.tick_params('y', colors='brown', labelsize = 25)
ax1.tick_params("x", labelsize = 25)
if limitarEjes:
    ax1.set_xlim(xmin=0.,xmax=270.0) #limites del eje x
line1, = ax1.plot(t[:],z[:,0], linewidth=2, color='b')
line2, = ax2.plot(t[:],z[:,4], linewidth=2, color='brown')
GuardarImagen(fig, "Pos_"+ simInfo)

fig, ax1 = plt.subplots()
fig.suptitle("Desplazamientos respecto al tiempo x2 y x3", fontsize = 25)
ax2 = ax1.twinx()
ax1.set_xlabel('time (s)',fontsize=15)
ax1.set_ylabel('x2 (m)', color='r', fontsize=15)
ax2.set_ylabel('x3 (m)', color='b', fontsize=15)
ax1.tick_params('y', colors='r', labelsize = 25)
ax2.tick_params('y', colors='b', labelsize = 25)
ax1.tick_params("x", labelsize = 25)
if limitarEjes:
    ax1.set_xlim(xmin=0.,xmax=270.0) #limites del eje x
line1, = ax1.plot(t[:],z[:,1], linewidth=2, color='r')
line2, = ax2.plot(t[:],z[:,4], linewidth=2, color='b')
GuardarImagen(fig, "Pos_"+ simInfo)
#%%% Grafico las energias
#%%%% Sistema
fig, ax4 = plt.subplots()
fig.suptitle("Energias del sistema",fontsize = 25)
ax5 = ax4.twinx()
ax4.set_xlabel('time (s)', fontsize = 15)
ax4.set_ylabel('T (J)', color='b', fontsize=15)
ax5.set_ylabel('U (J)', color='r', fontsize=15)
ax4.tick_params('y', colors='b', labelsize = 25)
ax5.tick_params('y', colors='r',labelsize = 25)
ax4.tick_params("x", labelsize = 25)
if limitarEjes:
    ax4.set_xlim(xmin=0.,xmax=270.0) #limites del eje x
line4, = ax4.plot(t[:],Ec[:], linewidth=2, color='b')
line5, = ax5.plot(t[:],Ep[:], linewidth=2, color='r')
line6, = ax4.plot(t[:],E[:], linewidth = 2, color = "green", label = "Energia")
# No hace falta legend; se explica mejor en el pie de figura.
GuardarImagen(fig, "Energias_"+ simInfo)
#%%%% M1
if energias_indiv:
    fig, ax4 = plt.subplots()
    fig.suptitle("Energias de la masa 1", fontsize = 15)
    ax5 = ax4.twinx()
    ax4.set_xlabel('time (s)',fontsize=15)
    ax4.set_ylabel('T (J)', color='b', fontsize=25)
    ax5.set_ylabel('U (J)', color='r', fontsize=25)
    ax4.tick_params('y', colors='b',labelsize = 25)
    ax5.tick_params('y', colors='r',labelsize = 25)
    ax5.tick_params("x", labelsize = 25)
    if limitarEjes:
        ax4.set_xlim(xmin=0.,xmax=290.0) #limites del eje x
    line4, = ax4.plot(t[:],Ec1[:], linewidth=2, color='b')
    line5, = ax5.plot(t[:],Ep1[:], linewidth=2, color='r')
    line6, = ax4.plot(t[:],E1[:], linewidth = 2, color = "green", label = "Energia")
    # No hace falta legend; se explica mejor en el pie de figura.
    GuardarImagen(fig, "Energias1_"+ simInfo)
    #%%%% M2
    fig, ax4 = plt.subplots()
    fig.suptitle("Energias de la masa 2", fontsize = 15)
    ax5 = ax4.twinx()
    ax4.set_xlabel('time (s)',fontsize=15)
    ax4.set_ylabel('T (J)', color='b', fontsize=25)
    ax5.set_ylabel('U (J)', color='r', fontsize=25)
    ax4.tick_params('y', colors='b',labelsize = 25)
    ax5.tick_params('y', colors='r',labelsize = 25)
    ax5.tick_params("x", labelsize = 25)
    if limitarEjes:
        ax4.set_xlim(xmin=0.,xmax=290.0) #limites del eje x
    line4, = ax4.plot(t[:],Ec2[:], linewidth=2, color='b')
    line5, = ax5.plot(t[:],Ep2[:], linewidth=2, color='r')
    line6, = ax4.plot(t[:],E2[:], linewidth = 2, color = "green", label = "Energia")
    # No hace falta legend; se explica mejor en el pie de figura.
    GuardarImagen(fig, "Energias2_"+ simInfo)

#%%% Graficamos las posiciones en un mismo grafico
fig, ax4 = plt.subplots()
fig.suptitle("Distancias de las partículas", fontsize=15)
ax1.set_xlabel('time (s)',fontsize=15)
ax1.set_ylabel('x (m)',fontsize=15)
ax1.tick_params(labelsize = 25)
line1, = ax4.plot(t,z[:,0], label = r"$x_1$")
line2, = ax4.plot(t,z[:,1], label = r"$x_2$")
line3, = ax4.plot(t,z[:,4], label = r"$x_3$")
fig.legend(fontsize = 25)
#%%% Aqui se hace una animacion del movimiento de los muelles acoplados

Long=4.0
Rbob=0.2

fig, ax3 = plt.subplots()
ax3 = plt.axes(xlim=(0.0,Long), ylim=(-0.4*Long,0.4*Long))
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')

line1,=ax3.plot([],[],lw=12,color='b')
line2,=ax3.plot([],[],lw=12,color='r')
line3,=ax3.plot([],[],lw=12,color='b') # La tengo que extender


bob1 = plt.Circle((1, 1),Rbob, fc='g')
bob2 = plt.Circle((1, 1),Rbob, fc='g')
bob3 = plt.Circle((1,1),Rbob, fc = "brown")

time_template = 'time = %.1fs'
time_text = ax3.text(0.05, 0.9, '', transform=ax3.transAxes)

def init():
    line1.set_data([],[])
    line2.set_data([],[]) 
    line3.set_data([],[]) 

    bob1.center = (-1, -1)
    ax3.add_artist(bob1)
    bob2.center = (1,1)
    ax3.add_artist(bob2)
    bob3.center = (0,0)
    ax3.add_artist(bob3)
    time_text.set_text('')
    return line1,line2,line3,bob1,bob2,time_text

def animate(i):
    x1, y1 = bob1.center
    x1 =d1+z[i,0]
    y1 =0.0


    x2, y2 = bob2.center
    x2 = d2+z[i,1]
    y2 = 0.0
    
    x3,y3 = bob3.center # Parto del instante anterior
    x3 = d3 + z[i,4] # Le añado el desplazamiento sufrido
    y3 = 0.0
    
    line1.set_data((0,x1-Rbob),(0,0))

    line2.set_data((x1+Rbob,x2-Rbob),(0,0))

    line3.set_data((x2+Rbob,3),(0,0))
   
    bob1.center = (x1, y1)
    bob2.center = (x2, y2)
    bob3.center = (x3,y3)

    time_text.set_text(time_template%(i*dt))
    return bob1,bob2,time_text

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=10000,
                               interval=5)

plt.show()





