#! /usr/bin/env python

'''
Este script resuelve el pendulo simple usando RK2.
'''

import numpy as np
import matplotlib.pyplot as plt


A = np.pi / 30
w = np.sqrt(10)

plt.figure(1)
plt.clf()

t = np.linspace(0, 5 * 2 * np.pi / w, 400)

plt.plot(t, A * np.sin((w * t) + np.pi/2.))
#a esta sinusoide le falta una fase, para que no parta en cero
#la funcion que se runge-kuttea no parte en cero, por eso.
#le agregué fase pi/2

def f(phi, w): #no depende explícitamente del tiempo, ojo al charqui
    return w, -10 * np.sin(phi)

def get_k1(phi_n, w_n, h, f): #esto está OK
    f_eval = f(phi_n, w_n)
    return h * f_eval[0], h * f_eval[1]

def get_k2(phi_n, w_n, h, f): #aquí eliminé los dividido en 2
    k1 = get_k1(phi_n, w_n, h, f)
    f_eval = f(phi_n + k1[0], w_n + k1[1])
    return h * f_eval[0], h * f_eval[1]

def rk2_step(phi_n, w_n, h, f): #agrego k1=get_k1 #se redefine el paso
    k1 = get_k1(phi_n, w_n, h, f)
    k2 = get_k2(phi_n, w_n, h, f)
    phi_n1 = phi_n + (k1[0] + k2[0])/2. #agrego (k1+k2)/2 en vez de lo que había antes
    w_n1 = w_n + (k1[1] + k2[1])/2.
    return phi_n1, w_n1

N_steps = 40000
h = 10. / N_steps
phi = np.zeros(N_steps)
w = np.zeros(N_steps)

phi[0] = A
w[0] = 0
for i in range(1, N_steps):
    phi[i], w[i] = rk2_step(phi[i-1], w[i-1], h, f)



t_rk = [h * i for i in range(N_steps)]

plt.plot(t_rk, phi, 'g')




plt.xlabel('Tiempo') #puse una mayúscula...
plt.ylabel('$\phi(t)$', fontsize=18)
plt.show()
plt.draw()


#Lo que hice fue cambiar la definición del método
#usé la que encontré en http://campus.usal.es/~mpg/Personales/PersonalMAGL/Docencia/MetNumTema4Teo(09-10).pdf).
