# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:33:35 2018

@author: Jennifer Studer
"""

import numpy as np
import matplotlib.pyplot as plt

def Temperature(V_temp):
    T = 20/798 * V_temp # V_time in microV
    return T # T in Celcsius

# Experimental parameters
L1 = 0.05 # m
L2 = 0.05 # m
area1 = np.pi* (1.015*10**(-3))**2 # m^2
area2 = np.pi* (3.52*10**(-3))**2 # m^2
#R1 = 1200 # Ohm
R2 = 389*10**(-3) # Ohm
shunt_V = 50 # mV
shunt_A = 20 # A
#rho1 = lambda T: (1.68 + 0.38*(T-20)/60) * 10**(-8) # Ohm m
#rho2 = (44) * 10**(-8) # Ohm m
kappa1 = lambda T: -1/15*(T+23) + 401
kappa2 = lambda T: 0.0376*(T-2) + 21.9

# The measured data
T_b = np.array([30, 50, 80, 110]) # Celsius
T_b_err = 4
kappa1_err = T_b_err/15
kappa2_err = T_b_err*0.0376
V_sh_plus = np.array([[9.95, 20.09, 30.43, 40.38], [9.95, 20.08, 30.51, 40.21], [9.98, 20.07, 30.37, 39.84], [9.98, 20.27, 30.42, 40.41]]) # mV
V_sh_minus = np.array([[-9.94, -20.09, -30.44, -40.40], [-9.94, -20.07, -30.47, -40.14], [-9.97, -20.06, -30.35, -39.77], [-9.97, -20.25, -30.40, -40.38]]) # mV 
V_sh_err = 0.01
V_p_plus = np.array([[3.29, 6.67, 10.12, 13.45], [3.61, 7.05, 10.51, 13.8], [3.58, 7.19, 10.89, 14.24], [3.94, 7.71, 11.40, 15.14]]) # mV
V_p_minus= np.array([[-3.27, 6.60, -9.96, -13.17], [-3.09, -6.62, -10.17, -13.47], [-3.49, -7.04, -10.66, -13.93], [-3.47, -7.27, -11.09, -14.70]]) # mV
V_p_err = 0.01
I_T_plus = np.array([[223.6, 360.2, 508.0, 717.0], [97.9, 264.4, 736.0, 859.0], [343.1, 497.9, 713.0, 858.0], [800.0, 813.0, 850.0, 1046.0]]) # microA
I_T_minus = np.array([[29.4, -67.5, -56.8, -13.4], [-97.8, -80.9, -3.3, 40.3], [101.4, 43.1, -24.8, -58.3], [512.9, 407.7, 81.2, 40.7]]) #microA
I_T_err = 1

colors = ['r', 'fuchsia', 'b', 'g']

### Plot DT_p, DT_m, (DT_p-DT_m) and (DT_p+DT_m)
I = np.zeros_like(I_T_plus)
I_plus = np.zeros_like(I_T_plus)
I_minus = np.zeros_like(I_T_plus)
T_plus = np.zeros_like(I_T_plus)
T_minus = np.zeros_like(I_T_minus)
I_pm_err = shunt_A/shunt_V*V_sh_err 
I_err = I_pm_err/np.sqrt(2)
V_T_err = R2 * I_T_err
T_err = 20/798 * V_T_err
f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for temp in range(4):
    for val in range(4):
        I_plus[temp][val] = V_sh_plus[temp][val]/shunt_V * shunt_A # A
        I_minus[temp][val] = abs(V_sh_minus[temp][val]/shunt_V * shunt_A) # A
        I[temp][val] = (I_plus[temp][val]+I_minus[temp][val])/2
        V_T_plus = I_T_plus[temp][val] * R2 # microV
        V_T_minus = I_T_minus[temp][val] * R2
        T_plus[temp][val] = Temperature(V_T_plus)
        T_minus[temp][val] = Temperature(V_T_minus)
    plt.errorbar(I[temp], T_plus[temp], T_err, I_err, color=colors[temp], 
                 marker='o', linestyle='None', label=r"$\Delta T^+$ for {} °C".format(T_b[temp]))
    plt.errorbar(I[temp], T_minus[temp], T_err, I_err, color=colors[temp], 
                 marker='x', linestyle='None', label=r"$\Delta T^-$ for {} °C".format(T_b[temp]))
plt.xlabel('Current [A]')
plt.ylabel('Temperature [°C]')
art1 = []
lgd1 = plt.legend(bbox_to_anchor=(1.03, 1.0))
art1.append(lgd1)
plt.savefig('Report/figures/Temperature.pdf',  additional_artists=art1, bbox_inches='tight')
plt.clf()
print('I+ = ', I_plus, '+/-', I_pm_err)
print('I- = ', I_minus, '+/-', I_pm_err)
print('I = ', I, '+/-', I_err)
print('T+ = ', T_plus, '+/-', T_err)
print('T- = ', T_minus, '+/-', T_err)


Del_T_minus = T_plus - T_minus
Del_T_err = np.sqrt(2)*T_err
f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for temp in range(4):
    plt.errorbar(I[temp], Del_T_minus[temp], Del_T_err, I_err, color=colors[temp], 
                 marker='x', linestyle='None', label=r"$\Delta T^+$ - $\Delta T^-$ for {} °C".format(T_b[temp]))
plt.xlabel('Current [A]')
plt.ylabel('Temperature [°C]')
plt.legend()
plt.savefig('Report/figures/Del_T_min.pdf')
plt.clf()

Del_T_plus = T_plus + T_minus
f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for temp in range(4):
    plt.errorbar(I[temp], Del_T_plus[temp], Del_T_err, I_err, color=colors[temp], 
                 marker='x', linestyle='None', label=r"$\Delta T^+$ + $\Delta T^-$ for {} °C".format(T_b[temp]))
plt.xlabel('Current [A]')
plt.ylabel('Temperature [°C]')
art2 = []
lgd2 = plt.legend(bbox_to_anchor=(1.03, 1.0))
art2.append(lgd2)
plt.savefig('Report/figures/Del_T_max.pdf',  additional_artists=art2, bbox_inches='tight')
plt.clf()

### Calculation of the Peltier coefficient
# With equation 10 and 12
Peltier_I = np.zeros_like(I)
Peltier_I_err = np.zeros_like(I)
Peltier_V = np.zeros_like(V_p_plus)
Peltier_V_err = np.zeros_like(V_p_plus)
V_P = np.zeros_like(V_p_plus)
V_P_err = V_p_err/np.sqrt(2)
f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for temp in range(4):
    T_bath = T_b[temp]
    for val in range(4):
        Peltier_I[temp][val] = (kappa1(T_bath)*area1/L1 + 
                 kappa2(T_bath)*area2/L2) *Del_T_minus[temp][val]/(2*I[temp][val]) # V
        Peltier_I_err[temp][val] = np.sqrt((area1*Del_T_minus[temp][val]/(L1*2*I[temp][val])*kappa1_err)**2 
                     + (area2*Del_T_minus[temp][val]/(L2*2*I[temp][val])*kappa2_err)**2 
                     + ((kappa1(T_bath)*area1/L1 + kappa2(T_bath)*area2/L2)*Del_T_err/(2*I[temp][val]))**2 
                     + ((kappa1(T_bath)*area1/L1 + kappa2(T_bath)*area2/L2)*Del_T_minus[temp][val]/(2*I[temp][val]**2)*I_err)**2)
        V_P[temp][val] = (V_p_plus[temp][val]+abs(V_p_minus[temp][val]))/2 # mV
        Peltier_V[temp][val] = V_P[temp][val]/2 * Del_T_minus[temp][val]/Del_T_plus[temp][val] # mV
        Peltier_V_err[temp][val] = np.sqrt((Del_T_minus[temp][val]/(2*Del_T_plus[temp][val])*V_P_err)**2 
                     + (V_P[temp][val]/(2*Del_T_plus[temp][val])*Del_T_err)**2 
                     + (V_P[temp][val]*Del_T_minus[temp][val]/(2*Del_T_plus[temp][val]**2)*Del_T_err)**2) 
    plt.errorbar(I[temp], Peltier_I[temp]*10**3, Peltier_I_err[temp]*10**3, I_err, color=colors[temp], marker='o', 
             linestyle='None', label = "Peltier coefficient for {} °C".format(T_b[temp]))
plt.xlabel('Current [A]')
plt.ylabel('Peltier coefficients [mV]')
art3 = []
lgd3 = plt.legend(bbox_to_anchor=(1.6, 1.0))
art3.append(lgd3)
plt.savefig('Report/figures/Peltier_I.pdf',  additional_artists=art3, bbox_inches='tight')
plt.clf()
print('V_P = ', V_P, '+/-', V_P_err)
print('Peltier(I) = ', Peltier_I*10**3)
print('Its error is: ', Peltier_I_err*10**3)

f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for temp in range(4):
    plt.errorbar(V_P[temp], Peltier_V[temp], Peltier_V_err[temp], V_P_err, color=colors[temp], marker='.', 
                 linestyle='None', label = "Peltier coefficient for {} °C".format(T_b[temp]))
plt.xlabel('Voltage [mV]')
plt.ylabel('Peltier coefficients [mV]')
art4 = []
lgd4 = plt.legend(bbox_to_anchor=(1.6, 1.0))
art4.append(lgd4)
plt.savefig('Report/figures/Peltier_V_large.pdf',  additional_artists=art4, bbox_inches='tight')
plt.clf()

f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for temp in range(4):
    plt.errorbar(V_P[temp], Peltier_V[temp], Peltier_V_err[temp], V_P_err, color=colors[temp], marker='o', 
                 linestyle='None', label = "Peltier coefficient for {} °C".format(T_b[temp]))
plt.xlabel('Voltage [mV]')
plt.ylabel('Peltier coefficients [mV]')
plt.ylim(0, 10)
art5 = []
lgd5 = plt.legend(bbox_to_anchor=(1.03, 1.0))
art5.append(lgd5)
plt.savefig('Report/figures/Peltier_V.pdf',  additional_artists=art5, bbox_inches='tight')
plt.clf()
print('Peltier(V) = ', Peltier_V)
print('Its error is: ', Peltier_V_err)

### Computation of the mean Peltier coefficient and its error.
### We will compute the weighted mean.
Peltier_I_mean = []
Peltier_I_mean_err = []
Peltier_V_mean = []
Peltier_V_mean_err = []
for temp in range(4):
    Peltier_I_mean.append(sum(Peltier_I[temp]/Peltier_I_err[temp]**2)/sum(1/Peltier_I_err[temp]**2))
    Peltier_I_mean_err.append(np.sqrt(1/sum(1/Peltier_I_err[temp]**2)))
    Peltier_V_mean.append(sum(Peltier_V[temp]/Peltier_V_err[temp]**2)/sum(1/Peltier_V_err[temp]**2))
    Peltier_V_mean_err.append(np.sqrt(1/sum(1/Peltier_V_err[temp]**2)))
Peltier_I_mean = np.array(Peltier_I_mean)   
Peltier_I_mean_err = np.array(Peltier_I_mean_err) 
Peltier_V_mean = np.array(Peltier_V_mean)   
Peltier_V_mean_err = np.array(Peltier_V_mean_err) 
print('The weighted mean Peltier coefficient from I is: ', Peltier_I_mean*10**3)
print('The weighted error of the mean is: ', Peltier_I_mean_err*10**3)
print('The weighted mean Peltier coefficient from V is: ', Peltier_V_mean)
print('The weighted error of the mean is: ', Peltier_V_mean_err)

### fit
x_range = np.linspace(0, 120, 500)
poly = np.polyfit(T_b, Peltier_I_mean*10**3, 1) #Polynom of degree 3 which fits the data
fit = np.poly1d(poly) #function of the polynomial fit
print(poly)

f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.errorbar(T_b, Peltier_I_mean*10**3, Peltier_I_mean_err*10**3, 4, 'ro', label='Peltier coeffcient')
plt.plot(x_range, fit(x_range), 'k--', label="Fit of the measured datas")
plt.xlabel('Oil bath temperature [°C]')
plt.ylabel('Peltier coefficent [mV]')
plt.legend()
plt.savefig('Report/figures/coeff_I.pdf')

#Creation of the constant function x=0 to make the errors around it
f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
const = lambda x: 0*x
plt.plot(x_range, const(x_range), 'k', label="Constant zero function") 
# Difference between fit and real variables
differences1 = []
N = len(T_b)
for i in range(N):
    differences1.append(fit(T_b[i]) - Peltier_I_mean[i]*10**3)
plt.plot(T_b, differences1, 'rx', label="theory - measurement points")
plt.xlabel('Oil bath temperature [°C]')
plt.ylabel('Peltier coefficent [mV]')
plt.legend()
plt.savefig('Report/figures/coeff_I_res.pdf')


poly2 = np.polyfit(T_b, Peltier_V_mean, 1) #Polynom of degree 3 which fits the data
fit2 = np.poly1d(poly2) #function of the polynomial fit
print(poly2)

f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.errorbar(T_b, Peltier_V_mean, Peltier_V_mean_err, 4, 'ro', label='Peltier coeffcient')
plt.plot(x_range, fit2(x_range), 'k--', label="Fit of the measured datas")
plt.xlabel('Oil bath temperature [°C]')
plt.ylabel('Peltier coefficent [mV]')
plt.legend()
plt.savefig('Report/figures/coeff_V.pdf')

#Creation of the constant function x=0 to make the errors around it
f = plt.figure()
ax = f.add_subplot(111)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.plot(x_range, const(x_range), 'k', label="Constant zero function") 
# Difference between fit and real variables
differences2 = []
N = len(T_b)
for i in range(N):
    differences2.append(fit2(T_b[i]) - Peltier_V_mean[i])
plt.plot(T_b, differences2, 'rx', label="theory - measurement points")
plt.xlabel('Oil bath temperature [°C]')
plt.ylabel('Peltier coefficent [mV]')
plt.legend()
plt.savefig('Report/figures/coeff_V_res.pdf')