import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sin, cos, radians, degrees,asin
from math import pi
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# INPUT

human_hip_angle = 170
human_knee_angle = 170


# constant value 73kg, 1741mm, male

L_hw = 0.1457 # human waist length[m]
L_ht = 0.4222 # human thigh length[m]
L_hs = 0.4340 # human shank length[m]
L_rw = 0.1 # robot waist length[m]
L_rt = 0.4222 # robot thigh length[m]
L_rs = 0.35 # robot shank length[m]
L_rtw = 0.2111 # robot thigh-waist length[m]
L_rts = 0.2111 # robot thigh-shank length[m]

m_hw = 8.1541 # human waist weight[kg]
m_ht = 10.3368 # human thigh weight[kg]
m_hs = 3.7518 # human shank weight[kg]
m_rw = 5.0000 # robot waist weight[kg]
m_rt = 4.2220 # robot thigh weight[kg]
m_rs = 3.5000 # robot waist weight[kg]

G_hw = 0.3885 # human waist gravity ratio
G_ht = 0.4095 # human thigh gravity ratio
G_hs = 0.4459 # human shank gravity ratio
G_rw = 0.5 # robot waist gravity ratio
G_rt = 0.5 # robot thigh gravity ratio
G_rs = 0.5 # robot shank gravity ratio

K_w = 10000 # human waist shear stiffness[N/m]
K_t = 10000 # human thigh shear stiffness[N/m]
K_s = 5000 # human shank shear stiffness[N/m]
K_tw = 5000 # human waist torsion stiffness[N/m]
K_tt = 5000 # human thigh torsion stiffness[N/m]
K_ts = 5000 # human shank torsion stiffness[N/m]

g = 9.81 # gravitational acceleration[m/s**2]


# import data

file = open("result.txt","r")
lines = np.loadtxt(file)
selected_rows = np.where((lines[:, 0] == human_hip_angle) & (lines[:, 2] == human_knee_angle))[0][0]

q_w = radians(lines[selected_rows, 4])
q_t = radians(lines[selected_rows, 5])

q_hh = radians(human_hip_angle)
q_hk = radians(human_knee_angle)

q_s = -asin((L_rts*sin(q_hk + q_t) - L_ht*sin(q_hk) + (sin(q_hk)*(L_rtw*sin(q_hh + q_t) + L_rw*sin(q_w)))/sin(q_hh))/L_rs)
l_t = (L_rtw*sin(q_hh + q_t) + L_rw*sin(q_w))/sin(q_hh)
l_w = -(L_rw*sin(q_hh + q_w) - L_hw*sin(q_hh) + L_rtw*sin(q_t))/sin(q_hh)
l_s = -(L_rs*sin(q_hk + q_s) - L_hs*sin(q_hk) + L_rts*sin(q_t))/sin(q_hk)
q_rh = q_s + q_t + q_hh
q_rk = q_w + q_t + q_hk


# initial value of spring length

l_w_0 = L_hw - L_rw # initial value of human waist spring[m] 
l_t_0 = L_rtw # initial value of human thigh spring[m] 
l_s_0 = L_hs - L_rs # initial value of human shank spring[m] 


## calculate safety socre

Mw = K_tw*q_w
Mt = K_tt*q_t
Ms = K_ts*q_s
Fw = K_w*(l_w - l_w_0)
Ft = K_t*(l_t - l_t_0)
Fs = K_s*(l_s - l_s_0)
ADh = degrees(q_hh) - degrees(q_rh)
ADk = degrees(q_hk) - degrees(q_rk)

if(abs(Mw)>100):
    SMw = 10**(-10)
else:
    SMw = (-1/100)*abs(Mw)+1

if(abs(Mt)>100):
    SMt = 10**(-10)
else:
    SMt = (-1/100)*abs(Mt)+1

if(abs(Ms)>100):
    SMs = 10**(-10)
else:
    SMs = (-1/100)*abs(Ms)+1

if(abs(Fw)>100):
    SFw = 10**(-10)
else:
    SFw = (-1/100)*abs(Fw)+1

if(abs(Ft)>100):
    SFt = 10**(-10)
else:
    SFt = (-1/100)*abs(Ft)+1

if(abs(Fs)>100):
    SFs = 10**(-10)
else:
    SFs = (-1/100)*abs(Fs)+1

if(abs(ADh)>5):
    SAh = 10**(-10)
else:
    SAh = (-1/5)*abs(ADh)+1

if(abs(ADk)>5):
    SAk = 10**(-10)
else:
    SAk = (-1/5)*abs(ADk)+1

St  = SMw*SMt*SMs*SFw*SFt*SFs*SAh*SAk
if(St<=10**(-10)):
    Stotal = 0
else:
    Stotal = 0.1*(10 + np.log(St))

x_hw = 0
y_hw = L_hw

x_rw = 0
y_rw = L_hw - l_w

x_hh = 0
y_hh = 0

x_rh = L_rw*cos(3*pi/2 + q_w)
y_rh = L_hw - l_w + L_rw*sin(3*pi/2 + q_w)

xx_rh = l_t*cos(5*pi/2 - q_hh) + L_rtw*cos(3*pi/2 - q_hh - q_t)
yy_rh = l_t*sin(5*pi/2 - q_hh) + L_rtw*sin(3*pi/2 - q_hh - q_t)

x_hk = L_ht*cos(5*pi/2 - q_hh)
y_hk = L_ht*sin(5*pi/2 - q_hh)

x_rk = l_t*cos(5*pi/2 - q_hh) + L_rts*cos(5*pi/2 - q_hh - q_t)
y_rk = l_t*sin(5*pi/2 - q_hh) + L_rts*sin(5*pi/2 - q_hh - q_t)

xx_rk = L_ht*cos(5*pi/2 - q_hh) + (L_hs - l_s)*cos(3*pi/2 + q_hk - q_hh) + L_rs*cos(pi/2 + q_hk + q_s - q_hh)
yy_rk = L_ht*sin(5*pi/2 - q_hh) + (L_hs - l_s)*sin(3*pi/2 + q_hk - q_hh) + L_rs*sin(pi/2 + q_hk + q_s - q_hh)

x_hs = L_ht*cos(5*pi/2 - q_hh) + L_hs*cos(3*pi/2 + q_hk - q_hh)
y_hs = L_ht*sin(5*pi/2 - q_hh) + L_hs*sin(3*pi/2 + q_hk - q_hh)

x_rs = L_ht*cos(5*pi/2 - q_hh) + (L_hs - l_s)*cos(3*pi/2 + q_hk - q_hh)
y_rs = L_ht*sin(5*pi/2 - q_hh) + (L_hs - l_s)*sin(3*pi/2 + q_hk - q_hh)


# print

print('[Interaction force]')
print('-human hip angle      : ',str(round(degrees(q_hh),4)))
print('-human knee angle     : ',str(round(degrees(q_hk),4)))
print('-robot hip angle      : ',str(round(degrees(q_rh),4)))
print('-robot knee angle     : ',str(round(degrees(q_rk),4)))

print('[Interaction force]')
print('-waist(moment)     : ',str(round(Mw,4)))
print('-waist(shear force): ',str(round(Fw,4)))
print('-thigh(moment)     : ',str(round(Mt,4)))
print('-thigh(shear force): ',str(round(Ft,4)))
print('-shank(moment)      : ',str(round(Ms,4)))
print('-shank(shear force) : ',str(round(Fs,4)))

print('[Safety score]')
print('-waist(moment)        : ',str(round(SMw,4)))
print('-waist(shear force)   : ',str(round(SFw,4)))
print('-thigh(moment)        : ',str(round(SMt,4)))
print('-thigh(shear force)   : ',str(round(SFt,4)))
print('-shank(moment)        : ',str(round(SMs,4)))
print('-shank(shear force)   : ',str(round(SFs,4)))
print('-hip angle similarity : ',str(round(SAh,4)))
print('-knee angle similarity: ',str(round(SAk,4)))
print('-Total                : ',str(round(Stotal,4)))


# figure 1 - human posture prediction

plt.title('Posture')
plt.grid(color='black')
plt.xlabel('X[m]')
plt.ylabel('Y[m]')
plt.xticks(np.arange(-1.0, 1.0, 0.2))
plt.xlim([-1.0, 1.0])
plt.yticks(np.arange(-1.0, 1.0, 0.2))
plt.ylim([-1.0, 1.0])
plt.plot([x_hw, x_hh], [y_hw, y_hh], color='black', linewidth=1)
plt.plot([x_rw, x_rh], [y_rw, y_rh], color='black', linewidth=1)
plt.plot([x_hh, x_hk], [y_hh, y_hk], color='green', linewidth=1)
plt.plot([x_rh, x_rk], [y_rh, y_rk], color='green', linewidth=1)
plt.plot([x_hk, x_hs], [y_hk, y_hs], color='green', linewidth=1)
plt.plot([x_rk, x_rs], [y_rk, y_rs], color='green', linewidth=1)
plt.plot(x_hw, y_hw, 'g.', markersize=15)
plt.plot(x_hh, y_hh, 'g.', markersize=15)
plt.plot(x_hk, y_hk, 'g.', markersize=15)
plt.plot(x_hs, y_hs, 'g.', markersize=15)
plt.plot(x_rw, y_rw, 'k.', markersize=15)
plt.plot(x_rh, y_rh, 'k.', markersize=15)
plt.plot(x_rk, y_rk, 'k.', markersize=15)
plt.plot(x_rs, y_rs, 'k.', markersize=15)


# figure 2 - interaction force

fig2, ax1 = plt.subplots()
plt.title('Interaction Force')
plt.grid(color='black')
ax1.set_ylabel('moment[Nm]', color = 'red')
ax1.set_ylim([-100, 100])
ax2 = ax1.twinx()
ax2.set_ylabel('force[N]', color = 'blue')
ax2.set_ylim([-100, 100])
x1 = np.arange(6)
X1 = ['Mw', 'Fw', 'Mt', 'Ft', 'Ms', 'Fs']
Y1 = [Mw, Fw, Mt, Ft, Ms, Fs]
C1 = ['red', 'blue', 'red', 'blue', 'red', 'blue']
plt.bar(x1, Y1, color=C1)
plt.xticks(x1, X1)


# figure 3 - rader chart

df = pd.DataFrame({
'Wearing_offset': ['0'],
'SMw': [SMw],
'SFw': [SFw],
'SMt': [SMt],
'SFt': [SFt],
'SMs': [SMs],
'SFs': [SFs],
'SAh': [SAh],
'SAk': [SAk]
})


# plt.figure(figsize=(15,20))
fig3 = plt.figure()
labels = df.columns[1:]
num_labels = len(labels)
angles = [x/float(num_labels)*(2*pi) for x in range(num_labels)] ## 각 등분점
angles += angles[:1] ## 시작점으로 다시 돌아와야하므로 시작점 추가
my_palette = plt.cm.get_cmap("Set2", len(df.index))
ax = fig3.add_subplot(polar=True)

for i, row in df.iterrows():
    color = my_palette(i)
    data = df.iloc[i].drop('Wearing_offset').tolist()
    data += data[:1]    
    ax.set_theta_offset(pi / 2) ## start point
    ax.set_theta_direction(-1) ## CW/CCW
    plt.xticks(angles[:-1], labels) 
    ax.tick_params(axis='x', which='major', pad=15)
    ax.set_rlabel_position(0)
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.ylim(0.0,1.0)
    ax.plot(angles, data, color=color, linewidth=2, linestyle='solid', label=row.Wearing_offset)
    ax.fill(angles, data, color=color, alpha=0.4)     

for g in ax.yaxis.get_gridlines():  
    g.get_path()._interpolation_steps = len(labels)

spine = Spine(axes=ax,
spine_type='circle',
path=Path.unit_regular_polygon(len(labels)))
spine.set_transform(Affine2D().scale(.5).translate(.5, .5)+ax.transAxes)
ax.spines = {'polar':spine}
plt.title('Safety', x = 0, y = 1)
# plt.legend(loc=(0.9,0.9))

plt.show()