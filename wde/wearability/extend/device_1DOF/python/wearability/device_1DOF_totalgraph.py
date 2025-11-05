import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos, radians, degrees, pi
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from scipy.io import loadmat

import matplotlib
matplotlib.use('agg')

# Clear console, variables, and close all plots (clc, clear, close all)
import sys
from io import BytesIO

CSV_PATH = os.path.dirname(os.path.realpath(__file__))

input_df = pd.read_csv(CSV_PATH +'/input.csv')
input_angle = np.round(input_df['Angle'])
input_angle[(input_angle > 179)] = 179
input_time  = np.round(input_df['Timer'],4)
input_length = len(input_angle)

start_input_time = min(input_time)
end_input_time = max(input_time)

duration_time = end_input_time - start_input_time
duration_matrix = input_time  - start_input_time

# offset
offset_value = 0.01
offset_matrix = np.array([[0, 0], [+offset_value, +offset_value], [+offset_value, -offset_value], [-offset_value, +offset_value], [-offset_value, -offset_value]])


def case(m):
    
    # constant value 73kg, 1741mm, male

    L_h_u = 0.2817 # human upper arm length[m]
    L_h_f = 0.2689 # human forearm length[m]
    L_d_u = 0.2 # device upper arm length[m]
    L_d_f = 0.2 # device forarm length[m]

    m_h_u = 1.9783 # human upper arm weight[kg]
    m_h_f = 1.1826 # human forearm weight[kg]
    m_d_u = 2 # device upper arm weight[kg]
    m_d_f = 2 # device forearm weight[kg]

    G_h_u = 0.4228 # human upper arm gravity ratio
    G_h_f = 0.4574 # human forearm gravity ratio
    G_d_u = 0.5 # robot upper arm gravity ratio
    G_d_f = 0.5 # robot forearm gravity ratio

    K_u = 10000 # human upper arm shear stiffness[N/m]
    K_f = 10000 # human forearm shear stiffness[N/m]
    K_t_u = 10000 # human upper arm torsion stiffness[N/m]
    K_t_f = 10000 # human forearm torsion stiffness[N/m]
    q_u_0 = 0 # initial value of human upper arm deg[deg]
    q_f_0 = 0 # initial value of human forearm deg[deg]

    g = 9.81 # gravitational acceleration[m/s^2]

    # Generate file paths
    ca = f'case{m - 1}.txt'
    sc = f'score{m - 1}.txt'

    # Open and write to the 'score.txt' file
    ID= open(sc,"w+")
    a = 0

    # Read data from the 'case?.txt' file
    file = open(CSV_PATH +'/' + ca,"r")
    G = np.loadtxt(file)

    # offset
    offset = offset_matrix[m - 1, :]
    l_s_f_offset = offset[0]
    l_s_u_offset = offset[1]
    l_s_u_0 = L_h_u - L_d_u + l_s_u_offset  # initial value of human upper arm spring[m] 
    l_s_f_0 = L_h_f - L_d_f + l_s_f_offset  # initial value of human forearm srping[m] 


    for n in input_angle:

        device_baselink_angle = 90
        device_elbow_angle = n

        q_u = radians(G[np.where((G[:, 0] == device_baselink_angle) & (G[:, 1] == device_elbow_angle))[0][0], 3])
        q_f = radians(G[np.where((G[:, 0] == device_baselink_angle) & (G[:, 1] == device_elbow_angle))[0][0], 4])

        d_d = radians(device_elbow_angle)       

        l_s_u = (L_h_u*sin(q_f - d_d + q_u) + L_d_f*sin(q_f) + L_d_u*sin(d_d - q_f))/sin(q_f - d_d + q_u)
        l_s_f = (L_h_f*sin(q_f - d_d + q_u) + L_d_u*sin(q_u) + L_d_f*sin(d_d - q_u))/sin(q_f - d_d + q_u)

        h_d = d_d - q_u - q_f

        baselink_angle = radians(device_baselink_angle)

        # calculate safety socre

        Mu = K_t_u*q_u
        Mf = K_t_f*q_f
        Fu = K_u*(l_s_u - l_s_u_0)
        Ff = K_u*(l_s_f - l_s_f_0)
        AD = degrees(h_d) - degrees(d_d)

        maximum_Fu = 150
        maximum_Mu = 100
        maximum_Ff = 150
        maximum_Mf = 100
        maximum_AD = 5

        maximum_F = max([maximum_Fu, maximum_Ff])
        maximum_M = max([maximum_Mu, maximum_Mf])

        if(abs(Fu)>maximum_Fu):
            SFu = 10**(-10)
        else:
            SFu = (-1/maximum_Fu)*abs(Fu)+1

        if(abs(Mu)>maximum_Mu):
            SMu = 10**(-10)
        else:
            SMu = (-1/maximum_Mu)*abs(Mu)+1

        if(abs(Ff)>maximum_Ff):
            SFf = 10**(-10)
        else:
            SFf = (-1/maximum_Ff)*abs(Ff)+1

        if(abs(Mf)>maximum_Mf):
            SMf = 10**(-10)
        else:
            SMf = (-1/maximum_Mf)*abs(Mf)+1

        if(abs(AD)>maximum_AD):
            SA = 10**(-10)
        else:
            SA = (-1/maximum_AD)*abs(AD)+1

        St  = SFu*SMu*SFf*SMf*SA
        if(St<=10**(-10)):
            Stotal = 0
        else:
            Stotal = 0.1*(10 + np.log(St))

        ID.write(f'{Stotal}\n')
    ID.close()

def G():
    G0 = np.loadtxt(CSV_PATH + '/score0.txt')
    G1 = np.loadtxt(CSV_PATH + '/score1.txt')
    G2 = np.loadtxt(CSV_PATH + '/score2.txt')
    G3 = np.loadtxt(CSV_PATH + '/score3.txt')
    G4 = np.loadtxt(CSV_PATH + '/score4.txt')

    All = [G0, G1, G2, G3, G4]

    # Calculate mean scores
    total_score_case0 = np.mean(G0)
    total_score_case1 = np.mean(G1)
    total_score_case2 = np.mean(G2)
    total_score_case3 = np.mean(G3)
    total_score_case4 = np.mean(G4)

    # Calculate the total score
    total_scores = [total_score_case0, total_score_case1, total_score_case2, total_score_case3, total_score_case4]
    total_score = np.mean(total_scores)
    minimum_value = np.argmin(total_scores)

    # Create a data table and save it to a CSV file
    data = {
        'Time': duration_matrix ,
        'Angle': input_angle,
        'Score': All[minimum_value]
    }
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH + '/output_expression.csv', index=False)

    # fig4 text

    txtc0 = 'safety score case0: ' + str(round(total_score_case0,4)) 
    txtc1 = 'safety score case1: ' + str(round(total_score_case1,4))
    txtc2 = 'safety score case2: ' + str(round(total_score_case2,4))
    txtc3 = 'safety score case3: ' + str(round(total_score_case3,4))
    txtc4 = 'safety score case4: ' + str(round(total_score_case4,4))
    txtto = 'total safety score : ' + str(round(total_score,4))
    txtwc = 'worst case : ' + str(minimum_value)

    # figure 4
    fig4 = plt.figure()
    plt.title('Total Safety Score')
    plt.xlabel('time[sec]')
    plt.ylabel('safety score')
    plt.xticks(range(math.ceil(duration_time) + 1))
    plt.xlim(0, math.ceil(duration_time))
    plt.yticks(np.arange(0, 1.2, 0.1))
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.plot(duration_matrix , G0, color='black', label='Solid', linewidth=2)
    plt.plot(duration_matrix , G1, color='red', label='Solid', linewidth=2)
    plt.plot(duration_matrix , G2, color='green', label='Solid', linewidth=2)
    plt.plot(duration_matrix , G3, color='blue', label='Solid', linewidth=2)
    plt.plot(duration_matrix , G4, color='magenta', label='Solid', linewidth=2)
    plt.text(0.1,0.26,txtc0)
    plt.text(0.1,0.21,txtc1)
    plt.text(0.1,0.16,txtc2)
    plt.text(0.1,0.11,txtc3)
    plt.text(0.1,0.06,txtc4)
    plt.text(0.1,0.01,txtto, weight='bold')
    plt.text(0.1,0.31,txtwc, color='red')
    
    plt.legend(('case0','case1','case2','case3','case4'),loc='lower right')
    # plt.savefig(CSV_PATH + '/total_safety.jpg')    
    # plt.show()

    totalgraph_img = BytesIO()
    plt.savefig(totalgraph_img, format='png', dpi=72)
    plt.clf()
    totalgraph_img.seek(0)

    return totalgraph_img

def totalgraph():
    for m in range(1,6):
        case(m)
    return G()

def totalscore():
    for m in range(1,6):
        case(m)

    G0 = np.loadtxt(CSV_PATH + '/score0.txt')
    G1 = np.loadtxt(CSV_PATH + '/score1.txt')
    G2 = np.loadtxt(CSV_PATH + '/score2.txt')
    G3 = np.loadtxt(CSV_PATH + '/score3.txt')
    G4 = np.loadtxt(CSV_PATH + '/score4.txt')

    All = [G0, G1, G2, G3, G4]

    # Calculate mean scores
    total_score_case0 = np.mean(G0)
    total_score_case1 = np.mean(G1)
    total_score_case2 = np.mean(G2)
    total_score_case3 = np.mean(G3)
    total_score_case4 = np.mean(G4)

    # Calculate the total score
    total_scores = [total_score_case0, total_score_case1, total_score_case2, total_score_case3, total_score_case4]
    total_score = np.mean(total_scores)
    minimum_value = np.argmin(total_scores)

    # Create a data table and save it to a CSV file
    data = {
        'Time': duration_matrix ,
        'Angle': input_angle,
        'Score': All[minimum_value]
    }
    df = pd.DataFrame(data)
    df.to_csv(CSV_PATH + '/output_expression.csv', index=False)

    return round(total_score,4)

# totalgraph()
