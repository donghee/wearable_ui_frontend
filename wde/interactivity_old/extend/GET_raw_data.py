import serial
import pandas as pd
import matplotlib.pyplot as plt
import xlwings as xw
from openpyxl import load_workbook
import os
import keyboard

def save_to_csv(dataframe, filepath):
    mode = 'a' if os.path.isfile(filepath) else 'w'
    dataframe.to_csv(filepath, mode=mode, index=False, header=not os.path.isfile(filepath))

# Define and open serial port settings
port = 'COM3'
baudrate = 9600
ser = serial.Serial(port, baudrate)

#################  File 이름 변경  #####################
file_path = 'result_without_human.csv'
########################################################


df = pd.DataFrame(columns=['Time', 'State','Repeat','Weight','Angle','Current'])
df.to_csv(file_path)

# Initialize data buffer
data_buffer = []

# Start reading data from serial port
while True:
    # Read data from serial port
    data = ser.readline().decode().strip()
    data2 = data.split(',')
    print(data2)
    
    try:
        # Convert data to float
        state = str(data2[0])
        repeat = float(data2[1])
        weight = float(data2[2])
        angle = float(data2[3])
        current = float(data2[4])
          
        # Get current time
        time = pd.Timestamp.now()
        # Add data to buffer
        data_buffer.append((time, state, repeat, weight, angle, current))
        
        # Save data to Excel file
        if len(data_buffer) % 50 ==0:
            df=pd.DataFrame(data_buffer)
            
            save_to_csv(df, file_path)
            print('saved')
            data_buffer = []

    except ValueError:
        pass
    except IndexError:
        pass

    if keyboard.is_pressed('ESC'):
        df=pd.DataFrame(data_buffer)
        save_to_csv(df, file_path)
        break

