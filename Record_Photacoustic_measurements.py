import pyvisa
import time
import math
import lmfit
from lmfit import Model
from lmfit.models import PseudoVoigtModel
from lmfit.models import LorentzianModel
import matplotlib.pyplot as plt
import xlsxwriter as xls
import pandas as pd
import numpy as np
from numpy import pi
from scipy import signal
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy import optimize
from scipy.optimize import curve_fit
#from statistics import mean
# import nidaqmx
# from nidaqmx.constants import FuncGenType
# from nidaqmx.constants import AcquisitionType
# from nidaqmx.constants import RegenerationMode
from nidaqmx import stream_writers

num_samples_LIA = 7000
runs = 1
t_const = 7
#Function_Generator_Commands to Control
rm = pyvisa.ResourceManager()
Instr = rm.list_resources()
Inst_list = list(Instr)
print(Inst_list)
# matching_usb = [Connected_USB for Connected_USB in Inst_list if "USB" in Connected_USB]
matching_GPIB = [Connected_GPIB for Connected_GPIB in Inst_list if "GPIB" in Connected_GPIB]
# print(matching_usb[0])
print(matching_GPIB[0])
# Func_Gen = rm.open_resource(matching_usb[0])
LOCK_IN_AMP = rm.open_resource(matching_GPIB[0])
LOCK_IN_AMP.timeout = 100000
print(LOCK_IN_AMP.query("*IDN?"))
#Devices Control
for item in range(runs):
    LOCK_IN_AMP.write("OFLT"+str(t_const))
    print(LOCK_IN_AMP.query("OFLT?"))
    LOCK_IN_AMP.write("REST")
    LOCK_IN_AMP.write("SEND0")
    LOCK_IN_AMP.write("SRAT10")
    LOCK_IN_AMP.write("DDEF0")
    LOCK_IN_AMP.write("FAST2")
    LOCK_IN_AMP.write("STRD")
    time.sleep(0.5)
    time.sleep(130)
    LOCK_IN_AMP.write("PAUS")
    TRCA_Command = "TRCA?0,"+str(num_samples_LIA)
    LOCK_IN_AMP.write(TRCA_Command)
    Data = LOCK_IN_AMP.read_raw()
    Data_Converted = str(Data)
    Data_points_bin = Data_Converted.split(",")
    Data_points = Data_points_bin[1:-1]
    Lia_Data = []
    for item in  range(len(Data_points)):
        value = Data_points[item]
        Lia_Data.append(float(value))

    workbook = xls.Workbook('Data_photo' + str(t_const)+ '.xlsx' )
    worksheet = workbook.add_worksheet()
    worksheet.write_column("B1", Lia_Data)
    workbook.close()
    t_const += 1



# fig, [plt1, plt2] = plt.subplots(nrows=2, ncols=1)
# plt1.grid()
# plt2.grid()
# plt1.set_xlabel('number of samples', fontsize=15)
# plt1.set_ylabel('Voltage (V)', fontsize=15)
# plt2.set_xlabel('Freq(Hz)', fontsize=15)
# plt2.set_ylabel('Magnitude', fontsize=15)
# plt1.plot(xt[0:1000], Lia_Data[0:1000])
# plt2.plot(fplot, fft_cal_mag_plot)
# Exc = "Excitation_Freq =" + str(excite_freq)
# plt.figtext(0.9, 0.2,
#             Exc, horizontalalignment ="center",
#             wrap = True, fontsize = 10,
#             bbox ={'facecolor':'grey',
#                    'alpha':0.3, 'pad':5})
# annot_max(fplot, fft_cal_mag_plot)
# excite_freq += freq_step
# # plt.show()
# plt.savefig(str(excite_freq) + '_FFT.png')
# plt.show(block=False)
# plt.pause(1)
# plt.close()
