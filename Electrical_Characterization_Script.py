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
from statistics import mean
import nidaqmx
from nidaqmx.constants import FuncGenType
from nidaqmx.constants import AcquisitionType
from nidaqmx.constants import RegenerationMode
from nidaqmx import stream_writers

t_const = 5                    #determine time constant according to values in LIA manual
harmonic = 1
smpl_rate = 13                 #determine sampling rate according to values in LIA manual
excite_freq = start_freq = 12430
freq_step = 1
end_freq = 12433
expected_freq = 12440
running_times = int((end_freq - excite_freq)/(freq_step))
excite_amplitude = 0.2
excite_time = 3
relax_time = 3
sampling_rate = 512
num_samples_LIA = 1000                              #real_data from lock in amplifier
resolution_var = 1000                                # factor to determine inbetween steps (resolution in FFT pins)
num_samples_pad = resolution_var * sampling_rate     #additional zeros to have sufficient samples for required resolution
zeros_pad = [0] * (num_samples_pad - num_samples_LIA)
buffer_delay = 0.5                                    #required delay for the LIA buffer to start recording
beat_freqs = []
beats = []
excite_freqs = []
reson_freqs = []

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

#Function_Generator_Commands to Control
rm = pyvisa.ResourceManager()
Instr = rm.list_resources()
Inst_list = list(Instr)
print(Inst_list)
matching_usb = [Connected_USB for Connected_USB in Inst_list if "USB" in Connected_USB]
matching_GPIB = [Connected_GPIB for Connected_GPIB in Inst_list if "GPIB" in Connected_GPIB]
print(matching_usb[0])
print(matching_GPIB[0])
Func_Gen = rm.open_resource(matching_usb[0])
LOCK_IN_AMP = rm.open_resource(matching_GPIB[0])
LOCK_IN_AMP.timeout = 10000
print(Func_Gen.query("*IDN?"))
print(LOCK_IN_AMP.query("*IDN?"))

#Runing several excitation frequencies
for i in range(running_times):
    excite_freqs.append(excite_freq)
    #Devices Control
    LOCK_IN_AMP.write("REST")
    Func_Gen.write("OUTP2 OFF")
    Func_Gen.write("OUTP2:COMP OFF")
    excite_command = ":APPL1:SIN " + str (excite_freq) + "," + str(excite_amplitude)
    Func_Gen.write(excite_command)
    time.sleep(excite_time)
    LOCK_IN_AMP.write("OFLT"+str(t_const))
    LOCK_IN_AMP.write("HARM"+str(harmonic))
    LOCK_IN_AMP.write("SEND0")
    LOCK_IN_AMP.write("SRAT"+str(smpl_rate))
    LOCK_IN_AMP.write("DDEF0")
    LOCK_IN_AMP.write("FAST2")
    LOCK_IN_AMP.write("STRD")
    time.sleep(0.5)
    Func_Gen.write("OUTP1 OFF")
    time.sleep(relax_time)
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

    workbook = xls.Workbook('Data_' + str(excite_freq) + '.xlsx' )
    worksheet = workbook.add_worksheet()
    worksheet.write_column('B1', Lia_Data)
    workbook.close()

    #offset removal
    avg = mean(Lia_Data)
    if avg > 0:
        Lia_Data = [x-np.abs(avg) for x in Lia_Data]
    else:
        Lia_Data = [x+np.abs(avg) for x in Lia_Data]

    #storing raw_data in excel sheets

    #SCIPY_FFT
    Lia_Data.extend(zeros_pad)
    xt = np.linspace(start = 0, stop = num_samples_pad, num = num_samples_pad-1)
    fft_cal = rfft(Lia_Data)
    fft_cal_mag = np.abs(fft_cal)/num_samples_pad
    fplott = rfftfreq(num_samples_pad, 1/sampling_rate)
    fplot = fplott[0:int(num_samples_pad/2)]
    fft_cal_mag_plot = 2 * fft_cal_mag[0:int(num_samples_pad/2+1)]

    #drawing_shifts
    beat_freq = fplot[np.argmax(fft_cal_mag_plot)]
    beats.append(beat_freq)
    beat_freq_change = beat_freq - float(int(beat_freq))
    beat_freqs.append(beat_freq_change)

    #calculate resonance frequency
    if excite_freq < expected_freq:
        reson_freq = excite_freq + beat_freq
    else:
        reson_freq = excite_freq - beat_freq
    reson_freqs.append(reson_freq)

    fig, [plt1, plt2] = plt.subplots(nrows=2, ncols=1)
    plt1.grid()
    plt2.grid()
    plt1.set_xlabel('number of samples', fontsize=15)
    plt1.set_ylabel('Voltage (V)', fontsize=15)
    plt2.set_xlabel('Freq(Hz)', fontsize=15)
    plt2.set_ylabel('Magnitude', fontsize=15)
    plt1.plot(xt[0:1000], Lia_Data[0:1000])
    plt2.plot(fplot, fft_cal_mag_plot)
    Exc = "Excitation_Freq =" + str(excite_freq)
    plt.figtext(0.9, 0.2,
                Exc, horizontalalignment ="center",
                wrap = True, fontsize = 10,
                bbox ={'facecolor':'grey',
                       'alpha':0.3, 'pad':5})
    annot_max(fplot, fft_cal_mag_plot)
    excite_freq += freq_step
    # plt.show()
    plt.savefig(str(excite_freq) + '_FFT.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

#export different Frequency values to excel sheet
index = [j for j in range(running_times)]
freqs = dict(zip(index, excite_freqs))
beats_dic = dict(zip(index, beats))
reson_dic = dict(zip(index, reson_freqs))
# fit_dic = dict(zip(index, fit_freqs))
shifts = dict(zip(index, beat_freqs))
beat_shifts = pd.DataFrame({'Frequencies': freqs, 'Shifts': shifts,
'Beat_Frequencies': beats_dic, 'FFT Resonance Frequency': reson_dic})
file_name = 'Beat_frequency_shifts.xlsx'
beat_shifts.to_excel(file_name)

plt.grid()
plt.xlabel('Excitation Frequencies (Hz)', fontsize=15)
plt.ylabel('FFT Resonance Frequency (Hz)', fontsize=15)
plt.plot(excite_freqs, reson_freqs, 'o')
relax = "Relaxation time =" + str(relax_time)
buffer = "Buffer Delay =" + str(buffer_delay)

plt.figtext(0.95, 0.2,
            relax, horizontalalignment ="center",
            wrap = True, fontsize = 10,
            bbox ={'facecolor':'grey',
                'alpha':0.3, 'pad':5})
plt.figtext(0.95, 0.5,
            buffer, horizontalalignment ="center",
            wrap = True, fontsize = 10,
            bbox ={'facecolor':'grey',
                'alpha':0.3, 'pad':5})
plt.show()

Func_Gen.write("OUTP1 OFF")
Func_Gen.write("OUTP1:COMP OFF")
Func_Gen.write("OUTP2 OFF")
Func_Gen.write("OUTP2:COMP OFF")
