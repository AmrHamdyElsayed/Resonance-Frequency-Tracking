import pyvisa
import time
import math
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
import PySimpleGUI as sg
from PIL import Image

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


t_consts = {"1": 4, "3": 5, "10": 6, "30": 7, "100": 8, "300": 9, "1000": 10, "3000": 11, "10000": 12, "30000": 13}
sampl_rates = {"512": 13, "256": 12, "128": 11, "64": 10, "32": 9, "16": 8, "8": 7, "4": 6, "2": 5, "1": 4}
t_const = 6                    #determine time constant according to values in LIA manual
harmonic = 0
smpl_rate = 13                 #determine sampling rate according to values in LIA manual
excite_freq = 0
start_freq = 0
freq_step = 1
end_freq = 12433
expected_freq = 0
running_times = int((end_freq - excite_freq)/(freq_step))
excite_amplitude = 0
excite_time = 3
relax_time = 3
sampling_rate = 0
num_samples_LIA = 1000                              #real_data from lock in amplifier
resolution_var = 10                                # factor to determine inbetween steps (resolution in FFT pins)
buffer_delay = 0.5                                    #required delay for the LIA buffer to start recording
reson_freq = 0
#image =  Image.open("thorlabs.png")
logo = [sg.Image("thorlabs.png", size =(350,100))]
#imagepath = "C:\Users\aelsayed\Downloads\thorlabs.png"
excit_freq = [sg.Text("Excitation Frequency (HZ)",text_color='White', justification = 'l'), sg.In(size=(15, 1), key="excite")]
excit_ampl = [sg.Text("Excitation Amplitude (mV)", justification = 'l'), sg.In(size=(15, 1), key="ampl")]
#ref_freq = [[sg.Text("Reference Frequency"), sg.In(size=(25, 1), key="ref")]]
expected_resonance = [sg.Text("Expected Resonance (HZ)", justification = 'l'), sg.In(size=(15, 1), key="expec")]

time_constants = [[sg.Text('Time Constant (ms)')],
[sg.Combo([1,3,10,30,100,300,1000,3000,10000,30000], size=(15,15), key='integration_time')]]
harmonics = [[sg.Text('Harmonic Order')],
[sg.Combo([1,2,3,4],  size=(15,15), key='harmonic')]]
sampling_rates = [[sg.Text('Sampling Rate (sample/s)')],
[sg.Combo([512,256,128,64,32,16,8,4,2,1], size=(15,15), key='smpl_rate')]]
Configuration = [[sg.Text('Preamplifier Config')],
[sg.Combo(['New_Board(INST)', 'Old_Board(TIA)'], size=(15,15), key='config')]]
lock_in_output = [[sg.Text('Lock in amplifier output')],
[sg.Combo(['X Value (Electrical)', 'R Value (Photoacoustic)'], size=(15,15), key='lia_out')]]

frequencies = [excit_freq, excit_ampl, expected_resonance]
buttons = [[sg.Button("Calculate Resonance Frequency")], [sg.Button("Plot BF Signal & FFT")]]
layout = [[sg.Column(frequencies), sg.VSeperator(), sg.Column([logo])],
[sg.Column(time_constants), sg.VSeperator(), sg.Column(harmonics),sg.VSeperator(),sg.Column(sampling_rates),sg.VSeperator(),sg.Column(Configuration),sg.VSeperator(),sg.Column(lock_in_output)],
[sg.Column(buttons),sg.VSeperator(),sg.Column([[sg.Button("Run at Resonance")]])]]

# Create the window
window = sg.Window("Resonance Frequency Calculation", layout, margins=(8, 8), size =(800,290))

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

# Create an event loop
while True:
    event, values = window.read()
    excite_freq = values['excite']
    start_freq = values['excite']
    excite_amplitude = 0.001 * float(values['ampl'])
    rate = values['smpl_rate']
    sampling_rate = sampl_rates[str(rate)]
    integ_time = values['integration_time']
    t_const = t_consts[str(integ_time)]
    harmonic = values['harmonic']
    expected_freq = values["expec"]
    num_samples_pad = resolution_var * int(rate)     #additional zeros to have sufficient samples for required resolution
    zeros_pad = [0] * (num_samples_pad - num_samples_LIA)
    amp = values['config']
    lia_output = values['lia_out']
    lia_out_value = 0
    if lia_output == 'R Value (Photoacoustic)':
        lia_out_value = 1
    print(lia_out_value)

    #if event == "values":
    #    print(excite_freq, start_freq, excite_amplitude, sampling_rate, t_const, num_samples_pad)
    if event == "Calculate Resonance Frequency":
        LOCK_IN_AMP.write("REST")
        excite_command = ":APPL1:SIN " + str (excite_freq) + "," + str(excite_amplitude)
        inst_switch_command = ":APPL2:SQU " + str (0.3333) + "," + str(3)
        Func_Gen.write(excite_command)
        time.sleep(excite_time)
        LOCK_IN_AMP.write("DDEF"+ str(lia_out_value)+ "{0,0}")
        LOCK_IN_AMP.write("OFLT"+str(t_const))
        LOCK_IN_AMP.write("HARM"+str(harmonic))
        LOCK_IN_AMP.write("SEND0")
        LOCK_IN_AMP.write("SRAT"+str(smpl_rate))
        LOCK_IN_AMP.write("DDEF0")
        LOCK_IN_AMP.write("FAST2")
        LOCK_IN_AMP.write("STRD")
        time.sleep(0.5)
        Func_Gen.write("OUTP1 OFF")
        if amp == "New_Board(INST)":
            Func_Gen.write(inst_switch_command)
        else:
            Func_Gen.write("OUTP2 OFF")
        time.sleep(relax_time)
        LOCK_IN_AMP.write("PAUS")
        Func_Gen.write("OUTP2 OFF")
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

        #offset removal
        avg = mean(Lia_Data)
        if avg > 0:
            Lia_Data = [x-np.abs(avg) for x in Lia_Data]
        else:
            Lia_Data = [x+np.abs(avg) for x in Lia_Data]

        #SCIPY_FFT
        Lia_Data.extend(zeros_pad)
        xt = np.linspace(start = 0, stop = num_samples_pad, num = num_samples_pad-1)
        fft_cal = rfft(Lia_Data)
        fft_cal_mag = np.abs(fft_cal)/num_samples_pad
        fplott = rfftfreq(num_samples_pad, 1/int(rate))
        fplot = fplott[0:int(num_samples_pad/2)]
        fft_cal_mag_plot = 2 * fft_cal_mag[0:int(num_samples_pad/2+1)]

        beat_freq = fplot[np.argmax(fft_cal_mag_plot)]
        #calculate resonance frequency
        if excite_freq < expected_freq:
            reson_freq = float(excite_freq) + float(beat_freq)
        else:
            reson_freq = float(excite_freq) - float(beat_freq)

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
        plt.show()
        sg.popup('Resonance Frequency = ' + str(reson_freq))

    if event == "Run at Resonance":
        excite_command = ":APPL1:SIN " + str (reson_freq-1) + "," + str(excite_amplitude)
        time.sleep(2)
        excite_command = ":APPL1:SIN " + str (reson_freq) + "," + str(excite_amplitude)
        time.sleep(2)
        excite_command = ":APPL1:SIN " + str (reson_freq + 1) + "," + str(excite_amplitude)
        time.sleep(2)

    if event == "Photoacoustic":
        photoacoustic_freq = reson_freq/2
        excite_command = ":APPL1:SIN " + str (photoacoustic_freq) + "," + str(0.1)
    if event == "run" or event == sg.WIN_CLOSED:
        break

window.close()
