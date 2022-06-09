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


#laser_dc = 0.5
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
# fit_freqs = []
# expected_beat = expected_freq - excite_freq
# x = np.linspace(start = 0, stop = resolution_var, num = resolution_var-51)

# def func(x, amp,freq, decay, shift):
#     model = amp * np.sin(x*2*pi*(freq)+shift) * np.exp(-x*decay)
#     return model

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
    print(excite_freq)
    #laser_command = ":APPL2:SIN " + str (excite_freq) + "," + str(amplitude) #+ "," + str(laser_dc)
    #control_command = ":APPL2:SQU " + str (0.25) + "," + str(4)
    #Func_Gen.write(laser_command)
    #Func_Gen.write(control_command)
    Func_Gen.write("OUTP2 OFF")
    Func_Gen.write("OUTP2:COMP OFF")
    #ref_command = ":APPL2:SIN " + str (excite_freq) + "," + str(excite_amplitude)
    excite_command = ":APPL1:SIN " + str (excite_freq) + "," + str(excite_amplitude)
    Func_Gen.write(excite_command)
    #Func_Gen.write(ref_command)
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

    #print(Lia_Data)
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

    #storing raw_data in excel sheets
    # workbook = xls.Workbook('FFT_Data_' + str(excite_freq) + '.xlsx' )
    # worksheet = workbook.add_worksheet()
    # worksheet.write_column('B1', Lia_Data)
    # workbook.close()

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

#######################################################################################
# def lorentzian(x,*p) :
#     # A lorentzian peak with:
#     #   Constant Background          : p[0]
#     #   Peak height above background : p[1]
#     #   Central value                : p[2]
#     #   Full Width at Half Maximum   : p[3]
#     lorentzian_model = p[0]+(p[1]/numpy.pi)/(1.0+((x-p[2])/p[3])**2)
#     return lorentzian_model
#Data_fit_scipi_curve_fit
# freq_var = 12451 - excite_freq
# for i in range(10):
#     print(freq_var)
#     guess = [0.0269, freq_var,0.0049, -2.52857]
#     parameters, covariance = curve_fit(func, x, Lia_Data[50:400], p0= guess, method='lm')
#     print(parameters)
#     Fit_freqq = "Fit frequency = " + str(parameters[1])
#     plt.grid()
#     plt.plot(x, Lia_Data[50:400], 'o', label='data')
#     plt.plot(x, func(x, *parameters), '-', label='Fit')
#     plt.figtext(0.9, 0.2,
#             Fit_freqq, horizontalalignment ="center",
#             wrap = True, fontsize = 10,
#             bbox ={'facecolor':'grey',
#             'alpha':0.4, 'pad':5})
#     plt.legend()
#     plt.show()
#     freq_var += 0.1
# fit_res = parameters[1] + excite_freq
# fit_freqs.append(fit_res)

# Data_fit_lmfit
# avg_fit_freqs = []
# data_model = Model(func)
# for i in range(10):
#     try:
#         fitting_result = data_model.fit(Lia_Data[50:], amp=0.0269, freq=freq_var, decay=0.0049, shift=-2.52857 , x=x)
#         print(fitting_result.fit_report())
#         # print(fitting_result.best_values)
#         fitting_freq = fitting_result.best_values["freq"]
#         avg_fit_freqs.append(fitting_freq)
#         print(fitting_freq)
#         #fitting_result.params.pretty_print()
#         fit_freqq = "input frequency = " + str(freq_var)
#         fitt_freqq = "best fit frequency = " + str(fitting_freq)
#         plt.plot(x, Lia_Data[50:], 'o')
#         #plt.plot(x, fitting_result.init_fit, '-', label='init_fit')
#         plt.plot(x, fitting_result.best_fit, '-', label='best fit')
#         plt.figtext(0.9, 0.2,
#             fit_freqq, horizontalalignment ="center",
#             wrap = True, fontsize = 10,
#             bbox ={'facecolor':'grey',
#             'alpha':0.4, 'pad':5})
#         plt.figtext(0.9, 0.4,
#             fitt_freqq, horizontalalignment ="center",
#             wrap = True, fontsize = 10,
#             bbox ={'facecolor':'grey',
#             'alpha':0.4, 'pad':5})
#         plt.legend()
#         plt.show()
#     except Exception as e:
#         print(e)
#     freq_var += 0.1
#
# print(mean(avg_fit_freqs))
# fit_res = mean(avg_fit_freqs) + excite_freq
# fit_freqs.append(fit_res)

#storing raw_data in excel sheets
# workbook = xls.Workbook('FFT_Data_' + str(excite_freq) + '.xlsx' )
# worksheet = workbook.add_worksheet()
# worksheet.write_column('A1', fft_cal_mag)
# worksheet.write_column('B1', Lia_Data)
# workbook.close()

    # print(len(fft_cal_mag))
    # xtt = x = np.arange(0, 256000)
    # mod = PseudoVoigtModel()
    # pars = mod.guess(fft_cal_mag, x=xtt)
    # print(type(pars), pars)
    # #pvoigt_pars = [4.32847e-006, 11591.46, 884.3008, 1.968311]
    # #pvoigt_pars = Parameters()
    # #pars['height'] = Parameter(name='height', value=4.32847e-006, min=0, max=1)
    # out = mod.fit(fft_cal_mag, pars, x=xtt)
    # print(out.fit_report(min_correl=0.25))
    # out.plot()
    # plt.show()
    # time.sleep(30)

#excite_freqs = np.linspace(start = start_freq, stop = end_freq, num = running_times)
    # freqss = np.linspace(start = 7, stop = 8, num = 100)
    # for item in freqss:
    #     plt.plot(x, func(x,0.3, item, 4), 'o', label='Fit_Function')
    #     plt.figtext(0.9, 0.2,
    #             str(item), horizontalalignment ="center",
    #             wrap = True, fontsize = 10,
    #             bbox ={'facecolor':'grey',
    #             'alpha':0.4, 'pad':5})
    #     plt.legend()
    #     plt.show()

    #Data_fit_lmfit
    # data_model = Model(func)
    # for i in range(2):
    #     beat_freqq = "Beat frequency = " + str(freq_var)
    #     try:
    #         fitting_result = data_model.fit(Lia_Data, amp=0.025, freq=freq_var, decay=4, x=x)
    #         print(fitting_result.fit_report())
    #         print(fitting_result.best_values)
    #         plt.plot(x, Lia_Data, 'o')
    #         plt.plot(x, fitting_result.init_fit, '-', label='init_fit')
    #         plt.plot(x, fitting_result.best_fit, '-', label='best fit')
    #         plt.figtext(0.9, 0.2,
    #                 beat_freqq, horizontalalignment ="center",
    #                 wrap = True, fontsize = 10,
    #                 bbox ={'facecolor':'grey',
    #                 'alpha':0.4, 'pad':5})
    #         plt.legend()
    #         plt.show()
    #     except Exception as e:
    #         print(e)
        #freq_var += 0.01

    # fitting_result.plot(datafmt='o', fitfmt='-', initfmt='--', xlabel=None, ylabel=None, yerr=None, numpoints=None, fig=None, data_kws=None, fit_kws=None, init_kws=None,
    #  ax_res_kws=None, ax_fit_kws=None, fig_kws=None, show_init=False, parse_complex='abs', title=None)
    # print(fitting_result.best_values)

    #fitting_result.plot_residuals(ax=None, datafmt='o', yerr=None, data_kws=None,
     #fit_kws=None, ax_kws=None, parse_complex='abs', title=None)
    #fitting_result.plot()


    #expected_beat = expected_freq - excite_freq
    # print(expected_beat)
    # fit_vars[2] -= 5
    # print(fit_vars[2])
    # params = lmfit.Parameters()
    # params.add('amp', fit_vars[0], min=0, max=1)
    # params.add('decay', fit_vars[1], min=0, max=50)
    # params.add('freq', 7.7, min=0, max=40)
    # fit = lmfit.minimize(resid, params, args=(x, Lia_Data))
    # #fit.params.pretty_print()
    # lmfit.report_fit(fit)
    #print('-------------------------------')
    #print('Parameter    Value       Stderr')
    #for name, param in fit.params.items():
    #    print(f'{name:7s} {param.value:11.5f} {param.stderr:11.5f}')
    #print(fit.params)
    #print(fit.success)
    # count = 0
    # for key in fit.params:
    #     #print(key, "=", fit.params[key].value, "+/-", fit.params[key].stderr)
    #     fit_vars[count] = fit.params[key].value
    #     print(fit_vars[count])
    #     count+= 1
    # print(fit_vars)
    #plt.plot(x, Lia_Data, 'ko', lw=2)
    #plt.plot(x, fit, 'b--', lw=2)
    #plt.legend(['data', 'leastsq'], loc='upper right')
    #plt.show()
    #Lorentzian_fit
    #yt = np.linspace(start = 0, stop = int(num_samples/2), num = int(num_samples/2))
    # model = LorentzianModel()
    # params = model.guess(fft_cal_mag, x=yt)
    # result = model.fit(fft_cal_mag, params, x=yt)
    # print(result.fit_report())
    # for key in result.params:
    #     print(key, "=", result.params[key].value, "+/-", result.params[key].stderr)
    # result.plot_fit()
    # plt.show()

# def resid(params, x, ydata):
#     decay = params['decay'].value
#     amp = params['amp'].value
#     freq = params['freq'].value
#     y_model = amp * np.sin(x*2*pi*(freq)) * np.exp(-x*decay)
#     diff = y_model - ydata
#     return diff
#
#
# def lorentzian(x,*p) :
#     # A lorentzian peak with:
#     #   Constant Background          : p[0]
#     #   Peak height above background : p[1]
#     #   Central value                : p[2]
#     #   Full Width at Half Maximum   : p[3]
#     lorentzian_model = p[0]+(p[1]/numpy.pi)/(1.0+((x-p[2])/p[3])**2)
#     return lorentzian_model
#
# plt.grid()
# plt.xlabel('Excitation Frequencies', fontsize=15)
# plt.ylabel('Beat Frequency', fontsize=15)
# plt.plot(excite_freqs, beat_freqs, 'r+')
# relax = "Relaxation time =" + str(relax_time)
# buffer = "Buffer Delay =" + str(buffer_delay)
# #Exc = "Num =" + str(excite_freq)
# plt.figtext(0.95, 0.2,
#             relax, horizontalalignment ="center",
#             wrap = True, fontsize = 10,
#             bbox ={'facecolor':'grey',
#                 'alpha':0.3, 'pad':5})
# plt.figtext(0.95, 0.5,
#             buffer, horizontalalignment ="center",
#             wrap = True, fontsize = 10,
#             bbox ={'facecolor':'grey',
#                 'alpha':0.3, 'pad':5})
# # plt.figtext(0.9, 0.2,
# #             relax, horizontalalignment ="center",
# #             wrap = True, fontsize = 10,
# #             bbox ={'facecolor':'grey',
# #                 'alpha':0.3, 'pad':5})
# plt.show()


    #step = np.linspace(start = 0,stop = 2,num = num_samples)

    #numpy_FFT
    #fft_cal = np.fft.fft(Lia_Data)
    #fft_cal_mag = np.abs(fft_cal)/num_samples
    #fstep = sampling_rate/num_samples
    #f = np.linspace(0, (num_samples-1)*fstep, num_samples)
    #fplot = f[0:int(num_samples/2+1)]
    #fft_cal_mag_plot = 2 * fft_cal_mag[0:int(num_samples/2+1)]


#Function_generator_Frequency_Sweep
# Func_Gen.write(":FREQ2:CENT 12450")
# Func_Gen.write(":FREQ2:SPAN 20HZ")
# Func_Gen.write(":FREQ2:STAR 12440")
# Func_Gen.write(":FREQ2:STOP 12460")
# Func_Gen.write(":SWE2:SPAC LIN")
# Func_Gen.write(":SWE2:TIME 60")
# Func_Gen.write(":SWE2:STAT ON")

    #storing raw_data in excel sheets
    # workbook = xls.Workbook('LIA_Data_' + str(excite_freq) + '.xlsx' )
    # worksheet = workbook.add_worksheet()
    # worksheet.write_column('A1', Lia_Data)
    # workbook.close()
    #print(Lia_Data)


# plt.figtext(0.9, 0.2,
#             relax, horizontalalignment ="center",
#             wrap = True, fontsize = 10,
#             bbox ={'facecolor':'grey',
#                 'alpha':0.3, 'pad':5})
#points_list = points.decode()
#points_lists = points_list.encode()
#convert_points_big =  int.from_bytes(points_lists, "big")
#convert_points_little =  int.from_bytes(points_lists, "little")
#num_points = LOCK_IN_AMP.query('SPTS?')
# points_bytes = LOCK_IN_AMP.read_raw()
# points_string = str(points_bytes)[6:-3]
# count = 1
# points_new_string = ""
# for i in points_string:
#     num = count-1
#     points_new_string += "0" + points_string[num:num+1]
#     count+=1
# points_new_string_sliced = points_new_string[4:]
# newbytes = bytearray.fromhex(points_string[4:])
# converted_points_little = int.from_bytes(newbytes, byteorder='little', signed=False)
#plt.figtext(0.7, 0.74,
#            BF, horizontalalignment ="center",
#            wrap = True, fontsize = 10,
#            bbox ={'facecolor':'grey',
#                   'alpha':0.3, 'pad':5})
#plt.plot(xt, Lia_Data, 'o', label='data')
#plt.plot(xt, fit_bf, '-', label='fit')
#plt.legend()
#plt.plot(maximas, peak_Data, 'r+')
#plt.plot(xt, bf_exp(12452, 12449), 'o', label='data')
# guess = 12451.2426
#print(xt, bf_exp(12450, 12400))
# parameters, covariance = curve_fit(bf_exp, xt, Lia_Data, p0= guess, method='lm')
# fit_fres = parameters[0]
# print(fit_fres)
# fit_bf = bf_exp(step, fit_fres)
#Find peaks and thier indices
#print(Lia_Data)
# data = np.array(Lia_Data)
# peaks = find_peaks(data, distance=80)
# maximas = peaks[0]
# peak_Data = []
# for i in  range(len(maximas)):
#     max = maximas[i]
#     peak_Data.append(Lia_Data[max])
# print(maximas)
#beat_freq = num_samples/(maximas[1]- maximas[0])
#BF =  "BF = " + str(beat_freq)
#Data_Converted.replace("b'\xff", '')
#Data_list = Data.decode()
