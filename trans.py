#coding=utf-8
import wave
import pyaudio
import struct
import numpy as np
import pylab as pl
#set file name
filename = './piano.wav'
output_file_name = 'pianopy.txt'
#open wav file
wf1 = wave.open(filename, 'r')

nframes = wf1.getnframes()
framerate = wf1.getframerate()

# 以string类型存储完整一帧数据
str_data = wf1.readframes(nframes)
wf1.close()

#将波形数据转换为数组
wave_data = np.fromstring(str_data, dtype=np.short)

#将wave_data数组改为两列 并转置 双声道
wave_data.shape=-1,2
wave_data=wave_data.T

#set values
LEN = 16384 #data per frame
MAXF = 700 #max frequency
MINF = 180 #min freq
MINA = 10 #min amplitude
FREQ = 44100 #sampling frequency
#获得等差数列
FREQ_LIST=pl.linspace(0,FREQ/2,LEN/2)
#num of frames
n = int(len(wave_data[0])/LEN)

notes_list = np.zeros(n-1)
freqs_list = np.zeros(n-1)
maxA_list = np.zeros(n-1)

MAX_INDEX = int(LEN/FREQ*MAXF)
MIN_INDEX = int(LEN/FREQ*MINF)
print(maxA_list.shape)

key=['G3 ','G3#','A3 ','A3#','B3 ',
'C4 ','C4#','D4 ','D4#','E4 ','F4 ','F4#','G4 ','G4#','A4 ','A4#','B4 ',
'C5 ','C5#','D5 ','D5#','E5 ','F5 ','F5#','G5 ','G5#','A5 ','A5#','B5 ',
'C6 ','C6#','D6 ','D6#','E6 ','F6 ','F6#','G6 ','G6#','A6 ','A6#','B6 ']

for i in range(0,n-1):
    X = np.fft.fft(wave_data[0][(i)*LEN:(i+1)*LEN-1])
    X_cut = X[0:MAX_INDEX-1]
    #pl.plot(abs(X_cut)) #显示一帧频谱
    #pl.show()

    A = np.sqrt(X_cut * X_cut.conj())     #取模
    #print(A)
    if A.any() > 0 :
        A = 20 * pl.log10(A)   #dB
    #print(A)
    max_index=MIN_INDEX
    maxA_list[i]=A[MIN_INDEX]

    for t in range(MIN_INDEX,MAX_INDEX-1):
        if(A[t]>maxA_list[i]):
            maxA_list[i]=A[t]
            max_index=t

    best_index=max_index

    for mult in range(2,5):
        test_index=int(max_index/mult)
        if(test_index>MIN_INDEX):
            if(A[test_index]>maxA_list[i]*0.9):
                best_index=int(max_index/mult)
        else:
            break

    freqs_list[i]=FREQ_LIST[best_index]
    notes_list[i]=np.log(freqs_list[i]/220)/np.log(2)*12+3

    if maxA_list[i]<MINA or notes_list[i]<-12:
        notes_list[i]=NaN

for i in range(0,n-1):
    if maxA_list[i]<MINA or notes_list[i]<-12:
        notes_list[i] = NaN

output_file = open(output_file_name,'w')
for i in range(0,n-1):
    if(notes_list[i]>0):
        output_file.write(str(round(notes_list[i])))
        output_file.write("\r\n")
    else:
        output_file.write("0\r\n")
output_file.close()
K = [0,0,0,0,0,0,0,0,0,0,0,0]

"""
#将波形数据转换为数组
wave_data = np.fromstring(str_data, dtype=np.short)

#将wave_data数组改为两列 然后转置
wave_data.shape = -1, 2
wave_data = wave_data.T

LEN=44100 # sampling frequency
start=0#开始采样位置
df = framerate/(LEN-1)#分辨率
freq = [df*n for n in range(0,LEN)]#N个元素
wave_data2=wave_data[0][start:start+LEN]
c=np.fft.fft(wave_data2)*2/LEN
#显示采样频率一半的频谱
d=int(len(c)/2)
#显示4000以下的频谱
while freq[d]>4000:
    d-=10
pylab.plot(freq[:d-1],abs(c[:d-1]),'r')
pylab.show()
"""