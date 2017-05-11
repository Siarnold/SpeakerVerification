#coding=utf-8
import wave
import pyaudio
import struct
import numpy as np
import pylab
#set file name
filename = 'D:\\1.wav'

#open wav file
wf1 = wave.open(filename, 'r')
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf1.getsampwidth()),
channels=wf1.getnchannels(),
rate=wf1.getframerate(),
output=True)
nframes = wf1.getnframes()
framerate = wf1.getframerate()

# 以string类型存储完整一帧数据
str_data = wf1.readframes(nframes)
wf1.close()

#将波形数据转换为数组
wave_data = np.fromstring(str_data, dtype=np.short)

#将wave_data数组改为两列 然后转职
wave_data.shape = -1, 2
wave_data = wave_data.T

N=44100 # sampling frequency
start=0#开始采样位置
df = framerate/(N-1)#分辨率
freq = [df*n for n in range(0,N)]#N个元素
wave_data2=wave_data[0][start:start+N]
c=np.fft.fft(wave_data2)*2/N
#显示采样频率一半的频谱
d=int(len(c)/2)
#显示4000以下的频谱
while freq[d]>4000:
    d-=10
pylab.plot(freq[:d-1],abs(c[:d-1]),'r')
pylab.show()