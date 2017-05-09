import wave
import pyaudio
import struct
import numpy as np
import pylab

filename = 'D:\\1.wav'
wf1 = wave.open(filename, 'r')
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf1.getsampwidth()),
channels=wf1.getnchannels(),
rate=wf1.getframerate(),
output=True)
nframes = wf1.getnframes()
framerate = wf1.getframerate()

str_data = wf1.readframes(nframes)
wf1.close()

wave_data = np.fromstring(str_data, dtype=np.short)

wave_data.shape = -1, 2
wave_data = wave_data.T

N=44100 # sampling frequency
start=0
df = framerate/(N-1)
freq = [df*n for n in range(0,N)]
wave_data2=wave_data[0][start:start+N]
c=np.fft.fft(wave_data2)*2/N
d=int(len(c)/2)

while freq[d]>4000:
    d-=10
pylab.plot(freq[:d-1],abs(c[:d-1]),'r')
pylab.show()