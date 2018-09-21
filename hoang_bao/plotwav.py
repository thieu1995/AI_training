import matplotlib.pyplot as plt 
import numpy as np 
import wave
import sys 

spf = wave.open("./Xe.wav")

signal = spf.readframes(-1)
signal = np.fromstring(signal,'Int16')
fs = spf.getframerate()

if spf.getnchannels() == 2:
    sys.exit()

Time = np.linspace(0,len(signal)/fs, num=len(signal))
plt.figure(1)
plt.title("signal wave")
plt.plot(Time,signal)
plt.show()