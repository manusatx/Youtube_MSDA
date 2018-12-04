# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 18:20:23 2018

@author: manor
"""
AudioPath=r'C:\Users\manor\Desktop\Mainproject\\'


import tensorflow as tf
# Requires latest tf-1.4 on Windows
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
def parse_wave_tf(filename):
    audio_binary = tf.read_file(filename)
    desired_channels = 1
    wav_decoder = contrib_audio.decode_wav(
        audio_binary,
        desired_channels=desired_channels)
    with tf.Session() as sess:
        sample_rate, audio = sess.run([
            wav_decoder.sample_rate,
            wav_decoder.audio])
        first_sample = audio[0][0] * (1 << 15)
        second_sample = audio[1][0] * (1 << 15)
        print('''
Parsed {filename}
-----------------------------------------------
Channels: {desired_channels}
Sample Rate: {sample_rate}
First Sample: {first_sample}
Second Sample: {second_sample}
Length in Seconds: {length_in_seconds}'''.format(
            filename=filename,
            desired_channels=desired_channels,
            sample_rate=sample_rate,
            first_sample=first_sample,
            second_sample=second_sample,
            length_in_seconds=len(audio) / sample_rate))
        
parse_wave_tf(AudioPath+'James P. Gorman interview1.wav')

#Spectogram of file
"""Plots
Time in MS Vs Amplitude in DB of a input wav signal
"""

import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft

myAudio = AudioPath+'James P. Gorman interview1.wav'

#Read file and get sampling freq [ usually 44100 Hz ]  and sound object
samplingFreq, mySound = wavfile.read(myAudio)

#Check if wave file is 16bit or 32 bit. 24bit is not supported
mySoundDataType = mySound.dtype

#We can convert our sound array to floating point values ranging from -1 to 1 as follows
mySound = mySound / (2.**15)

#Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel
mySoundShape = mySound.shape
samplePoints = float(mySound.shape[0])

#Get duration of sound file
signalDuration =  mySound.shape[0] / samplingFreq

#If two channels, then select only one channel
mySoundOneChannel = mySound[:,0]

#Plotting the tone
# We can represent sound by plotting the pressure values against time axis.
#Create an array of sample point in one dimension
timeArray = numpy.arange(0, samplePoints, 1)

#
timeArray = timeArray / samplingFreq

#Scale to milliSeconds
timeArray = timeArray * 1000

#Plot the tone
plt.plot(timeArray, mySoundOneChannel, color='G')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.show()

#Plot frequency content
#We can get frquency from amplitude and time using FFT , Fast Fourier Transform algorithm

#Get length of mySound object array
mySoundLength = len(mySound)

#Take the Fourier transformation on given sample point 
#fftArray = fft(mySound)
fftArray = fft(mySoundOneChannel)

numUniquePoints = numpy.ceil((mySoundLength + 1) / 2.0)
fftArray = fftArray[0:numUniquePoints]

#FFT contains both magnitude and phase and given in complex numbers in real + imaginary parts (a + ib) format.
#By taking absolute value , we get only real part

fftArray = abs(fftArray)

#Scale the fft array by length of sample points so that magnitude does not depend on
#the length of the signal or on its sampling frequency

fftArray = fftArray / float(mySoundLength)

#FFT has both positive and negative information. Square to get positive only
fftArray = fftArray **2

#Multiply by two (research why?)
#Odd NFFT excludes Nyquist point
if mySoundLength % 2 > 0: #we've got odd number of points in fft
    fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2

else: #We've got even number of points in fft
    fftArray[1:len(fftArray) -1] = fftArray[1:len(fftArray) -1] * 2  

freqArray = numpy.arange(0, numUniquePoints, 1.0) * (samplingFreq / mySoundLength);

#Plot the frequency
plt.plot(freqArray/1000, 10 * numpy.log10 (fftArray), color='B')
plt.xlabel('Frequency (Khz)')
plt.ylabel('Power (dB)')
plt.show()

#Get List of element in frequency array
#print freqArray.dtype.type
freqArrayLength = len(freqArray)
print("freqArrayLength =", freqArrayLength)
numpy.savetxt("freqData.txt", freqArray, fmt='%6.2f')

#Print FFtarray information
print("fftArray length =", len(fftArray))
numpy.savetxt("fftData.txt", fftArray)










#Download speech dataset and try to train the audio
    
import tensorflow as tf
import numpy as np

class SoundCNN():
	def __init__(self, classes):
		self.x = tf.placeholder(tf.float32, [None, 1024])
		self.y_ = tf.placeholder(tf.float32, [None, classes])

		self.x_image = tf.reshape(self.x, [-1,32,32,1])
		self.W_conv1 = weight_variable([5, 5, 1, 32])
		self.b_conv1 = bias_variable([32])
		self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
		self.h_pool1 = max_pool_2x2(self.h_conv1)
		self.W_conv2 = weight_variable([5, 5, 32, 64])
		self.b_conv2 = bias_variable([64])

		self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
		self.h_pool2 = max_pool_2x2(self.h_conv2)
		self.W_fc1 = weight_variable([8 * 8 * 64, 1024])
		self.b_fc1 = bias_variable([1024])

		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 8*8*64])
		self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
		self.keep_prob = tf.placeholder("float")
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
		self.W_fc2 = weight_variable([1024, classes])
		self.b_fc2 = bias_variable([classes])
		self.h_fc2 = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
		self.y_conv=tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

		self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(tf.clip_by_value(self.y_conv,1e-10,1.0)))
		self.correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


