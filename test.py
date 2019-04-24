import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import datetime

# for  frequency fourier transform
from scipy import fftpack

def create_dataset(X, Y = None, look_back=1):
  '''
    convert an array of values into a dataset matrix
    input: 
      Training data: X, Y, both are 1-D array
      Testing data: X is 1-D array, while Y is None
      look_back: number of X entries considered to predict the next Y 
    return: formed output X and Y
  '''

  outX, outY = [], []
  for i in range(len(X)-look_back-1):
    x0 = X[i:(i+look_back)] # each X element is a list of $look_back elements
    outX.append( x0 )

  if Y is not None:
    assert (len(X) == len(Y)), "ERROR: Size of X and Y different!"
    for i in range(len(Y)-look_back-1):
      y0 = Y[i+look_back]     # each Y element is a single number
      outY.append( y0 )

  return outX, outY

def make_plot( ftrain="data/train.csv", istart = 0, npoints = 175):

  print( "now is ", str( datetime.datetime.now() ), " start index = ", str(istart), " number of data points = ", npoints, sep="" )
  #nrows=nTest + testStart + 1, 
  dataset = pd.read_csv(ftrain, nrows = istart + npoints, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
  dataset.rename({"acoustic_data": "X", "time_to_failure": "Y"}, axis="columns", inplace=True)

  trainX = dataset.X.values[istart : istart + npoints]
  trainY = dataset.Y.values[istart : istart + npoints] 
  indexs = range(0, npoints)

  print("idx start: ", istart, " X mean = ", np.mean(trainX), ", std = ", np.std(trainX), sep = "")
  
  fullfX = fftpack.fft( trainX )
  magniX = np.abs( fullfX ) # magnitude: r^2 + i^2
  freqsX = fftpack.fftfreq(len(trainX)) * npoints
  backoX = fftpack.ifft( fullfX )
  backoX = backoX.real

  for jfx, ffx in enumerate( freqsX ):
    if math.fabs(ffx) < 1.:
      print("Index ", jfx, " freqency ", ffx, " amplitude ", magniX[jfx])

  fig, ax = plt.subplots(5,1, figsize=(20,30))
  ax[0].plot(indexs, trainX, c="darkred")
  ax[0].set_title("Acoustic")
  ax[0].set_xlabel("Index")
  ax[0].set_ylabel("X value");


  #ax[1].plot(freqsX, magniX, c="skyblue")
  ax[1].bar(freqsX[1:npoints//2], magniX[1:npoints//2], color="skyblue",align='center')
  ax[1].set_title("Acoustic fourier transform")
  #ax[1].set_xlim(-1, 1);
  ax[1].set_xlabel("freqency")
  ax[1].set_ylabel("Amplitude");

  ax[2].plot(indexs, backoX, c="darkred")
  ax[2].set_title("Acoustic iFFT")
  ax[2].set_xlabel("Index")
  ax[2].set_ylabel("X value");

  magniX_max = np.max(magniX[1:])
  print("DEBUG: max mag: ", magniX_max)
  cutedX = [ ]
  for i, a in enumerate(fullfX):
    if i %100 == 0:  print("DEBUG: i =", i, " current mag: ", magniX[i], " a = ", a )
    if magniX[i] > 0.25 * magniX_max: cutedX.append( a )
    else : cutedX.append( 0 )
    

  bcutsX = fftpack.ifft( cutedX )
  bcutsX = bcutsX.real
  print("DEBUG cutted size ", len(cutedX), " and ", len(bcutsX) )

  #ax[1].plot(freqsX, magniX, c="skyblue")
  ax[3].bar(freqsX[1:npoints//2], np.abs( cutedX[1:npoints//2] ), color="skyblue",align='center')
  ax[3].set_title("Acoustic fourier transform after cut")
  ax[3].set_xlabel("freqency")
  ax[3].set_ylabel("Amplitude");


  ax[4].plot(indexs, bcutsX, c="darkred")
  ax[4].set_title("Acoustic iFFT cutted")
  ax[4].set_xlabel("Index")
  ax[4].set_ylabel("X value");


 
  # ax[2].plot(indexs, trainY, c="mediumseagreen")
  # ax[2].set_title("Time to failure")
  # ax[2].set_xlabel("Index")
  # ax[2].set_ylabel("Y value");
  # #ax[2].set_ylim(-200, 200);

  plotname = "plot/detail/s"+str(istart)+"_np" + str( npoints ) + ".png"
  plt.savefig(plotname)
  
  return None
  #plt.plot(range(0,len(trainY)), trainY[:,0], c="darkred")
  #plt.plot(range(0,len(trainY)), trainPredict[:,0], c="mediumseagreen")
  plt.plot(range(0,len(trainY)), [trainPredict[i,0] - trainY[i,0] for i in range(0,len(trainY))], c="mediumseagreen")
  plt.title("Prediction vs measured")
  plt.xlabel("Index")
  plt.ylabel("Time to quake (ms)");
  #plt.ylim(0, 2.);
  plt.savefig('plot/train_0.png')


  #plt.clf()
  #plt.plot(range(0,len(testY)), [testPredict[i,0] - testY[i,0] for i in range(0,len(testY))], c="darkred")
  #plt.title("Prediction vs measured")
  #plt.xlabel("Index")
  #plt.ylabel("Time to quake (ms)");
  ##plt.ylim(0, 2.);
  #plt.savefig('plot/test_0.png')


def main():
  '''
    Main function.

    :Example: python rnn.py [ data/train.csv ]
  '''
  make_plot()
  make_plot( istart = 1000)
  make_plot( istart = 2000)
  make_plot( istart = 5000)
  make_plot( istart = 10000)
  make_plot( istart = 20000)
  make_plot( istart = 50000)
  make_plot( istart = 100000)

if __name__ == '__main__' :
  main()


