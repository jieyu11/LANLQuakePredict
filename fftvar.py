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

from matplotlib.ticker import MaxNLocator


#from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#majorLocator   = MultipleLocator(20)
#xFormatter = FormatStrFormatter('%d')
#yFormatter = FormatStrFormatter('%.7f')
#minorLocator   = MultipleLocator(5)
#


def make_fft( X ):
  n = len(X)
  fullfX = fftpack.fft( X )
  magniX = np.abs( fullfX ) # magnitude: r^2 + i^2
  freqsX = fftpack.fftfreq(len(X)) * n # same for all inputs X as long as they have same length
  # only return the positive part of the frequencies.
  return magniX[0:n//2], freqsX[0:n//2]

def make_plot( ftrain="data/train.csv", istart = 0, npoints = 20000, nperiod=100, nfilter = 1):

  print( "Running @", str( datetime.datetime.now() ), sep="" )
  print( "Index to start = ", istart, sep="" )
  print( "Total number of data points used = ", npoints, sep="" )
  print( "Each period data points used = ", nperiod, sep="" )
  #nrows=nTest + testStart + 1, 
  dataset = pd.read_csv(ftrain, nrows = istart + npoints + 1, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
  dataset.rename({"acoustic_data": "X", "time_to_failure": "Y"}, axis="columns", inplace=True)

  trainX = dataset.X.values[istart : istart + npoints + 1]
  trainY = dataset.Y.values[istart : istart + npoints + 1] 
  # some variables from X to be tested if they have any correlation with Y
  xmean = []
  xfreq = []
  xmagn = [] # frequency list's magnitues
  yreff = []
  idall = range(0, npoints // nperiod )
  for idx in idall:
    yreff.append( trainY[ (idx+1)*nperiod ] ) # use the point after nperiod points as y
    X = trainX[idx * nperiod : (idx+1) * nperiod] 
    xmean.append( np.mean( X ) )
    _xm, _xf = make_fft( X )
    xmagn.append( _xm )
    xfreq.append( _xf )

  if len(xmean) == 0:
    print("ERROR: no output, something is wrong.")
    return None

  

  plt.clf()
  plt.bar(xfreq[0], xmagn[0], color="skyblue",align='center')
  plt.title("Frequency to Magnitue")
  plt.xlabel("Frequency of FFT")
  plt.ylabel("Magnitude of FFT");
  plt.savefig('plot/fft/s'+str(istart)+'_p'+str(nperiod)+'_n'+str(npoints)+'_FreqVsMag_idx0.png')

  plt.clf()
  fig, ax = plt.subplots(2,1, figsize=(20,20))
  ax[0].plot(idall, yreff, '--', color="green")
  ax[0].set_title("Y ~ index")
  ax[0].set_xlabel("index")
  ax[0].set_ylabel("Y time to failure");
  ax[1].plot(idall, xmean, 'ro', color="skyblue", alpha=0.3)
  ax[1].set_title("mean(X) ~ index")
  ax[1].set_xlabel("index")
  ax[1].set_ylabel("mean of X")
  plt.savefig('plot/fft/s'+str(istart)+'_p'+str(nperiod)+'_n'+str(npoints)+'_YvsXmean.png')

  # number of frequency points
  nfreq = len(xfreq[0])

  # plt.xaxis.set_major_locator(majorLocator)
  # plt.xaxis.set_major_formatter(xFormatter)
  # plt.yaxis.set_major_formatter(yFormatter)
  # #for the minor ticks, use no labels; default NullFormatter
  # plt.xaxis.set_minor_locator(minorLocator)

  for i in range(0, nfreq, nfilter):
    #xf = [ xfreq[j][i] for j in range(0, npoints // nperiod) ]
    #plt.clf()
    #plt.plot(xf, yreff, 'ro', color="skyblue")
    #plt.title("Y ~ frequency idx = {d} ".format(i) )
    #plt.xlabel("Frequencies index = {d} ".format(i))
    #plt.ylabel("Y time to failure");
    #plt.savefig('plot/fft/s'+istart+'_p'+period+'_YvsFreqIdx'+str(i)+'.png')

    xm = [ xmagn[j][i] for j in range(0, npoints // nperiod) ]
    plt.clf()
    fig, ax = plt.subplots(2,1, figsize=(20,20))
    ax[0].plot(idall, yreff, '--', color="green")
    ax[0].set_title("Y ~ index")
    ax[0].set_xlabel("index")
    ax[0].set_ylabel("Y time to failure");
    ax[1].plot(idall, xm, 'ro', color="skyblue", alpha=0.3)
    ax[1].set_title("magnitude idx = {} ~ index".format(i) )
    ax[1].set_xlabel("index");
    ax[1].set_ylabel("Magnitudes index = {} ".format(i))
    plt.savefig('plot/fft/s'+str(istart)+'_p'+str(nperiod)+'_n'+str(npoints)+'_YvsMagnIdx'+str(i)+'.png')

    #plt.clf()
    #plt.plot(yreff, xm, 'g^', color="lightgreen")
    #plt.title("magnitude idx = {} ~ Y ".format(i) )
    #plt.ylabel("Magnitudes index = {} ".format(i))
    #plt.xlabel("Y time to failure");
    #plt.savefig('plot/fft/s'+str(istart)+'_p'+str(nperiod)+'_n'+str(npoints)+'_MagnIdx'+str(i)+'VsY.png')

  return None

def main():
  '''
    Main function.

    :Example: python rnn.py [ data/train.csv ]
  '''
  #make_plot( istart = 0, npoints =   20000, nperiod=100, nfilter = 10)
  #make_plot( istart = 0, npoints = 1000000, nperiod=100, nfilter = 10)
  #make_plot( istart = 0, npoints = 50000000, nperiod=100, nfilter = 10)
  make_plot( istart = 0, npoints = 600000000, nperiod=100, nfilter = 10)

if __name__ == '__main__' :
  main()


