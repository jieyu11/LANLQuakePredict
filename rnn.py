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

def make_model( ftrain="data/train.csv", NPOINTMAX = -1, look_back = 1):

  print( "Step 1 @", str( datetime.datetime.now() ), " start of building a model.", sep="" )
  #nrows=nTest + testStart + 1, 
  dataset = pd.read_csv(ftrain, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
  print( "Step 2 @", str( datetime.datetime.now() ), " input train.csv loaded.", sep="" )
  dataset.rename({"acoustic_data": "X", "time_to_failure": "Y"}, axis="columns", inplace=True)
  npoints = len(dataset.X.values) 

  #
  # in train.csv, there are 17 experiments of lab earth quake.
  # e.g. from index 0 to 5656573, time_to_failure (time to the lab earth quake event) decreases until very close to 0.
  #      at index 5656574, time_to_failure jump to a very high number indicating the next experiment starts. 
  period_idx = [0,5656574,   50085878,  104677356,  138772453,  
                 187641820,  218652630,  245829585,  307838917,  
                 338276287,  375377848,  419368880,  461811623,  
                 495800225,  528777115,  585568144,  621985673, npoints]

  arrTrainX, arrTrainY = [], [] #None, None
  for k in range(1, len(period_idx)):
    if NPOINTMAX > 0 and period_idx[k] > NPOINTMAX: break;
    print("Step 3.",k," @", str( datetime.datetime.now() ), " reading index from ", period_idx[k-1], " to ", period_idx[k])

    arrTrainX_0, arrTrainY_0 = create_dataset( dataset.X.values[ period_idx[k-1] : period_idx[k] ], dataset.Y.values[ period_idx[k-1] : period_idx[k] ], look_back )
    arrTrainX.extend( arrTrainX_0 )
    arrTrainY.extend( arrTrainY_0 )
    #if arrTrainX is None: arrTrainX = arrTrainX_0
    #else:                 arrTrainX.extend( arrTrainX_0 )
    #if arrTrainY is None: arrTrainY = arrTrainY_0
    #else:                 arrTrainY.extend( arrTrainY_0 )

  print( "Step 4 @", str( datetime.datetime.now() ), " build X Y dataset." )
  trainX = np.array(arrTrainX)
  trainY = np.array(arrTrainY)
  print("trainX shape: ", trainX.shape, "trainY shape: ", trainY.shape)
  trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
  trainY = trainY.reshape(trainY.shape[0], 1)
  print("trainX shape: ", trainX.shape, "trainY shape: ", trainY.shape)

  print( "Step 5 @", str( datetime.datetime.now() ), " build training model." )
  model = Sequential()
  model.add(LSTM(4, input_shape=(1, look_back)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

  print( "Step 6 @", str( datetime.datetime.now() ), " fit training model." )
  # make predictions
  trainPredict = model.predict(trainX)

  print("trainY shape: ", trainY.shape)
  print("trainP shape: ", trainPredict.shape)
  

  print( "Step 7 @", str( datetime.datetime.now() ), " save training model." )
  # serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model.h5")

  # calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
  print('Train Score: %.5f RMSE' % (trainScore))

  # fig, ax = plt.subplots(2,1, figsize=(20,12))
  # ax[0].plot(train.index.values, train.timetoquake.values, c="darkred")
  # ax[0].set_title("Quaketime of 10 Mio rows")
  # ax[0].set_xlabel("Index")
  # ax[0].set_ylabel("Quaketime in ms");
  # ax[1].plot(train.index.values, train.measured.values, c="mediumseagreen")
  # ax[1].set_title("Signal of 10 Mio rows")
  # ax[1].set_xlabel("Index")
  # ax[1].set_ylabel("Acoustic Signal");
  # ax[1].set_ylim(-100, 100);

  #plt.plot(range(0,len(trainY)), trainY[:,0], c="darkred")
  #plt.plot(range(0,len(trainY)), trainPredict[:,0], c="mediumseagreen")
  plt.plot(range(0,len(trainY)), [trainPredict[i,0] - trainY[i,0] for i in range(0,len(trainY))], c="mediumseagreen")
  plt.title("Prediction vs measured")
  plt.xlabel("Index")
  plt.ylabel("Time to quake (ms)");
  #plt.ylim(0, 2.);
  plt.savefig('train_0.png')


  #plt.clf()
  #plt.plot(range(0,len(testY)), [testPredict[i,0] - testY[i,0] for i in range(0,len(testY))], c="darkred")
  #plt.title("Prediction vs measured")
  #plt.xlabel("Index")
  #plt.ylabel("Time to quake (ms)");
  ##plt.ylim(0, 2.);
  #plt.savefig('test_0.png')


def load_model( jsmodelname = "model.json", weightname = "model.h5"):
  # load json and create model
  json_file = open( jsmodelname, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights( weightname )
  print("Loaded model from disk")
  return loaded_model
 

def load_testdata( ftest, look_back = 1):
  dataset = pd.read_csv(ftest, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
  dataset.rename({"acoustic_data": "X"}, axis="columns", inplace=True)
  npoints = len(dataset.X.values) 
  #print("Test data: ", ftest, " has ", npoints, " data points.", sep = "")
  arrTestX, __Ynone = create_dataset( dataset.X.values, look_back = look_back )
  testX = np.array(arrTestX)
  #print(" - Test data shape: ", testX.shape)
  return testX.reshape(testX.shape[0], 1, testX.shape[1])
 
def main():
  '''
    Main function.

    :Example: python rnn.py [ data/train.csv ]
  '''

  ftrain = "data/train.csv"
  #5656575
  #ntrain_max = -1 # -1 for all
  ntrain_max = -1
  ftestlist = "tests.txt"
  iarg = 1
  remodel = True
  look_back = 1
  while iarg < len(sys.argv):
    if str(sys.argv[iarg]) == "-train":
      iarg += 1
      ftrain = sys.argv[ iarg ]
    elif str(sys.argv[iarg]) == "-ntrain_max":
      iarg += 1
      ntrain_max = int(sys.argv[ iarg ])
    elif str(sys.argv[iarg]) == "-test":
      iarg += 1
      ftestlist = sys.argv[ iarg ]
    elif str(sys.argv[iarg]) == "-make":
      remodel = True
    elif str(sys.argv[iarg]) == "-pred":
      remodel = False
    elif str(sys.argv[iarg]) == "-look_back":
      iarg += 1
      look_back = int(sys.argv[ iarg ])
    elif str(sys.argv[iarg]) == "-help" or str(sys.argv[iarg]) == "-h":
      print("Argument List: [-train train.csv] [-test tests.txt] [-make] [-pred]")
      return None
    else:
      print("Argument: ", sys.argv[iarg], " not known. Use: [-train train.csv] [-test tests.txt] [-make] [-pred] [-look_back 3]")

    iarg += 1


  print("==============================================")
  print("Read training: ", ftrain)
  print("Number of training data points: ", ntrain_max)
  print("Read testing list file: ", ftestlist)
  print("==============================================")
 
  #
  # make model takes a long time, up to a few days, depending on the size of the training data.
  #
  if remodel:
    make_model(ftrain, NPOINTMAX = ntrain_max, look_back = look_back)
    return None


  #
  # load the saved model with its default names
  #
  model = load_model()

  #
  # read test list file, each line contains a test segmentation inputs
  #
  lines = [line.rstrip('\n') for line in open( ftestlist )] 
  nline = len(lines)
  print("Number of test datasets: ", nline)
  
  df = {'seg_id': [], 'time_to_failure': []}
  for jl, ftest in enumerate( lines ):
    if jl %100 == 0:
      print ( "Finished: ", jl, " out of ", nline )
    testX = load_testdata( ftest, look_back )
    predY = model.predict(testX)
    #print( "Shape of output: ", predY.shape)
    
    #keep the tail of the string: seg_ffe7cc.csv
    k0=len(ftest)
    df['seg_id'].append( ftest[k0 - 14 : k0 - 4] )
    df['time_to_failure'].append( predY[-1][0] )
    #print( "Test dataset: ", ftest[k0 - 14 : k0 - 4], ", predicted next: ", predY[-1][0])

  pddf = pd.DataFrame( df )
  pddf.to_csv('sample_submission.csv', index=False)


if __name__ == '__main__' :
  main()


