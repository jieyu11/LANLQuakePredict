# Introduction
Kaggle challenge, found [here](https://www.kaggle.com/c/LANL-Earthquake-Prediction).

# Run the code
1. Run the trainning:
```bash
python3 rnn.py -make [-train train.csv] [-look_back 3]
```
where,
 * -make: making the model
 * -train: set up training input, default is data/train.csv
 * -look_back: set up the number of X data points used to predict Y

2. Run the testing:
```
python3 rnn.py -pred [-test tests.txt] [-look_back 3]
```
where,
 * -pred: do prediction using the giving model
 * -test: text file with all the test segment data locations
 * -look_back: same as above, must be consistent with the built model

# Results
The prediction results are saved in sample_submission.csv, which 
can be submitted to Kaggle.

