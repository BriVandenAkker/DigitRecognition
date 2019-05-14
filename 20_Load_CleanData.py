# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:02:52 2019
@author: brian
Purpose: Prepare the data to maximize model training potential
Inputs: Digit data
Outputs: (1) Scaled data as defined by the scaling function and (2) A function which splits the digit data
         into a 50/50 subset with respect to the value of interest to lower the no information rate.
         Example: When training to predict a '0' we use a training set which contains a 50/50 split between
                  0's and values 1-9. 
"""

#%%
#Import Data
from sklearn.model_selection import train_test_split as skl_traintest_split
from sklearn.utils import shuffle

kdigits = pd.read_csv(os.path.join(rawDataFld, "KDigits_train.csv"))

kdigits.shape
kdigits.head() #784 pixels in 28x28 frame
kdigits.max()  #Pixel values cover full range 0-255

train, test = skl_traintest_split(kdigits.copy(), test_size = 0.20, random_state = 2019)
train.shape
test.shape

#%% Scaling for better comparison

#Function:
#Purpose: Increase comparability b/w digit data
#Philosophy: Digits with homogenized pixel magnitude/variance will improve model performance.
#Input: train/test dataframe
#Output: Scaled dataframe
def scale(data):
    X = data.drop(columns = {'label'})
    #Scale variance
    stdev = np.std(X, axis = 1)
    X_scaled = X.mul(1/stdev, axis = 0)
    #np.std(X_scaled, axis = 1) #Consistent variance across digits
    
    #Scale magnitude
    max_scale = X_scaled.max(axis=1) #Generate a vector of max values from each row
    X_scaled = X_scaled.mul(1/max_scale, axis=0) #Scale from 0-1 for each digit
    
    #np.std(X_train_scaled, axis = 1) # No longer consistet variance b/c we scaled the magnitude
    #This shows scaling variance does nothing b/c we also scaled magnitude. No difference in variance after magnitudes scaled
    #np.mean(np.std(X_train.mul(1/X_train.max(axis=1),axis =0),axis=1) - np.std(X_train_scaled, axis = 1))
    
    #Add labels back
    data = pd.concat([data['label'],X_scaled], axis = 1)
    return data

train = scale(train)
test = scale(test)

#Save Scaled Data to Folder
scaledTrain = os.path.join(savedDataFld, "scaledTrain")
scaledTest = os.path.join(savedDataFld, "scaledTest")
train.to_pickle(scaledTrain + ".pkl")
train.to_csv(scaledTrain +".csv", index = None)
test.to_pickle(scaledTest + ".pkl")
test.to_csv(scaledTest +".csv", index = None)

#%%
# split Train/Test for ~50/50 bifurcation

#Function:
#Purpose: Create train/test dataframe of ~50/50 sample for the value to be predicted.
#Philosophy: This will lower the no information rate which might allow us to select better 
#            model parameters.
#Inputs: (1a) training series or (1b) test series, (2) the value identifyer, (3) T/F is this a training series?
#Outputs: (1) Array of T/F for the specified value, (2) Dataframe of predictors X associated with (1)
def bifurcateDigits(whichSeries, value, train = True):
    temp = whichSeries.copy()
    if(value >= 10):
        print("Value's only range from 0-9")
        return
    #Boolean for label = value
    temp['label'] = (temp['label'] == value).astype(int)
    #Split T/F
    x1 = temp[temp['label'] == 1]
    x2 = temp[temp['label'] == 0]
    #Identify proportion of value to not value
    prop = len(x1)/len(x2)
    #Randomly select a subset of size prop from x2 (value = False)
    random.seed(2019)
    rndindex = [random.uniform(0,1) for x in range(x2.shape[0])]
    rndindex = [True if x < prop else False for x in rndindex]
    x2 = x2.loc[rndindex]
    #Shuffled subset of original with ~50/50 value to not value
    temp = shuffle(x1.append(x2))
    #Isolate dependent var from predictors for either train/test data
    if(train == False):
        global y_test, X_test
        y_test = temp['label']
        X_test = temp.drop(columns = {'label'})
    else:
        global y_train, X_train
        y_train = temp['label']
        X_train= temp.drop(columns = {'label'})

bifurcateDigits(train, 0, train = True)

#%%

#Returns y/n for 'value' without adjusting for 90/10 split
def biasedBifurcatedDigits(whichSeries, value, train = True):
    temp = whichSeries.copy()
    if(value >= 10):
        print("Value's only range from 0-9")
        return
    #Boolean for label = value
    temp['label'] = (temp['label'] == value).astype(int)
    #Split T/F
    if(train == False):
        global y_test, X_test
        y_test = temp['label']
        X_test = temp.drop(columns = {'label'})
    else:
        global y_train, X_train
        y_train = temp['label']
        X_train= temp.drop(columns = {'label'})
        
#%% 
#View data
#Function:
#Purpose: View 28x28 image of pixels for specified digit.
#Input: Dataframe of pixel values, some threshold
#Output: Image of pixels with varying magnitude for handwritten digit with the optional
#        adjustment by threshold
def viewDigits(whichSeries, threshold=None):
    whichSeries = np.array(whichSeries, dtype='float')
    if threshold is not None:
        whichSeries = (whichSeries > threshold).astype(int)
    plt.imshow(whichSeries.reshape((28,28)), cmap='gray')

viewDigits(X_train.iloc[100])


