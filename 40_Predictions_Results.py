# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:48:56 2019
@author: brian
Purpose: Use Random Forest to make Global Predictions
Inputs: Digits Test Data, 10 RF models
Output: (1) Digit Predictions and (2) Performance Measures
"""
#%%
#Make Global Predictions
testX = test.drop(columns = {'label'})
testY = test['label']                              

#Generate dictionary of final probabalistic predictions
globalpred = {}
for i in digit:          
    globalpred[i] = modellist[i].predict_proba(testX)[:,1]   

#Set to dataframe
predictions = pd.DataFrame(globalpred)
#Select value with highest predictive confidence and store in predValue
predValue = predictions.idxmax(axis = 1)

#Write predictions to output folder
predictFile = os.path.join(outputFld, "digit_predictions")
predValue.to_pickle(predictFile + ".pkl")
predValue.to_csv(predictFile+".csv", index = None)

#Show the confusion matrix
cfmatrix = confusion_matrix(testY, predValue)

#Calculate the global accuracy
tptn = 0
for i in digit:
    tptn += cfmatrix[i,i]
accuracy = tptn/len(testY)

#Call classification report
target_names = ['0','1','2','3','4','5','6','7','8','9']
report = classification_report(testY, predValue, target_names = target_names)

#%%
#Summary of Results:
#(1) Training RF on biased (90/10) and unscaled data with non-optimized parameters: Accuracy = 0.893
#(2) Training RF on unbiased (50/50) and scaled data with non-optimized parameters: Accuracy = 0.907
#(3) Training RF on unbiased (50/50) and scaled data with locally optimized parameters: Accuracy = 0.931

#Expansion of Results for (3)
print('Global Confusion Matrix: ' + '\n'  + str(cfmatrix) + '\n')
#Global Confusion Matrix: 
#[[794   0   0   0   4   0   3   0   6   0]
# [  0 862   5   4   3   2   2   3   2   0]
# [  6   2 785   8  11   0  16  14   7   9]
# [  1   6  22 795   2  16   4   8  19  11]
# [  3   2   1   0 782   1   4   1   4  34]
# [ 15   5   2  14   3 711  16   1   6   8]
# [ 13   3   1   0   3   3 772   0  12   0]
# [  6   6   9   1   7   0   1 794   5  31]
# [  0  11   4  18   3   8   7   3 775  18]
# [  3   2   2  18  33   0   0  17  13 753]]

print('Global Accuracy: ' + str(accuracy) + '\n')
#Global Accuracy: 0.9313

print('Classification Report: ' + '\n' + str(report))
#Classification Report: 
#              precision    recall  f1-score   support
#
#           0       0.94      0.98      0.96       807
#           1       0.96      0.98      0.97       883
#           2       0.94      0.91      0.93       858
#           3       0.93      0.90      0.91       884
#           4       0.92      0.94      0.93       832
#           5       0.96      0.91      0.93       781
#           6       0.94      0.96      0.95       807
#           7       0.94      0.92      0.93       860
#           8       0.91      0.91      0.91       847
#           9       0.87      0.90      0.88       841
#
#   micro avg       0.93      0.93      0.93      8400
#   macro avg       0.93      0.93      0.93      8400
#weighted avg       0.93      0.93      0.93      8400