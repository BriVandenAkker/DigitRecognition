# -*- coding: utf-8 -*-

#%%
import os
import pandas as pd
import numpy as np
import random
import sys

import scipy
import sklearn

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix as skm_conf_mat
import sklearn.metrics as skm

import datetime as DT

#%%
projFld = "C:/Users/brian/Documents/ADEC7430_Spring2019/Midterm2"
codeFld = os.path.join(projFld, "PyCode")
fnsFld = os.path.join(codeFld,"_Functions")
outputFld = os.path.join(projFld, "Output")
rawDataFld = os.path.join(projFld, "RawData")
savedDataFld = os.path.join(projFld, "SavedData")

fnList = [
         "fn_logMyInfo"
        ,"fn_confusionMatrixInfo"
        ,"fn_MakeDummies"
        ,"fn_InfoFromTree"
        ] 
for fn in fnList:
    exec(open(os.path.join(fnsFld, fn + ".py")).read())

#@@ see how the functions are documented for quick review prior to use - help your own future self...
print(fn_MakeDummies.__doc__)

