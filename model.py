import pickle
import numpy as np
import pandas as pd
import xgboost as xgb





def fraud_prediction(details):
    modelXGB = pickle.load(open('modelXGB.pkl','rb'))
    test = details
   


    return modelXGB.predict(test)
