from matplotlib.pyplot import axes
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('online_payment.csv')
del df['nameOrig']
del df['nameDest']
X = df.drop(['isFraud'],axis=1)
y = df['isFraud']
le = LabelEncoder()
var = X.select_dtypes(include='object').columns
for i in var:
    X[i] = le.fit_transform(X[i])
model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0,learning_rate=0.5,max_depth=6,sampling_method='gradient_based',max_bin=768,min_child_weight=1,gamma=0.2)
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.30 , random_state=672,stratify=y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("Train Accuracy: ",model.score(X_train,y_train))
print("Test Accuracy: ",model.score(X_test,y_test))    
import pickle
pickle.dump(model,open('modelXGB.pkl','wb'))




