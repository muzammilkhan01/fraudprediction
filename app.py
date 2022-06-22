from flask import Flask ,render_template, request
from sympy import re
import pandas as pd
import numpy as np

import model

app = Flask(__name__)


@app.route("/",methods=["GET","POST"])
def pred():
    if request.method == "POST":
        var1=int(request.form['step'])
        var2=int(request.form['type'])
        var3=float(request.form['amount'])
        var4=float(request.form['oldbalanceOrg'])
        var5=float(request.form['newbalanceOrig'])
        var6=float(request.form['oldbalanceDest'])
        var7=float(request.form['newbalanceDest'])
        var8=float(request.form['isFlaggedFraud'])
        data = {
        'step':[var1],
        'type':[var2],
        'amount':[var3],
        'oldbalanceOrg':[var4],
        'newbalanceOrig':[var5],
        'oldbalanceDest':[var6],
        'newbalanceDest':[var7],
        'isFlaggedFraud':[var8]
        }
        columns = ['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud']
    
        details = pd.DataFrame(data,columns=columns)
        prediction = model.fraud_prediction(details)
        if prediction==0:
            
            return render_template("legit.html")
        else:
            
            return render_template("fraudulent.html")

    
    return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True)
    

