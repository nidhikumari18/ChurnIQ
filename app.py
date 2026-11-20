from pathlib import Path
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for
from joblib import load

app=Flask(__name__)
app.secret_key="local-churniq-demo-key"
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dataset"/"WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL=ROOT/"model"/"artifacts"/"best_model.joblib"

def get_data():
    if not DATA.exists(): return None
    df=pd.read_csv(DATA,dtype={"TotalCharges":"string"})
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"].str.strip(),errors="coerce")
    return df

@app.get("/")
def home(): return render_template("index.html")

@app.get("/health")
def health(): return {"status":"ok","application":"ChurnIQ"}

@app.get("/dashboard")
def dashboard():
    df=get_data()
    if df is None:
        return render_template("dashboard.html",stats=None,error="Add the IBM Telco CSV to the dataset folder.")
    n=len(df); churn=int(df["Churn"].eq("Yes").sum())
    ct=pd.crosstab(df["Contract"],df["Churn"]).reindex(["Month-to-month","One year","Two year"],fill_value=0)
    it=pd.crosstab(df["InternetService"],df["Churn"])
    stats={"total":n,"churn":churn,"retained":n-churn,"rate":round(churn/n*100,1),
      "contract_labels":ct.index.tolist(),"contract_yes":ct.get("Yes",pd.Series(0,index=ct.index)).tolist(),
      "contract_no":ct.get("No",pd.Series(0,index=ct.index)).tolist(),
      "internet_labels":it.index.tolist(),"internet_yes":it.get("Yes",pd.Series(0,index=it.index)).tolist(),
      "model_ready":MODEL.exists()}
    return render_template("dashboard.html",stats=stats,error=None)

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="GET": return render_template("predict.html")
    if not MODEL.exists():
        flash("Train the model first: python model/train_model.py","error")
        return redirect(url_for("predict"))
    cats=["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
    row={c:request.form.get(c,"") for c in cats}
    row.update({"SeniorCitizen":int(request.form.get("SeniorCitizen",0)),"tenure":int(request.form.get("tenure",0)),
      "MonthlyCharges":float(request.form.get("MonthlyCharges",0)),"TotalCharges":float(request.form.get("TotalCharges",0))})
    model=load(MODEL); x=pd.DataFrame([row])
    p=float(model.predict_proba(x)[0][list(model.classes_).index("Yes")]); label=model.predict(x)[0]
    pct=round(p*100,1); risk="High" if pct>=60 else "Medium" if pct>=30 else "Low"
    factors=[]
    if row["Contract"]=="Month-to-month": factors.append("Month-to-month contract")
    if row["tenure"]<12: factors.append("Short customer tenure")
    if row["TechSupport"]=="No" and row["InternetService"]!="No": factors.append("No technical support subscription")
    if row["PaymentMethod"]=="Electronic check": factors.append("Electronic check payment method")
    if not factors: factors=["No common risk signal from this profile"]
    advice="Consider a personal retention call, a suitable contract offer and proactive support." if risk=="High" else "Monitor engagement and offer relevant service or support options." if risk=="Medium" else "Maintain service quality and regular customer engagement."
    return render_template("result.html",probability=pct,risk=risk,prediction=label,factors=factors,advice=advice)

if __name__=="__main__": app.run(debug=True)
