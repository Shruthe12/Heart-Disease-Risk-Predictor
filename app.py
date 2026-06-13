from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"

def train_and_save_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    try:
        df = pd.read_csv(url, names=cols, na_values="?")
    except Exception:
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "age":      np.random.randint(30, 80, n),
            "sex":      np.random.randint(0, 2, n),
            "cp":       np.random.randint(0, 4, n),
            "trestbps": np.random.randint(90, 180, n),
            "chol":     np.random.randint(150, 350, n),
            "fbs":      np.random.randint(0, 2, n),
            "restecg":  np.random.randint(0, 3, n),
            "thalach":  np.random.randint(80, 200, n),
            "exang":    np.random.randint(0, 2, n),
            "oldpeak":  np.round(np.random.uniform(0, 5, n), 1),
            "slope":    np.random.randint(0, 3, n),
            "ca":       np.random.randint(0, 4, n),
            "thal":     np.random.choice([3, 6, 7], n),
            "target":   np.random.randint(0, 2, n),
        })

    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"[Model] Accuracy: {acc*100:.2f}%")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "accuracy": round(acc*100, 2)}, f)

    return model, scaler, round(acc*100, 2)


if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model     = bundle["model"]
    scaler    = bundle["scaler"]
    MODEL_ACC = bundle.get("accuracy", "N/A")
else:
    print("[Model] model.pkl not found – training now...")
    model, scaler, MODEL_ACC = train_and_save_model()


patients = []


@app.route("/")
def index():
    return render_template("index.html", accuracy=MODEL_ACC)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        cp_map    = {"Typical angina": 0, "Atypical angina": 1,
                     "Non-anginal pain": 2, "Asymptomatic": 3}
        fbs_map   = {"Lower than 120 mg/ml": 0, "Greater than 120 mg/ml": 1}
        ecg_map   = {"Normal": 0, "ST-T wave abnormality": 1,
                     "Left ventricular hypertrophy": 2}
        exang_map = {"No": 0, "Yes": 1}
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        thal_map  = {"Normal": 3, "Fixed defect": 6, "Reversable defect": 7}

        age      = float(request.form["age"])
        sex      = float(request.form["sex"])
        cp       = float(cp_map[request.form["cp"]])
        trestbps = float(request.form["trestbps"])
        chol     = float(request.form["chol"])
        fbs      = float(fbs_map[request.form["fbs"]])
        restecg  = float(ecg_map[request.form["restecg"]])
        thalach  = float(request.form["thalach"])
        exang    = float(exang_map[request.form["exang"]])
        oldpeak  = float(request.form["oldpeak"])
        slope    = float(slope_map[request.form["slope"]])
        ca       = float(request.form["ca"])
        thal     = float(thal_map[request.form["thal"]])

        features = np.array([[age, sex, cp, trestbps, chol, fbs,
                               restecg, thalach, exang, oldpeak,
                               slope, ca, thal]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]
        risk_pct = round(prob[1] * 100, 1)

        result     = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"
        risk_level = "high" if prediction == 1 else "low"

        record = {
            "id": len(patients) + 1,
            "age": int(age), "sex": "Male" if sex == 1 else "Female",
            "cp": request.form["cp"], "trestbps": int(trestbps),
            "chol": int(chol), "fbs": request.form["fbs"],
            "restecg": request.form["restecg"], "thalach": int(thalach),
            "exang": request.form["exang"], "oldpeak": oldpeak,
            "slope": request.form["slope"], "ca": int(ca),
            "thal": request.form["thal"],
            "result": result, "risk_pct": risk_pct
        }
        patients.append(record)

        return render_template("result.html",
                               result=result,
                               risk_level=risk_level,
                               risk_pct=risk_pct,
                               record=record)
    except Exception as e:
        return render_template("index.html",
                               error=f"Error: {str(e)}",
                               accuracy=MODEL_ACC)


@app.route("/records")
def records():
    return render_template("records.html", patients=patients)


@app.route("/clear_records", methods=["POST"])
def clear_records():
    patients.clear()
    return redirect(url_for("records"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)