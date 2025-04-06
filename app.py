from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pickle
import gzip

app = Flask(__name__)

def stroke1(ll):
    

    features = np.array(ll).reshape(1, -1)
    pred_c = model_c.predict(features)[0]
    pred_r = model_r.predict(features)[0]  # Risk percentage

    return pred_c, pred_r

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/user_inputs', methods=['POST'])
def user_inputs():
    if request.method == 'POST':
        a = int(request.form['age'])
        b = int(request.form['gender'])
        c = int(request.form['chest_pain'])
        d = int(request.form['shortness_of_breath'])
        e = int(request.form['irregular_heartbeat'])
        f = int(request.form['fatigue_weakness'])
        g = int(request.form['dizziness'])
        h = int(request.form['swelling_edema'])
        i = int(request.form['neck_jaw_pain'])
        j = int(request.form['excessive_sweating'])
        k = int(request.form['persistent_cough'])
        l = int(request.form['nausea_vomiting'])
        m = int(request.form['high_blood_pressure'])
        n = int(request.form['chest_discomfort'])
        o = int(request.form['cold_hands_feet'])
        p = int(request.form['snoring_sleep_apnea'])
        q = int(request.form['anxiety_doom'])

        # Create input list for the model
        ls = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q]

        # Get predictions
        pred_c, pred_r = stroke1(ls)

        # Define risk status
        risk_status = "Yes" if pred_c == 1 else "No"

        # Redirect to result page with data
        return redirect(url_for("result", status=risk_status, percent=pred_r))

@app.route("/result")
def result():
    status = request.args.get("status", "No Data")
    percent = request.args.get("percent", "0")
    xx=float(percent)
    percent1=round(xx,1)
    return render_template("result.html", status=status, percent=str(percent1))

if __name__ == "__main__":
    with gzip.open(r"C:\Users\subra\OneDrive\Desktop\Development\Stroke_Prediction_Model\model_compressed.pkl.gz", "rb") as f:
        model_c = pickle.load(f)
    with gzip.open(r"C:\Users\subra\OneDrive\Desktop\Development\Stroke_Prediction_Model\model_compressed_re.pkl.gz", "rb") as f:
        model_r = pickle.load(f)
    app.run(debug=True)
