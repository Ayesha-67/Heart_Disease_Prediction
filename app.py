from flask import Flask
from flask import render_template,request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#Prediction Function
def ValuePredictor(to_predict_list):
    if len(to_predict_list) != 15:
        raise ValueError(f"Expected 15 features, got {len(to_predict_list)}")
    
    # Load trained model and scaler
    loaded_model = pickle.load(open("heart_prediction.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))

    # Convert input list to NumPy array and reshape
    to_predict = np.array(to_predict_list, dtype=float).reshape(1, -1)
    
    # Scale the input data
    to_predict_scaled = scaler.transform(to_predict)

    # Predict using the trained model
    result = loaded_model.predict(to_predict_scaled)
    return result[0]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/index', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method=='POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        
        to_predict_list = list(map(float, to_predict_list))
        
        result = ValuePredictor(to_predict_list)
        
        if int(result) == 1:
            prediction = "You are likely to suffer from Heart diseases"
        else:
            prediction = " Congrats!! You are not suffering from heart diseases"
        return render_template("result.html", prediction = prediction)
    
if __name__ == "__main__":
    app.run(debug=True)