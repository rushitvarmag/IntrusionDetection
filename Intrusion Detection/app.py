from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def Home():
    return render_template("detection.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]  
    final=[np.array(int_features)]
    prediction = model.predict(final)
    print(prediction)
    return render_template("detection.html", prediction_text = "The attack is {}".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)



