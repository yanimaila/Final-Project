import numpy as np
import pandas as pd
from process import preparation, generate_response
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

loaded_model = joblib.load("model/randomforest.sav")

# download nltk
preparation()

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/prediksi')
def pred():
    return render_template('prediksi.html')

@app.route('/prediksi', methods=['POST'])
def prediksi():
    um, jm, Mood, Sering_Pusing, Sering_Menangis, Sulit_Tidur, Pola_Makan, Sering_Gelisah = [x for x in request.form.values()]

    data = []

    data.append(float(Mood))
    data.append(float(Sering_Pusing))
    data.append(float(Sering_Menangis))
    data.append(float(Sulit_Tidur))
    data.append(float(Pola_Makan))
    data.append(float(Sering_Gelisah))

    data = np.array(data)
    
    #reshape array
    data = data.reshape(1,-1)

    prediction = loaded_model.predict(data)

    if prediction==[0]:
        hasil = 'CUKUP SEHAT'
    elif prediction==[1]:
        hasil = 'KURANG SEHAT'
    elif prediction==[2]:
        hasil = 'SEHAT'
    elif prediction==[3]:
        hasil = 'TIDAK SEHAT'

    prediction = str(hasil)
    
    return render_template('hasil.html', hasil_prediksi=hasil)
    
    
@app.route('/chatbot')
def bot():
    return render_template('chatbot.html')

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result
    
if __name__ == '__main__':
    app.run(debug=True)
