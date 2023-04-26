from flask import Flask, request, jsonify
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))


app = Flask(__name__)

@app.route('/predict', methods=['POST'] )
def predict():
    SGravity = request.form.get('Specific Gravity')
    Albumin = request.form.get('Albumin')
    Hemoglobin = request.form.get('Hemoglobin')
    Rbcc = request.form.get('Red Blood Cell Count')
    Hypertension = request.form.get('Hypertension')

    input_query = np.array([[SGravity,Albumin,Hemoglobin,Rbcc,Hypertension]],dtype=object)

    result = model.predict(input_query)[0]

    return jsonify({'Chronic disease detected': str(result)})

if __name__ == '__main__':
    app.run(debug=True)
