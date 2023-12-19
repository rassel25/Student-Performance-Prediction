import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('Student_Performance_Prediction')

@app.route('/predict', methods=['POST'])
def predict():
    individual = request.get_json()
    student_math_score = model.predict(individual)

    result = round(student_math_score[0])

    output = {
        'The student has scored in Math': int(result)
    }

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)



# Use this code to run Flask app: poetry run python -m waitress --listen=0.0.0.0:9696 predict:app
