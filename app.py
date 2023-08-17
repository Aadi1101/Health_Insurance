from flask import Flask,request,render_template
import dill
import numpy as np
import pandas as pd


app=Flask('__name__')
@app.route('/')
def read_main():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def generate_output():
    json_data = False
    input_data = request.args.get('data')
    if input_data==None:
        input_data = request.get_json()
        json_data = True
    # input_text = SPX_USO_SLV_EUR_USD_comma_separated_values
    insurance = process_and_predict(input_text=input_data,json_data=json_data)
    return {'predicted':insurance}

def process_and_predict(input_text,json_data):
    # Split the input text into elements
    if(json_data==True):
        output_text = [item for item in input_text['data'].split(',')]
    else:
        output_text = [item for item in input_text.split(',')]

    # Reshape the array
    output_text = np.array(output_text).reshape(1, -1)

    age, sex, bmi, children, smoker, region = output_text[0]
    age = int(age)
    bmi = float(bmi)
    children = int(children)
    # Transform using the preprocessor
    features = pd.DataFrame({
        'age':[age],
        'sex':[sex],
        'bmi':[bmi],
        'children':[children],
        'smoker':[smoker],
        'region':[region]
    })

    # Load the preprocessor
    with open('src/models/preprocessor.pkl', 'rb') as p:
        preprocessor = dill.load(p)
    print("PREPROCESSOR LOADED....")
    output_text_dims = preprocessor.transform(features)
    print('OUTPUT TEXT DIMS....::', output_text_dims)

    # Load the machine learning model
    with open('src/models/best_model.pkl', 'rb') as m:
        model = dill.load(m)

    # Make a prediction using the model
    insurance = model.predict(output_text_dims)
    return insurance[0]

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)