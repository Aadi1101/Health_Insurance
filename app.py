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
    insurance = process_and_predict(input_text=input_data,json_data=json_data)
    return {'predicted':insurance}

def process_and_predict(input_text,json_data):
    if(json_data==True):
        output_text = [item for item in input_text['data'].split(',')]
    else:
        output_text = [item for item in input_text.split(',')]

    output_text = np.array(output_text).reshape(1, -1)

    age, sex, bmi, children, smoker, region = output_text[0]
    age = int(age)
    bmi = float(bmi)
    children = int(children)

    features = pd.DataFrame({
        'age':[age],
        'sex':[sex],
        'bmi':[bmi],
        'children':[children],
        'smoker':[smoker],
        'region':[region]
    })

    with open('src/models/preprocessor.pkl', 'rb') as p:
        preprocessor = dill.load(p)
    print("PREPROCESSOR LOADED....")
    output_text_dims = preprocessor.transform(features)
    print('OUTPUT TEXT DIMS....::', output_text_dims)

    with open('src/models/best_model.pkl', 'rb') as m:
        model = dill.load(m)

    insurance = model.predict(output_text_dims)
    return insurance[0]

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)