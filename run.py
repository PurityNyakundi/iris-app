import numpy as np
from flask import Flask, request,  render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():

    int_features = [float(y) for y in request.form.values()]
    final_features = [np.array(int_features)]
    #final1 = final_features.reshape(1,-1)
    prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    if prediction==0:
        return render_template('index.html',
                               prediction_text='Iris setosa'.format(prediction),
                               )
    elif prediction==1:
        return render_template('index.html',
                               prediction_text='Iris virginica'.format(prediction),
                              )
    else:
        return render_template('index.html',prediction_text = 'versicolor'.format(prediction),)



if __name__ == "__main__":
    app.run(debug=True)