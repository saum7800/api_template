from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import numpy as np
import tensorflow as tf
import traceback
import json

app = Flask(__name__)
api = Api(app)

model = tf.keras.models.load_model('model/model.h5')


# convert request_input dict to input accepted by model.
def parse_input(request_input):
    request_list=request_input.values()
    request_list=list(request_list)
    return np.array([request_list])


# convert model prediction to dict to return as JSON
def parse_prediction(prediction):
    prediction=np.argmax(prediction,axis=1)
    prediction_index=prediction[0]
    output_json={ "prediction":int(prediction_index) }
    # print(output_json)
    return output_json

class MakePrediction(Resource):
    @staticmethod
    def post():
        if model:
            try:
                request_input = request.get_json()

                model_input = parse_input(request_input)

                prediction = model.predict(model_input)

                model_output = parse_prediction(prediction)

                return jsonify(**model_output)

            except:

                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({'trace': 'No model found'})


api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True,port=8080)
