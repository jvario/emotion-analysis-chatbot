from datetime import datetime
import pytz
from flask import Flask, jsonify
from flask_restful import Api
from endpoints.EmotionsTopicModel import LoadTopicModel
from endpoints.TrainEmotionsModel import TrainTopicModel, logger
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
pathRoot = "/emotions_clf"
CORS(app, supports_credentials=True)



api.add_resource(TrainTopicModel, pathRoot + "/TrainEmotionsModel", endpoint="TrainEmotionsModel")
api.add_resource(LoadTopicModel, pathRoot + "/LoadEmotionsModel", endpoint="LoadEmotionsModel")

release_date = str(datetime.now(tz=pytz.timezone("Europe/Athens")))


@app.route(f'/{pathRoot}/health', methods=['GET'])
def health():
    logger.info("Health endpoint Triggered.")
    return jsonify({"emotion_classification service, release_date": release_date})



if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
