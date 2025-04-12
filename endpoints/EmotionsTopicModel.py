import asyncio
import logging
from json import dumps
from flask import request, jsonify, Response
from flask_restful import Resource

from endpoints.emotion_app_call import train_evaluate_model, load_model
from endpoints.utils import log_request, response_object

logger = logging.getLogger('Emotions CLF MS')


class LoadTopicModel(Resource):
    """
    POST /topic_clf/LoadTopicModel?model_name=""
    Train topic classification model
    """

    @staticmethod
    def post():
        log_request(request)
        logger.info(f"Endpoint: POST /TrainEmotionsModel/TrainEmotionsModel Triggered with request {request}.")
        model_name = request.args.get('model_name')
        text = request.form.get('text')
        response, status_code = load_model(model_name, text)
        logger.info(f"Endpoint: POST /TrainEmotionsModel/TrainEmotionsModel respond with {response}.")


        return Response(dumps(response), content_type="application/json", status=status_code)