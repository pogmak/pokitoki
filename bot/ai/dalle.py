"""DALL-E model from OpenAI."""

import json
import logging
import time

import requests

from bot.config import config
logger = logging.getLogger(__name__)

kandinsky_api_key = config.kandinsky.get("api_key")
kandinsky_secret_key = config.kandinsky.get("secret_key")


class Text2ImageAPI:

    def __init__(self, url, api_key, secret_key):
        self.URL = url
        self.AUTH_HEADERS = {
            'X-Key': f'Key {api_key}',
            'X-Secret': f'Secret {secret_key}',
        }

    def get_model(self):
        logger.info(f"Fetching model")
        response = requests.get(self.URL + 'key/api/v1/models', headers=self.AUTH_HEADERS)
        data = response.json()
        return data[0]['id']

    def generate(self, prompt, model, style, images=1, width=1024, height=1024):
        logger.info(f"Generating image")
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            "generateParams": {
                "query": f"{prompt}"
            },
            "style": style,
        }

        data = {
            'model_id': (None, model),
            'params': (None, json.dumps(params), 'application/json')
        }
        response = requests.post(self.URL + 'key/api/v1/text2image/run', headers=self.AUTH_HEADERS, files=data)
        data = response.json()
        return data['uuid']

    def check_generation(self, request_id, attempts=10, delay=10):
        while attempts > 0:
            logger.info(f"Checking status of generation attempt is {10 - attempts + 1}")
            response = requests.get(self.URL + 'key/api/v1/text2image/status/' + request_id, headers=self.AUTH_HEADERS)
            data = response.json()
            logger.info(f"Status: {data}")
            if data['status'] == 'DONE':
                return data['images'][0]

            attempts -= 1
            time.sleep(delay)


class Model:
    """OpenAI DALL-E wrapper."""

    async def imagine(self, prompt: str, style: str) -> str:
        """Generates an image of the specified size according to the description."""
        api = Text2ImageAPI('https://api-key.fusionbrain.ai/',
                            api_key=kandinsky_api_key,
                            secret_key=kandinsky_secret_key)
        model_id = api.get_model()
        uuid = api.generate(prompt, model_id, style=style)
        image = api.check_generation(uuid)

        if len(image) == 0:
            raise ValueError("received an empty answer")
        return image