import requests, json

class Experiment:
    def __init__(self, url: str):
        self.url = url
        self.inference_times = []
        self.results = []

    @staticmethod
    def _build_request(messages: list, inference_parameters: dict, rag_parameters: dict):
        return {
            "user_name": "",
            "messages": messages,
            "inference_parameters": inference_parameters,
            "rag_parameters": rag_parameters
        }

    def run(self, messages: list, inference_parameters: dict, rag_parameters: dict):
        request_data = self._build_request(messages, inference_parameters, rag_parameters)
        try:
            response = requests.post(self.url, data=json.dumps(request_data))
            if response.status_code == 200:
                self.inference_times.append(response.json().get('inference_time_in_seconds'))
                self.results.append(response.json())
            else:
                print('An error occurred:', response.status_code)
        except requests.exceptions.RequestException as e:
            print('Request failed:', e)