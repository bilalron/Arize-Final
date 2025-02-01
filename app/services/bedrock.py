import boto3
import json
from typing import List
from ..config import settings

class BedrockService:
    def __init__(self):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )

    def get_embedding(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=body
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding')

    def get_llm_response(self, prompt: str) -> str:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens_to_sample": 512,
            "temperature": 0.7,
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        })
        response = self.client.invoke_model(
            modelId='anthropic.claude-v2',
            body=body
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get('completion', '').strip()