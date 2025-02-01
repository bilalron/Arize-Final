from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Document QA API"
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-west-2"
    pinecone_api_key: str
    pinecone_region: str = "us-east-1"
    pinecone_cloud: str = "aws"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()