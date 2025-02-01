from pydantic import BaseModel
from typing import List, Optional

class IndexRequest(BaseModel):
    url: str
    index_name: Optional[str] = "document-store"

class QueryRequest(BaseModel):
    question: str
    index_name: Optional[str] = "document-store"
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    sources: List[str]