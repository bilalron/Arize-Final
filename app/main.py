from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from bs4 import BeautifulSoup
import boto3
import json
import requests
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(title="Document QA API 1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True  # Set this to False when using "*"
)

# Initialize clients
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION', 'us-west-2'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY')
)

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

def get_embedding(text: str) -> List[float]:
    body = json.dumps({"inputText": text})
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=body
    )
    response_body = json.loads(response.get('body').read())
    return response_body.get('embedding')

def get_llm_response(prompt: str) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens_to_sample": 512,
        "temperature": 0.7,
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
    })
    response = bedrock.invoke_model(
        modelId='anthropic.claude-v2',
        body=body
    )
    response_body = json.loads(response.get('body').read())
    return response_body.get('completion', '').strip()

@app.post("/index")
async def create_index(request: IndexRequest):
    try:
        # Create or get index
        if request.index_name in pc.list_indexes().names():
            pc.delete_index(request.index_name)

        # Create index with ServerlessSpec
        pc.create_index(
            name=request.index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

        # Get PDFs
        read = requests.get(request.url)
        soup = BeautifulSoup(read.content, "html.parser")
        
        pdf_links = []
        for link in soup.find_all('a', href=True):
            if link['href'].endswith('.pdf'):
                pdf_url = link['href']
                if not pdf_url.startswith('http'):
                    pdf_url = requests.compat.urljoin(request.url, pdf_url)
                if pdf_url not in pdf_links and "hw" not in pdf_url:
                    pdf_links.append(pdf_url)

        # Process PDFs
        pages = []
        for link in pdf_links:
            loader = PyPDFLoader(link)
            pages.extend(await loader.aload())

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(pages)

        # Get index
        index = pc.Index(request.index_name)

        # Process chunks
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk.page_content)
            index.upsert(
                vectors=[{
                    'id': f"doc_{i}",
                    'values': embedding,
                    'metadata': {
                        'text': chunk.page_content,
                        'source': chunk.metadata['source']
                    }
                }]
            )

        return {"message": f"Index '{request.index_name}' created and documents processed"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Get query embedding
        query_vector = get_embedding(request.question)
        
        # Search in index
        index = pc.Index(request.index_name)
        results = index.query(
            vector=query_vector,
            top_k=request.top_k,
            include_metadata=True
        )
        
        if not results.matches:
            return QueryResponse(
                response="No relevant information found.",
                sources=[]
            )
        
        # Build context
        context_text = "\n\n---\n\n".join(
            [match.metadata['text'] for match in results.matches]
        )

        prompt = """
        Please provide a clear explanation using this context:
        {context_text}

        Question to address:
        {request.question}

        Format your response following these guidelines:

        MATHEMATICAL NOTATION:
        1. Inline Mathematics
        - Use single $...$ for inline expressions
        - Example: $x^2 + y^2 = z^2$
        - Keep simple expressions inline

        2. Display Mathematics
        - Use $$...$$ for standalone equations
        - Center important equations on their own line
        - Example:
        $$
        E = mc^2
        $$

        3. Multi-line Equations
        - Use align* environment within $$
        - Example:
        $$
        \\begin opencurly brace align* closecurly brace
        y &= mx + b \\\\
        &= 2x + 3
        \\end opencurly brace align* closecurly brace
        $$

        4. Matrices and Arrays
        - Wrap in $$
        - Use appropriate environment (matrix, pmatrix, bmatrix)
        - Example:
        $$
        \\begin opencurly brace pmatrix closecurly brace
        a & b \\\\
        c & d
        \\end opencurly brace pmatrix closecurly brace
        $$

        WRITING STYLE:
        1. Structure
        - Start with a brief overview
        - Break complex ideas into steps
        - Use clear transitions between concepts
        - No need to use astricks for to bold text. Bold text is not needed.
        - Avoid using bullet point style explanation. Just make sure everything is organized. 

        2. Clarity
        - Define variables on first use
        - Explain key steps clearly
        - Use examples for complex concepts

        3. Formatting
        - Break long explanations into paragraphs
        - Do not use any astricks for fomatting
        - Avoid bullet points or things similar to bullet points

        4. Mathematical Writing
        - Introduce equations before using them
        - Explain the meaning of symbols
        - Connect formulas to concepts

        Remember:
        - Use LaTeX only when it adds value
        - Keep explanations conversational but precise
        - Prioritize understanding over formality
        - Balance thoroughness with conciseness

        Your response should feel like a knowledgeable friend explaining the concept clearly and naturally.
        """
        
 
        
        response_text = get_llm_response(prompt)
        
        # Get sources
        sources = list(set(
            match.metadata.get('source') for match in results.matches 
            if match.metadata.get('source')
        ))
        
        return QueryResponse(response=response_text, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Document QA API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)