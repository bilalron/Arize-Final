from bs4 import BeautifulSoup
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from typing import List, Dict
from urllib.parse import urljoin

class PDFService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )

    async def get_pdf_links(self, url: str) -> List[str]:
        read = requests.get(url)
        soup = BeautifulSoup(read.content, "html.parser")
        
        pdf_links = []
        for link in soup.find_all('a', href=True):
            if link['href'].endswith('.pdf'):
                pdf_url = link['href']
                if not pdf_url.startswith('http'):
                    pdf_url = urljoin(url, pdf_url)
                if pdf_url not in pdf_links and "hw" not in pdf_url:
                    pdf_links.append(pdf_url)
        return pdf_links

    async def process_pdfs(self, pdf_links: List[str]) -> List[Dict]:
        pages = []
        for link in pdf_links:
            try:
                loader = PyPDFLoader(link)
                pages.extend(await loader.aload())
            except Exception as e:
                print(f"Error loading PDF {link}: {str(e)}")
                continue
        
        return self.text_splitter.split_documents(pages)