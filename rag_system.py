"""
Simple RAG System for MacroEnergy.jl and GenX.jl Documentation

Requirements:
pip install requests beautifulsoup4 sentence-transformers faiss-cpu openai python-dotenv

Set your OpenAI API key as environment variable: OPENAI_API_KEY
"""

# Install the Portkey AI Gateway SDK with pip
#   pip -i portkey-ai
# 
# For more information on the SDK see https://portkey.ai/docs/api-reference/sdk/python
# 

import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
import pickle
# from openai import OpenAI
from portkey_ai import Portkey
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Before executing this code, define the API Key within an enironment variable in your OS
# Linux BASH example: export AI_SANDBOX_KEY=<key provided to you>

# Import API key from OS environment variables
AI_SANDBOX_KEY = os.getenv("AI_SANDBOX_KEY")

class DocumentationRAG:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        """Initialize RAG system with embedding model."""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.client = Portkey(api_key=AI_SANDBOX_KEY)
        
    def scrape_documentation(self, base_url: str, max_pages: int = 50) -> List[dict]:
        """Scrape documentation pages from a given base URL."""
        docs = []
        visited = set()
        to_visit = [base_url]
        
        print(f"Scraping documentation from {base_url}...")
        
        while to_visit and len(docs) < max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
                
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                visited.add(url)
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract main content (adjust selectors based on doc structure)
                content_divs = soup.find_all(['article', 'main', 'div'], 
                                            class_=['content', 'documentation', 'doc-content'])
                
                if not content_divs:
                    content_divs = [soup.find('body')]
                
                for content_div in content_divs:
                    if content_div:
                        # Remove navigation, scripts, styles
                        for tag in content_div.find_all(['nav', 'script', 'style', 'footer']):
                            tag.decompose()
                        
                        text = content_div.get_text(separator='\n', strip=True)
                        
                        # Split into chunks (by paragraphs or sections)
                        chunks = self._split_into_chunks(text)
                        
                        for chunk in chunks:
                            if len(chunk.strip()) > 100:  # Minimum chunk size
                                docs.append({
                                    'text': chunk,
                                    'source': url,
                                    'package': 'MacroEnergy.jl' if 'macroenergy' in url.lower() else 'GenX.jl'
                                })
                
                # Find links to other documentation pages (same domain)
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.startswith('/'):
                        full_url = base_url.rstrip('/') + href
                    elif href.startswith('http') and base_url.split('/')[2] in href:
                        full_url = href
                    else:
                        continue
                    
                    if full_url not in visited and full_url not in to_visit:
                        to_visit.append(full_url)
                        
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        
        print(f"Scraped {len(docs)} document chunks")
        return docs
    
    def _split_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - 50):  # 50 word overlap
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
        return chunks
    
    def build_index(self, documentation_urls: List[str]):
        """Build vector index from documentation URLs."""
        # Scrape all documentation
        for url in documentation_urls:
            self.documents.extend(self.scrape_documentation(url))
        
        if not self.documents:
            raise ValueError("No documents were scraped. Check URLs and connectivity.")
        
        print(f"Building embeddings for {len(self.documents)} chunks...")
        
        # Create embeddings
        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print("Index built successfully!")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[dict, float]]:
        """Retrieve most relevant documents for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return documents with scores
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.documents[idx], float(dist)))
        
        return results
    
    def answer_query(self, query: str, top_k: int = 5) -> str:
        """Answer a query using RAG."""
        # Retrieve relevant documents
        results = self.retrieve(query, top_k)
        
        # Build context
        context = "\n\n".join([
            f"[Source: {doc['package']} - {doc['source']}]\n{doc['text']}" 
            for doc, _ in results
        ])
        
        # Generate answer using GPT
        prompt = f"""You are a helpful assistant answering questions about MacroEnergy.jl and GenX.jl Julia packages.

Context from documentation:
{context}

Question: {query}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain enough information, say so."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant expert in Julia energy modeling packages."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def save(self, filepath: str):
        """Save the RAG system to disk."""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index separately
        faiss.write_index(self.index, filepath + '.faiss')
        print(f"Saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the RAG system from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.index = faiss.read_index(filepath + '.faiss')
        print(f"Loaded from {filepath}")


def main():
    """Example usage."""
    # Initialize RAG system
    rag = DocumentationRAG()
    
    # Documentation URLs
    docs_urls = [
        'https://macroenergy.github.io/MacroEnergy.jl/stable/',
        'https://genxproject.github.io/GenX.jl/stable/'
    ]
    
    # Build index (do this once, then save)
    rag.build_index(docs_urls)
    
    # Save for future use
    rag.save('energy_packages_rag.pkl')
    
    # Or load existing index
    # rag.load('energy_packages_rag.pkl')
    
    # Example queries
    queries = [
        "How do I set up a basic optimization problem in GenX?",
        "What are the main features of MacroEnergy.jl?",
        "How do I define generators in GenX.jl?"
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        answer = rag.answer_query(query)
        print(f"\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()