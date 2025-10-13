# MacroGenXGPT
MacroGenXGPT is an Python-based LLM using Retrieval Augmented Generation (RAG) for interacting with users of these models and guiding the users as to how to make use of and understand the various features of the model. 

# How to Use:

Install dependencies:

```bash
pip install requests beautifulsoup4 sentence-transformers faiss-cpu openai python-dotenv
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-key-here'
```

Run the script to build the index (first time only):

```bash
python rag_system.py
```

Use it in your code:

```python
from rag_system import DocumentationRAG

# Load existing index
rag = DocumentationRAG()
rag.load('energy_packages_rag.pkl')

# Ask questions
answer = rag.answer_query("How do I configure temporal resolution in GenX?")
print(answer)
```
## Customization Options:

- **Change LLM:** Replace OpenAI with local models (Ollama, LlamaCpp)
- **Adjust chunk size:** Modify `chunk_size` in `_split_into_chunks()`
- **Change embedding model:** Use different sentence-transformers models
- **Adjust retrieval:** Change `top_k` for more/fewer context chunks