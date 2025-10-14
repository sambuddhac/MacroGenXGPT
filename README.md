# MacroGenXGPT
MacroGenXGPT is an Python-based LLM using Retrieval Augmented Generation (RAG) for interacting with users of these models and guiding the users as to how to make use of and understand the various features of the model. 

# How to Use:

Install dependencies:

## Create a new conda environment (Recommended)
This is the cleanest approach and avoids conflicts with your existing environment:

```bash
# Create new environment with Python 3.10 (better compatibility)
conda create -n rag_env python=3.10

# Activate it
conda activate rag_env

# Install packages
pip install requests beautifulsoup4 sentence-transformers faiss-cpu openai python-dotenv
```

Then in VSCode:

Press Cmd+Shift+P (Mac) or Ctrl+Shift+P (Windows)
Type "Python: Select Interpreter"
Choose the rag_env environment

Then verify it worked:
```bash
python -c "import sentence_transformers; print('Success!')"
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