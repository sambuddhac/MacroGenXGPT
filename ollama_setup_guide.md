# Ollama Setup Guide for RAG System

## What Changed from OpenAI Version?

### Key Differences:

1. **No API Key Needed** ❌ `OPENAI_API_KEY`
2. **Runs Locally** 💻 All processing on your machine
3. **Free Forever** 💰 No usage costs
4. **Complete Privacy** 🔒 Data never leaves your computer
5. **Requires Installation** 📦 One-time setup

### Code Changes Summary:

```python
# OLD (OpenAI):
from openai import OpenAI
self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
response = self.client.chat.completions.create(...)

# NEW (Ollama):
import requests, json
self.ollama_url = 'http://localhost:11434'
response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
```

---

## Step 1: Install Ollama

### macOS
```bash
# Download and install from website
# Visit: https://ollama.ai/download

# Or use Homebrew
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Windows
Download installer from: https://ollama.ai/download

---

## Step 2: Start Ollama

### macOS/Linux
```bash
# Ollama usually starts automatically after installation
# If not, start it manually:
ollama serve
```

### Windows
Ollama starts automatically as a service after installation.

### Verify It's Running
```bash
# Check if Ollama is responding
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

---

## Step 3: Download Models

### Recommended Models for RAG:

```bash
# BEST OVERALL: Llama 3.2 (3B parameters, 2GB)
ollama pull llama3.2

# FASTER: Phi-3 (3.8B parameters, 2.3GB)
ollama pull phi3

# LARGER/BETTER: Llama 3.1 (8B parameters, 4.7GB)
ollama pull llama3.1:8b

# CODING-FOCUSED: Qwen 2.5 Coder (7B parameters, 4.7GB)
ollama pull qwen2.5-coder:7b

# FAST & SMALL: Gemma 2 (2B parameters, 1.6GB)
ollama pull gemma2:2b
```

### Check Downloaded Models
```bash
ollama list
```

Output:
```
NAME              ID              SIZE    MODIFIED
llama3.2:latest   a80c4f17acd5    2.0 GB  2 hours ago
phi3:latest       d184c916657e    2.3 GB  1 day ago
```

---

## Step 4: Test Ollama

### Quick Test
```bash
ollama run llama3.2
```

You'll get an interactive prompt:
```
>>> What is Julia?
Julia is a high-level, high-performance programming language...

>>> /bye
```

### API Test
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "What is GenX.jl?",
  "stream": false
}'
```

---

## Step 5: Install Python Dependencies

```bash
# Create new environment (recommended)
conda create -n rag_ollama python=3.10
conda activate rag_ollama

# Install packages (NO openai needed!)
pip install requests beautifulsoup4 sentence-transformers faiss-cpu
```

---

## Step 6: Run the RAG System

### First Time (Build Index)

```python
from rag_system import DocumentationRAG

# Initialize with Ollama
rag = DocumentationRAG(
    embedding_model='all-MiniLM-L6-v2',
    ollama_model='llama3.2'  # or 'phi3', 'mistral', etc.
)

# Build index (takes ~1-2 minutes)
docs_urls = [
    'https://macroenergy.github.io/MacroEnergy.jl/stable/',
    'https://genxproject.github.io/GenX.jl/stable/'
]
rag.build_index(docs_urls)

# Save for future use
rag.save('energy_packages_rag.pkl')
```

### Subsequent Uses (Load Saved Index)

```python
from rag_system import DocumentationRAG

# Load saved index
rag = DocumentationRAG(ollama_model='llama3.2')
rag.load('energy_packages_rag.pkl')

# Ask questions!
answer = rag.answer_query("How do I configure GenX solver settings?")
print(answer)
```

---

## Model Comparison for RAG

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **llama3.2** | 2GB | Fast | ⭐⭐⭐⭐ | General RAG (RECOMMENDED) |
| **phi3** | 2.3GB | Very Fast | ⭐⭐⭐⭐ | Quick responses |
| **llama3.1:8b** | 4.7GB | Medium | ⭐⭐⭐⭐⭐ | Best quality |
| **mistral** | 4.1GB | Medium | ⭐⭐⭐⭐ | Balanced |
| **gemma2:2b** | 1.6GB | Very Fast | ⭐⭐⭐ | Resource-constrained |
| **qwen2.5-coder** | 4.7GB | Medium | ⭐⭐⭐⭐⭐ | Code/technical docs |

### Hardware Requirements:

- **Minimum**: 8GB RAM, can run 2-3B models (llama3.2, phi3)
- **Recommended**: 16GB RAM, can run 7-8B models comfortably
- **Optimal**: 32GB RAM + GPU, can run larger models

---

## Advanced Usage

### 1. Streaming Responses (Like ChatGPT)

```python
# Streaming is enabled by default
answer = rag.answer_query("What is GenX?", stream=True)
# Prints response token-by-token as it's generated

# Disable streaming for full response at once
answer = rag.answer_query("What is GenX?", stream=False)
```

### 2. Switch Models on the Fly

```python
# Load with one model
rag = DocumentationRAG(ollama_model='llama3.2')
rag.load('energy_packages_rag.pkl')

# Change model for next query
rag.ollama_model = 'phi3'
answer = rag.answer_query("Your question here")
```

### 3. Adjust Generation Parameters

Modify in the `answer_query` method:

```python
payload = {
    "model": self.ollama_model,
    "prompt": prompt,
    "stream": stream,
    "options": {
        "temperature": 0.3,    # Lower = more focused (0.0-1.0)
        "top_p": 0.9,         # Nucleus sampling (0.0-1.0)
        "top_k": 40,          # Consider top K tokens
        "repeat_penalty": 1.1, # Penalize repetition
        "num_ctx": 4096,      # Context window size
    }
}
```

### 4. Use Remote Ollama Server

If running Ollama on another machine:

```python
rag = DocumentationRAG(
    ollama_model='llama3.2',
    ollama_url='http://192.168.1.100:11434'  # Remote server IP
)
```

---

## Troubleshooting

### Problem: "Cannot connect to Ollama"

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve

# On another terminal, test
ollama list
```

### Problem: "Model not found"

**Solution:**
```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.2

# Verify
ollama list
```

### Problem: Response is slow

**Solutions:**
1. Use smaller model: `phi3` or `gemma2:2b`
2. Reduce context window in options: `"num_ctx": 2048`
3. Close other applications
4. Consider GPU acceleration (see below)

### Problem: Out of memory

**Solutions:**
1. Use smaller model (2-3B parameters)
2. Close other applications
3. Reduce `top_k` documents retrieved: `rag.answer_query(query, top_k=3)`

---

## GPU Acceleration (Optional)

Ollama automatically uses GPU if available:

### Check GPU Usage (macOS)
```bash
# Activity Monitor → Window → GPU History
```

### Check GPU Usage (Linux with NVIDIA)
```bash
nvidia-smi
```

### Force CPU-only (if needed)
```bash
OLLAMA_NUM_GPU=0 ollama serve
```

---

## Performance Comparison

### OpenAI API vs Ollama (on M2 Mac, 16GB RAM)

| Aspect | OpenAI GPT-4o-mini | Ollama llama3.2 |
|--------|-------------------|-----------------|
| **Initial Setup** | 1 min (API key) | 5 min (download) |
| **Response Time** | 2-3 seconds | 5-8 seconds |
| **Cost per 1M tokens** | $0.15-$0.60 | $0 (electricity) |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Privacy** | Data sent to OpenAI | 100% local |
| **Offline** | ❌ Needs internet | ✅ Works offline |
| **Token Limit** | 128K context | 4-32K (model dependent) |

---

## Code Comparison: OpenAI vs Ollama

### OpenAI Version
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3
)

answer = response.choices[0].message.content
```

### Ollama Version
```python
import requests, json

payload = {
    "model": "llama3.2",
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.3}
}

response = requests.post(
    "http://localhost:11434/api/generate",
    json=payload
)

answer = response.json()['response']
```

---

## Tips for Best Results

1. **Use appropriate model size**: 
   - Quick tests: `gemma2:2b` or `phi3`
   - Production: `llama3.2` or `llama3.1:8b`
   
2. **Optimize retrieval**:
   - Start with `top_k=3` for faster responses
   - Increase to `top_k=5-7` if answers lack context

3. **Temperature settings**:
   - Factual Q&A: `0.1-0.3` (more deterministic)
   - Creative tasks: `0.7-0.9` (more varied)

4. **Monitor resources**:
   ```bash
   # Watch memory usage
   watch -n 1 free -h  # Linux
   ```

5. **Batch processing**:
   ```python
   questions = ["Q1", "Q2", "Q3"]
   answers = [rag.answer_query(q) for q in questions]
   ```

---

## Next Steps

Once you have Ollama working:

1. **Experiment with models**: Try different models to find the best balance
2. **Fine-tune prompts**: Adjust system prompts for better answers
3. **Add evaluation**: Compare answers against ground truth
4. **Optimize chunking**: Experiment with chunk sizes and overlap
5. **Add reranking**: Use cross-encoders for better retrieval

---

## Quick Start Script

Save this as `test_ollama.py`:

```python
from rag_system import DocumentationRAG

# Initialize
rag = DocumentationRAG(ollama_model='llama3.2')

# Build index (first time only)
rag.build_index([
    'https://macroenergy.github.io/MacroEnergy.jl/stable/',
    'https://genxproject.github.io/GenX.jl/stable/'
])
rag.save('energy_rag.pkl')

# Ask a question
answer = rag.answer_query("What is MacroEnergy.jl?", stream=True)
```

Run it:
```bash
python test_ollama.py
```