from rag_system_ollama import DocumentationRAG

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