from rag_system_ollama import DocumentationRAG

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