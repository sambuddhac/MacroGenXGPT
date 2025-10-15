from rag_system_ollama import DocumentationRAG

# Load saved index
rag = DocumentationRAG(ollama_model='llama3.2')
rag.load('energy_packages_rag.pkl')

# Ask questions!
answer = rag.answer_query("Does parameter scaling affect objective function in GenX.jl?", stream=True)
print(answer)