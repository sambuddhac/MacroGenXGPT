from rag_system_ollama import DocumentationRAG
#from rag_system import DocumentationRAG

rag = DocumentationRAG(ollama_model='llama3.2')
#rag = DocumentationRAG()
rag.load('energy_packages_rag.pkl')

# Retrieve the most relevant documents and answer
answer = rag.answer_query(
    "How do I configure GenX solver settings?", 
    top_k=3,  # Reduce to 3 most relevant
    verbose=True,  # Show what's being retrieved
    stream=True  # Stream the response
)

# Increase max_pages to get more content
rag = DocumentationRAG()
rag.build_index([
    'https://macroenergy.github.io/MacroEnergy.jl/stable/',
    'https://genxproject.github.io/GenX.jl/stable/'
])
# Should see "Scraped 100+ document chunks" not 5!