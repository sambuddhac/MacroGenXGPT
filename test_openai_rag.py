"""
Test script for the OpenAI/Portkey RAG system (rag_system.py)
This version uses the Portkey AI Gateway with remote models like GPT-4o-mini
"""

from rag_system import DocumentationRAG

def main():
    # Initialize the RAG system (uses Portkey with AI_SANDBOX_KEY)
    print("🔧 MacroEnergy.jl & GenX.jl Documentation Assistant")
    print("=" * 60)
    
    try:
        rag = DocumentationRAG()
        
        # Load the pre-built index
        print("📚 Loading pre-built RAG index...")
        rag.load('energy_packages_rag.pkl')
        print("✅ Index loaded successfully!\n")
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure AI_SANDBOX_KEY environment variable is set")
        print("2. Verify the API key is valid for Portkey service") 
        print("3. Check if energy_packages_rag.pkl file exists")
        return
    
    # Interactive loop
    print("💡 Ask me anything about MacroEnergy.jl or GenX.jl!")
    print("   Type 'quit', 'exit', or press Ctrl+C to stop\n")
    
    while True:
        try:
            # Get user input
            query = input("🤔 Your question: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Skip empty queries
            if not query:
                print("❓ Please enter a question or type 'quit' to exit.\n")
                continue
            
            print(f"\n🔍 Processing your query...")
            print("=" * 60)
            
            # Query the system
            answer = rag.answer_query(
                query,
                top_k=3  # Use top 3 most relevant documents
            )
            
            print("\n🤖 Answer (from Portkey/GPT):")
            print("-" * 40)
            print(answer)
            print("\n" + "=" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
            
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            print("\nTroubleshooting:")
            print("1. Check your internet connection")
            print("2. Verify API key is still valid")
            print("3. Try a simpler question")
            print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()

# Optional: Test building a fresh index (commented out to save time)
# print("\n" + "=" * 80)
# print("Building fresh index from documentation...")
# try:
#     rag_fresh = DocumentationRAG()
#     rag_fresh.build_index([
#         'https://macroenergy.github.io/MacroEnergy.jl/stable/',
#         'https://genxproject.github.io/GenX.jl/stable/'
#     ])
#     
#     # Test with fresh index
#     answer_fresh = rag_fresh.answer_query(query, top_k=3)
#     print(f"\nAnswer (fresh index): {answer_fresh}")
#     
# except Exception as e:
#     print(f"Error building fresh index: {e}")