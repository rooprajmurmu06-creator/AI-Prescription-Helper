# main.py
import os
from tree_rag import TreeRAG
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
def main():
    # Initialize with Gemini API
    rag = TreeRAG(
        tree_file_path="medical_tree.json",
        gemini_api_key=api_key,  # Replace with your actual key
        model="gemini-2.5-flash"  # or "gemini-1.5-pro" for better quality
    )
    
    # Example queries
    queries = [
        "I have dizziness, nausea, dehydration and blurred vision what are medicines i need",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        print(f"{'='*60}")
        
        answer = rag.ask_structured_json(query)
        print(f"A: {answer}")

def interactive_chat():
    """Interactive chat mode"""
    rag = TreeRAG(
        tree_file_path="medical_tree.json",
        gemini_api_key=api_key,
        model="gemini-2.5-flash"
    )
    
    print("Medical Information Assistant (Gemini)")
    print("Type 'quit' to exit, 'compare' to test retrieval methods")
    print("-" * 50)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if query.lower() == 'compare':
            test_query = input("Enter query to compare methods: ")
            rag.compare_retrieval_methods(test_query)
            continue
        
        if not query:
            continue
        
        answer = rag.ask(query)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    # Use environment variable for API key (recommended)
    import os
    # os.environ["GEMINI_API_KEY"] = "your_key_here"
    
    # Or pass directly
    main()