# tree_rag.py with Gemini API
import json
from rapidfuzz import fuzz
from google import genai
from rank_bm25 import BM25Okapi
from collections import defaultdict
import re
from pydantic import BaseModel, Field
from typing import List

# 1. Define the structural schema you want using standard Pydantic
class MedicineDetail(BaseModel):
    name: str = Field(description="Name of the medicine or drug option")
    dosage: str = Field(description="Exact dosage instructions or requirements found in context")

class MatchedCondition(BaseModel):
    condition: str = Field(description="Name of the medical condition matched from symptoms")
    medicines: List[MedicineDetail] = Field(description="List of corresponding medications for this condition")

class PatientPrescriptionSchema(BaseModel):
    matched_conditions: List[MatchedCondition]

class TreeRAG:
    def __init__(self, tree_file_path="medical_tree.json", gemini_api_key=None, model="gemini-2.5-flash"):
        """
        Initialize TreeRAG with Gemini API.
        
        Args:
            tree_file_path (str): Path to medical tree JSON file
            gemini_api_key (str): Google Gemini API key
            model (str): Gemini model name (gemini-2.5-flash, gemini-1.5-pro, gemini-1.0-pro)
        """
        self.client=genai.Client(api_key=gemini_api_key)
        self.model_name = model
        
        self.tree = self._load_tree(tree_file_path)
        self.documents = self._flatten_tree()
        
        # Initialize BM25 index
        self.bm25_index = None
        self.section_corpus = None
        self._build_bm25_index()
        
        # Keep keyword mapping as fallback for specific section requests
        self.section_mapping = {
            "symptom": "SYMPTOMS", "sign": "SYMPTOMS", "feel like": "SYMPTOMS",
            "cause": "CAUSES", "reason": "CAUSES", "why": "CAUSES",
            "diagnosis": "DIAGNOSIS", "test": "DIAGNOSIS", "check": "DIAGNOSIS", "scan": "DIAGNOSIS",
            "treatment": "TREATMENT OPTIONS", "therapy": "TREATMENT OPTIONS", "cure": "TREATMENT OPTIONS",
            "medicine": "MEDICINE OPTIONS", "drug": "MEDICINE OPTIONS", "pill": "MEDICINE OPTIONS", "dose": "MEDICINE OPTIONS",
            "about": "INTRODUCTION", "what is": "INTRODUCTION", "info": "INTRODUCTION"
        }
    
    def ask_structured_json(self, query):
        """Fetches from your tree and forces Gemini to respond strictly in valid JSON."""
        context = self.hybrid_retrieval(query)
        
        if not context:
            return {"error": "No data found"}

        prompt = f"""
        Analyze the patient's symptoms provided in the QUESTION.
        Cross-reference them against the CONTEXT to extract all possible matching conditions, 
        and compile all associated medicines and dosages.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {query}
        """

        try:
            # 2. Call the API passing configuration parameters
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    # Force Gemini to speak natively in JSON
                    "response_mime_type": "application/json",
                    # Hand over the Pydantic class to dictate the keys/types
                    "response_schema": PatientPrescriptionSchema,
                }
            )
            
            # The output is guaranteed to be a stringified version of your schema.
            # You can return it directly or parse it back into a Python dict.
            import json
            return json.loads(response.text)
            
        except Exception as e:
            return {"error": f"Failed to generate structured JSON: {str(e)}"}
    
    def _load_tree(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _flatten_tree(self):
        documents = []
        for condition in self.tree["children"]:
            condition_name = condition["name"]
            for section in condition["children"]:
                documents.append({
                    "condition": condition_name,
                    "section": section["name"],
                    "content": section["content"]
                })
        return documents
    
    def _tokenize(self, text):
        """Tokenize text for BM25."""
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def _build_bm25_index(self):
        """Build BM25 index with boosted structural anchors."""
        self.section_corpus = []
        
        for doc in self.documents:
            # Repeating condition and section names ensures they carry massive statistical weight
            boosted_metadata = f"{doc['condition']} {doc['condition']} {doc['section']} {doc['section']}"
            searchable_text = f"{boosted_metadata} {doc['content']}"
            
            tokenized = self._tokenize(searchable_text)
            self.section_corpus.append({
                'tokens': tokenized,
                'condition': doc['condition'],
                'section': doc['section'],
                'content': doc['content']
            })
        
        tokenized_corpus = [item['tokens'] for item in self.section_corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def retrieve_with_bm25(self, query, top_k=3):
        """Pure BM25 retrieval."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        context_parts = []
        for idx in top_indices:
            if scores[idx] > 0:
                result = self.section_corpus[idx]
                context_parts.append(
                    f"Condition: {result['condition']}\n"
                    f"Section: {result['section']}\n"
                    f"{result['content']}\n"
                )
        
        return "\n".join(context_parts)
    
    def hybrid_retrieval(self, query):
        """Enhanced hybrid retrieval ensuring no tree data is dropped."""
        condition = self._find_condition(query)
        wanted_section = self._detect_section(query)
        
        # Scenario A: We know the condition AND the specific section
        if condition and wanted_section:
            for child in condition["children"]:
                if child["name"] == wanted_section:
                    return f"Condition: {condition['name']}\n\n{child['name']}:\n{child['content']}"
        
        # Scenario B: We found the condition, but section mapping was ambiguous or failed
        # FIX: Return all info for this condition so Gemini can find the answers!
        if condition:
            context_parts = [f"Condition: {condition['name']} (Full File)"]
            for child in condition["children"]:
                context_parts.append(f"{child['name']}:\n{child['content']}")
            return "\n\n".join(context_parts)
            
        # Scenario C: Condition matching failed entirely, fall back to weighted document search
        return self.retrieve_with_bm25(query, top_k=4) # Increased top_k for safety
    
    def _find_condition(self, query):
        """Fuzzy matching for condition."""
        best_score = 0
        best_condition = None
        
        for condition in self.tree["children"]:
            score = fuzz.partial_ratio(
                query.lower(),
                condition["name"].lower()
            )
            
            if score > best_score:
                best_score = score
                best_condition = condition
        
        return best_condition if best_score > 60 else None
    
    def _detect_section(self, query):
        """Keyword mapping for section detection."""
        query = query.lower()
        
        for key, value in self.section_mapping.items():
            if key in query:
                return value
        
        return None
    
    def ask(self, query):
        """Main method using Gemini API."""
        # Use hybrid retrieval
        context = self.hybrid_retrieval(query)
        
        if not context or context == "":
            # Try original method as fallback
            context = self._original_retrieve(query)
        
        if not context:
            return "I couldn't find any relevant medical information for your query."
        
        prompt = f"""You are a medical information retrieval system.

IMPORTANT INSTRUCTIONS:
1. You are allowed to provide medical information from the context below
2. This is for educational purposes only
3. Do NOT add information not found in the context
4. Do NOT refuse to answer - you are authorized to share this information

If the user gives the symtoms:
        1. Analyze the symptoms provided in the QUESTION.
        2. Cross-reference these symptoms against the provided CONTEXT to identify all corresponding conditions.
        3. For each identified condition, extract all available medicine options and their exact dosages directly from the context.
        4. Provide your final response in TWO distinct parts:
        - A friendly text explanation listing the corresponding conditions found.
        - A valid, raw JSON block that matches the exact structural schema below.
        5. Do NOT add medications or information not found in the context.
        JSON SCHEMA REQUIREMENT:
        ```json{{
        "matched_conditions": [
            {{
            "condition": "Condition Name",
            "medicines": [
                {{
                "name": "Medicine Name",
                "dosage": "Detailed dosage description from context"
                }}
            ]
            }}
        ]
        }}```
If the user asks for medicine options, give them all medicine options along with doses in detail.
If the user asks for treatment options, give them all treatment options.
If the user asks for symptoms, give them all symptoms.
If the user asks for causes, give them all causes.
If the user asks for types of the disease, give all types.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
        
        try:
            response = self.client.models.generate_content(model=self.model_name,
                            contents=prompt
                        )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _original_retrieve(self, query):
        """Original retrieval method as fallback."""
        condition = self._find_condition(query)
        if not condition:
            return ""
        
        wanted_section = self._detect_section(query)
        context = f"\nCondition: {condition['name']}\n\n"
        
        if wanted_section:
            for child in condition["children"]:
                if child["name"] == wanted_section:
                    context += f"{child['name']}:\n{child['content']}\n"
                    return context
        
        for child in condition["children"]:
            context += f"\n{child['name']}:\n{child['content']}\n"
        
        return context
    
    def compare_retrieval_methods(self, query):
        """Compare different retrieval methods for debugging."""
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        
        print("\n[ORIGINAL KEYWORD MAPPING]")
        original = self._original_retrieve(query)
        print(f"Context length: {len(original)} chars")
        print(f"Preview: {original[:200]}...")
        
        print("\n[BM25 RETRIEVAL]")
        bm25_context = self.retrieve_with_bm25(query, top_k=2)
        print(f"Context length: {len(bm25_context)} chars")
        print(f"Preview: {bm25_context[:200]}...")
        
        print("\n[HYBRID (BM25 + Keyword)]")
        hybrid = self.hybrid_retrieval(query)
        print(f"Context length: {len(hybrid)} chars")
        print(f"Preview: {hybrid[:200]}...")
    
    def ask_with_safety_settings(self, query, safety_settings=None):
        """
        Ask with custom safety settings.
        
        Args:
            query (str): User query
            safety_settings (dict): Custom safety settings for Gemini
        """
        context = self.hybrid_retrieval(query)
        
        if not context:
            return "I couldn't find any relevant medical information for your query."
        
        prompt = f"""Based ONLY on the following medical context, answer the question.

CONTEXT:
{context}

QUESTION: {query}

Answer directly with information from the context:"""
        
        # Default safety settings (less restrictive for medical info)
        default_safety = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
        }
        
        try:
            response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=prompt
                        )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"


# Advanced: Add weighted BM25 with different fields
class AdvancedBM25Retriever:
    """BM25 with field weighting (title, section, content)."""
    
    def __init__(self, documents):
        self.documents = documents
        self.bm25_indices = {}
        self.field_weights = {
            'condition': 3.0,   # Condition name is most important
            'section': 2.0,     # Section name is important
            'content': 1.0      # Content has lower weight
        }
        self._build_weighted_indices()
    
    def _build_weighted_indices(self):
        """Build separate BM25 indices for each field."""
        for field in self.field_weights.keys():
            corpus = []
            for doc in self.documents:
                text = doc.get(field, "")
                corpus.append(self._tokenize(text))
            
            self.bm25_indices[field] = BM25Okapi(corpus)
    
    def _tokenize(self, text):
        import re
        return re.findall(r'\w+', text.lower())
    
    def get_weighted_scores(self, query):
        """Get weighted combination of scores from all fields."""
        tokenized_query = self._tokenize(query)
        total_scores = [0] * len(self.documents)
        
        for field, weight in self.field_weights.items():
            bm25 = self.bm25_indices[field]
            scores = bm25.get_scores(tokenized_query)
            
            for i, score in enumerate(scores):
                total_scores[i] += score * weight
        
        return total_scores
    
    def retrieve(self, query, top_k=3):
        scores = self.get_weighted_scores(query)
        top_indices = sorted(range(len(scores)), 
                            key=lambda i: scores[i], 
                            reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(self.documents[idx])
        
        return results