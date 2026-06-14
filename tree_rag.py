# tree_rag.py with Gemini API - Final version with quota error handling
import json
from rapidfuzz import fuzz
from google import genai
from rank_bm25 import BM25Okapi
import re
from pydantic import BaseModel, Field
from typing import List

# Pydantic schemas
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
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_name = model
        self.tree = self._load_tree(tree_file_path)
        self.documents = self._flatten_tree()
        self.bm25_index = None
        self.section_corpus = None
        self._build_bm25_index()
        self.section_mapping = {
            "symptom": "SYMPTOMS", "sign": "SYMPTOMS", "feel like": "SYMPTOMS",
            "cause": "CAUSES", "reason": "CAUSES", "why": "CAUSES",
            "diagnosis": "DIAGNOSIS", "test": "DIAGNOSIS", "check": "DIAGNOSIS", "scan": "DIAGNOSIS",
            "treatment": "TREATMENT OPTIONS", "therapy": "TREATMENT OPTIONS", "cure": "TREATMENT OPTIONS",
            "medicine": "MEDICINE OPTIONS", "drug": "MEDICINE OPTIONS", "pill": "MEDICINE OPTIONS", "dose": "MEDICINE OPTIONS",
            "about": "INTRODUCTION", "what is": "INTRODUCTION", "info": "INTRODUCTION"
        }
    
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
        return re.findall(r'\w+', text.lower())
    
    def _build_bm25_index(self):
        self.section_corpus = []
        for doc in self.documents:
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
        tokenized_query = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        context_parts = []
        for idx in top_indices:
            if scores[idx] > 0:
                result = self.section_corpus[idx]
                context_parts.append(f"Condition: {result['condition']}\nSection: {result['section']}\n{result['content']}\n")
        return "\n".join(context_parts)
    
    def hybrid_retrieval(self, query):
        condition = self._find_condition(query)
        wanted_section = self._detect_section(query)
        if condition and wanted_section:
            for child in condition["children"]:
                if child["name"] == wanted_section:
                    return f"Condition: {condition['name']}\n\n{child['name']}:\n{child['content']}"
        if condition:
            context_parts = [f"Condition: {condition['name']} (Full File)"]
            for child in condition["children"]:
                context_parts.append(f"{child['name']}:\n{child['content']}")
            return "\n\n".join(context_parts)
        return self.retrieve_with_bm25(query, top_k=4)
    
    def _find_condition(self, query):
        best_score = 0
        best_condition = None
        for condition in self.tree["children"]:
            score = fuzz.partial_ratio(query.lower(), condition["name"].lower())
            if score > best_score:
                best_score = score
                best_condition = condition
        return best_condition if best_score > 60 else None
    
    def _detect_section(self, query):
        query = query.lower()
        for key, value in self.section_mapping.items():
            if key in query:
                return value
        return None
    
    def _original_retrieve(self, query):
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
    
    def _is_symptom_query(self, query):
        query_lower = query.lower()
        symptom_indicators = [
            r'\b(i|im|i am|im feeling|i feel|i have|ive got|suffering from)\b',
            r'\b(fever|cough|pain|ache|nausea|vomiting|dizzy|fatigue|headache)\b',
            r'\b(shortness of breath|chest pain|sore throat|runny nose)\b'
        ]
        for pattern in symptom_indicators:
            if re.search(pattern, query_lower):
                return True
        symptom_keywords = ["symptom", "feel", "ache", "pain", "hurt"]
        info_keywords = ["what is", "define", "tell me about", "medicine", "treatment", "cause"]
        has_symptom = any(k in query_lower for k in symptom_keywords)
        has_info = any(k in query_lower for k in info_keywords)
        return has_symptom and not has_info
    
    def ask(self, query):
        """Returns JSON for symptom queries, natural language otherwise."""
        context = self.hybrid_retrieval(query)
        if not context or context == "":
            context = self._original_retrieve(query)
        if not context:
            return {"error": "No relevant medical information found."}
        
        if self._is_symptom_query(query):
            prompt = f"""Return ONLY a valid JSON object. No extra text.
Extract all conditions matching the symptoms. For each, list medicines with exact dosages.
CONTEXT:
{context}
QUESTION (symptoms):
{query}
JSON format:
{{"matched_conditions":[{{"condition":"name","medicines":[{{"name":"med","dosage":"dose"}}]}}]}}"""
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json", "response_schema": PatientPrescriptionSchema}
                )
                return json.loads(response.text)
            except Exception as e:
                error_str = str(e)
                # Detect quota / rate limit error
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    return {"error": "⚠️ API quota exceeded. Please wait a few minutes or use a different API key. Free tier limit is 20 requests per day for gemini-2.5-flash."}
                return {"error": f"JSON generation failed: {error_str}"}
        else:
            prompt = f"Based ONLY on context, answer concisely.\nCONTEXT:\n{context}\nQUESTION:\n{query}\nANSWER:"
            try:
                response = self.client.models.generate_content(model=self.model_name, contents=prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    return "⚠️ API quota exceeded. Please wait or switch API key. Free tier: 20 requests/day."
                return f"Error: {error_str}"
    
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
        context = self.hybrid_retrieval(query)
        if not context:
            return "I couldn't find any relevant medical information for your query."
        prompt = f"""Based ONLY on the following medical context, answer the question.
CONTEXT:
{context}
QUESTION: {query}
Answer directly with information from the context:"""
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

# Advanced: Weighted BM25 (optional)
class AdvancedBM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.bm25_indices = {}
        self.field_weights = {'condition': 3.0, 'section': 2.0, 'content': 1.0}
        self._build_weighted_indices()
    
    def _build_weighted_indices(self):
        for field in self.field_weights.keys():
            corpus = []
            for doc in self.documents:
                text = doc.get(field, "")
                corpus.append(self._tokenize(text))
            self.bm25_indices[field] = BM25Okapi(corpus)
    
    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())
    
    def get_weighted_scores(self, query):
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
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]
