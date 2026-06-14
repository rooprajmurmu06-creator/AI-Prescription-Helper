# 📋 Medical RAG – Intelligent Medical Information System

## 🎯 Project Description

Medical RAG is an intelligent medical information system that leverages **Vectorless Retrieval-Augmented Generation (RAG)** to analyze patient symptoms and recommend appropriate medications with dosage information. The system combines traditional information retrieval techniques such as **BM25** and **fuzzy matching** with **Google Gemini** to provide accurate, context-aware medical recommendations.

It also includes an **interactive prescription builder**, allowing doctors to add, remove, and manage medications before generating the final prescription.

---

# 🎯 Problem Statement

Healthcare professionals often face challenges in:

* Quickly accessing structured medical knowledge from large unstructured documents.
* Matching patient-described symptoms to relevant medical conditions.
* Extracting accurate medication names and dosage recommendations for multiple conditions.
* Building and managing prescriptions interactively during patient consultations.

---

# ✨ Features

## Core Features
* 🌳 Tree-Based Knowledge Base

  * Medical information organized as hierarchical trees (Conditions → Symptoms → Treatments → Medicines)
  
  * Preserves structural relationships between medical entities
  
  * Enables section-specific retrieval (e.g., only SYMPTOMS or only MEDICINE OPTIONS)

* ⚡ Vectorless RAG Architecture

  * No embeddings or vector databases required
  
  * Uses lightweight BM25 + fuzzy matching instead of heavy vector search
  
  * Faster initialization and lower memory footprint
  
  * Ideal for resource-constrained environments

* 🔍 **Hybrid Retrieval System**

  * Combines BM25, fuzzy matching, and keyword mapping for accurate context retrieval.

* 🤖 **LLM Integration**

  * Uses Google Gemini for intelligent symptom-to-condition matching.

* 📊 **Structured JSON Output**

  * Returns clean JSON responses containing conditions, medicines, and dosages.

* 💊 **Interactive Prescription Builder**

  * Allows doctors to add or remove medicines with real-time updates.

* 📋 **Final Prescription Generation**

  * Produces both machine-readable JSON and human-readable prescription formats.

---

## Technical Features

* 🚀 Streamlit-based responsive web interface
* 🎨 Clean table visualization with merged condition rows
* ⚡ Real-time prescription updates using session state
* 🔄 Conversation history support
* 📁 Flexible JSON-based medical knowledge base

---
## Advantages Over Traditional Clinical Reference Systems

  While Electronic Health Records (EHR) and Clinical Decision Support Systems 
  (CDSS) like UpToDate and Lexicomp provide medication recommendations, they 
  have key limitations:
  
  1. **Structured Input Required** – Doctors must know and type the exact 
     condition name, not patient-described symptoms.
  
  2. **Unstructured Output** – Information is returned as dense text paragraphs, 
     not machine-readable JSON for automated processing.
  
  3. **No Prescription Workflow** – These systems suggest medicines but don't 
     help build, edit, or finalize a prescription interactively.
  
  4. **Cost & Lock-in** – Most solutions are proprietary, expensive, and 
     require ongoing subscriptions.
  
  5. **No Symptom-to-Condition Matching** – They can't take "I have a fever 
     with chest pain" and automatically map it to potential conditions.
  
  Medical RAG bridges this gap by accepting natural language symptom descriptions, 
  automatically matching them to conditions, extracting structured medicine data, 
  and providing an interactive prescription builder – all in an open-source, 
  vectorless RAG architecture.
