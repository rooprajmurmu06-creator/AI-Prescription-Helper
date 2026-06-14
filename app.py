# app.py - with PDF prescription export (fixed font error)
import streamlit as st
import json
import pandas as pd
import os
from tree_rag import TreeRAG
from fpdf import FPDF
from datetime import datetime
import re

st.set_page_config(page_title="Medical RAG - Prescription Builder", page_icon="🏥", layout="wide")

st.title("🏥 Medical RAG - Prescription Builder")
st.markdown("**Describe symptoms → Review medicines. Use + Add / ❌ Remove to build prescription. Then download as PDF.**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password")
    model_name = st.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    tree_file = st.text_input("Medical Knowledge JSON file", value="medical_tree.json")
    
    if os.path.exists(tree_file):
        st.success(f"✅ Found {tree_file}")
    else:
        st.error(f"❌ {tree_file} not found.")
        uploaded = st.file_uploader("Upload medical_tree.json", type=["json"])
        if uploaded:
            with open(tree_file, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success("Saved! Refresh.")
    
    st.divider()
    st.markdown("### 📌 Examples")
    st.info("• `I have fever, cough, and difficulty breathing`\n• `My head hurts and I feel nauseous`")
    
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.prescription_lists = {}
        st.rerun()

# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prescription_lists" not in st.session_state:
    st.session_state.prescription_lists = {}

def clean_text(text):
    """Replace Unicode characters with ASCII equivalents for PDF compatibility."""
    replacements = {
        '\u2013': '-',   # en dash
        '\u2014': '-',   # em dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2022': '*',   # bullet
        '\u2026': '...', # ellipsis
        '\u00a0': ' ',   # non-breaking space
        '—': '-',        # em dash
        '–': '-',        # en dash
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def generate_prescription_pdf(prescription_list, patient_name=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Use standard Arial font (Latin-1 only, but text is cleaned)
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(0, 10, "MEDICAL PRESCRIPTION", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, "AI-Prescription Helper", ln=True, align="C")
    pdf.cell(0, 5, datetime.now().strftime("%Y-%m-%d %H:%M"), ln=True, align="C")
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(0, 10, "Patient Information:", ln=True)
    pdf.set_font("Arial", size=11)
    patient_name_clean = clean_text(patient_name if patient_name else "_______________")
    pdf.cell(0, 8, f"Name: {patient_name_clean}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
    pdf.ln(5)
    
    # Medicines Table Header
    pdf.set_font("Arial", size=11, style="B")
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(70, 10, "Medicine Name", border=1, align="C", fill=True)
    pdf.cell(120, 10, "Dosage Instructions", border=1, align="C", fill=True)
    pdf.ln()
    
    # Medicines Table Rows
    pdf.set_font("Arial", size=10)
    for med in prescription_list:
        medicine_name = clean_text(med["name"])
        dosage = clean_text(med["dosage"])
        pdf.cell(70, 8, medicine_name, border=1)
        pdf.multi_cell(120, 8, dosage, border=1)
        # Adjust Y position after multi_cell
        pdf.set_y(pdf.get_y())
    pdf.ln(5)
    
    # Doctor Signature & Notes
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, "Prescribing Physician: ________________________", ln=True)
    pdf.cell(0, 8, "Signature: ________________________", ln=True)
    pdf.cell(0, 8, "Stamp: ________________________", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", size=8, style="I")
    pdf.cell(0, 5, "This is a computer-generated prescription. Valid only with doctor's signature.", ln=True, align="C")
    
    # Return PDF as bytes (Latin-1 safe because text is cleaned)
    return pdf.output(dest='S').encode('latin-1', errors='replace')

def display_response(content, msg_idx):
    if isinstance(content, dict) and "matched_conditions" in content:
        matched = content["matched_conditions"]
        if not matched:
            st.info("No conditions found.")
            return
        
        st.markdown("### 📋 Suggested Medicines")
        
        if msg_idx not in st.session_state.prescription_lists:
            st.session_state.prescription_lists[msg_idx] = []
        
        for condition_data in matched:
            condition_name = condition_data["condition"]
            medicines = condition_data.get("medicines", [])
            if not medicines:
                continue
            
            st.markdown(f"#### 🩺 {condition_name}")
            for med in medicines:
                col1, col2, col3 = st.columns([3, 4, 1])
                with col1:
                    st.write(med["name"])
                with col2:
                    st.write(med["dosage"])
                with col3:
                    already_added = any(
                        p["condition"] == condition_name and p["name"] == med["name"]
                        for p in st.session_state.prescription_lists[msg_idx]
                    )
                    if not already_added:
                        if st.button("➕ Add", key=f"add_{msg_idx}_{condition_name}_{med['name']}"):
                            st.session_state.prescription_lists[msg_idx].append({
                                "condition": condition_name,
                                "name": med["name"],
                                "dosage": med["dosage"]
                            })
                            st.rerun()
                    else:
                        if st.button("❌ Remove", key=f"del_{msg_idx}_{condition_name}_{med['name']}"):
                            st.session_state.prescription_lists[msg_idx] = [
                                p for p in st.session_state.prescription_lists[msg_idx]
                                if not (p["condition"] == condition_name and p["name"] == med["name"])
                            ]
                            st.rerun()
            st.divider()
        
        # Current prescription section
        if st.session_state.prescription_lists[msg_idx]:
            st.markdown("### ✅ Current Prescription")
            pres_df = pd.DataFrame(st.session_state.prescription_lists[msg_idx])
            pres_df = pres_df.rename(columns={"condition": "Condition", "name": "Medicine", "dosage": "Dosage"})
            st.dataframe(pres_df, use_container_width=True, hide_index=True)
            
            patient_name = st.text_input("Patient Name (for prescription)", key=f"patient_name_{msg_idx}", placeholder="Enter patient name")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📄 Show Final Prescription JSON", key=f"final_{msg_idx}"):
                    final_pres = st.session_state.prescription_lists[msg_idx]
                    st.json(final_pres)
                    text_pres = "\n".join([f"- {p['condition']}: {p['name']} – {p['dosage']}" for p in final_pres])
                    st.text(text_pres)
            
            with col2:
                if st.session_state.prescription_lists[msg_idx]:
                    try:
                        pdf_bytes = generate_prescription_pdf(
                            st.session_state.prescription_lists[msg_idx],
                            patient_name
                        )
                        st.download_button(
                            label="📥 Download Prescription as PDF",
                            data=pdf_bytes,
                            file_name=f"prescription_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key=f"pdf_{msg_idx}"
                        )
                    except Exception as e:
                        st.error(f"PDF generation error: {e}")
        else:
            st.info("No medicines added yet. Use + to add.")
        
        with st.expander("View raw AI response"):
            st.json(content)
    elif isinstance(content, dict) and "error" in content:
        st.error(content["error"])
    else:
        st.markdown(content)

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            display_response(msg["content"], idx)
        else:
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Describe your symptoms or ask a medical question..."):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    if not os.path.exists(tree_file):
        st.error(f"Medical knowledge file '{tree_file}' not found.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("🔍 Analyzing..."):
        try:
            rag = TreeRAG(tree_file_path=tree_file, gemini_api_key=api_key, model=model_name)
            answer = rag.ask(prompt)
        except Exception as e:
            answer = {"error": f"Unexpected error: {str(e)}"}
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        display_response(answer, len(st.session_state.messages) - 1)
    st.rerun()
