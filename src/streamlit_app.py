import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Adjust if backend runs elsewhere

st.title("Document Preprocessing Frontend")

st.header("1. Upload Document")
document_id = st.text_input("Document ID", "doc1")
file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg", "tiff", "bmp"])
image_url = st.text_input("Or provide an image URL")

if st.button("Start Preprocessing"):
    if not document_id:
        st.error("Please provide a document ID.")
    elif not file and not image_url:
        st.error("Please upload a file or provide an image URL.")
    else:
        files = {"file": file} if file else None
        data = {"image_url": image_url} if image_url and not file else None
        try:
            response = requests.post(f"{API_URL}/preprocess/{document_id}", files=files, data=data)
            if response.status_code == 200:
                st.success(f"Processing started for {document_id}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.header("2. Check Processing Status")
status_doc_id = st.text_input("Document ID to check status", "doc1", key="status_doc_id")
if st.button("Check Status"):
    try:
        response = requests.get(f"{API_URL}/status/{status_doc_id}")
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
