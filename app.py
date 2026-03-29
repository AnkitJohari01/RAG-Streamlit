import os

import streamlit as st

from rag_utility import process_document_to_chromadb, answer_question, working_dir

st.title("💻 Llama-3.3-70b - Document RAG")

# File uploaded widget

uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])

if uploaded_file is not None:
    save_path = os.path.join(working_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chromadb(uploaded_file.name)
    st.info("Document Processed Sucessfully")


# Text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer = answer_question(user_question)

    st.markdown("### Llama-3.3-70b Response")
    st.markdown(answer)
