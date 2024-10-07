import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_KEY")

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts.append(text)
    return texts

# Split text into smaller chunks for processing
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to compute embeddings
def compute_embeddings(texts):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    embedded_texts = embeddings.embed_documents(texts)  # Use embed_documents for multiple texts
    return embedded_texts

# Function to match description with PDFs and find the best match
def find_best_match(pdf_texts, description_embedding):
    # Compute embeddings for each PDF text
    pdf_embeddings = compute_embeddings(pdf_texts)

    # Calculate cosine similarity between the description and each PDF embedding
    similarities = [cosine_similarity([description_embedding], [pdf_emb])[0][0] for pdf_emb in pdf_embeddings]

    # Find the index of the PDF with the highest similarity score
    best_match_index = np.argmax(similarities)

    return best_match_index, similarities[best_match_index]

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Find Best Matching PDF", page_icon=":mag_right:")

    st.header("Find the Best Matching PDF Based on a Description :mag_right:")

    # Hardcoded description (You can modify this)
    hardcoded_description = """
    Full-stack developer with experience in AI, ML, backend technologies, and modern web development frameworks.
    Expertise in Python, JavaScript, Node.js, and React.js. Seeking opportunities to leverage these skills.
    """

    # Input: Multiple PDFs
    with st.sidebar:
        st.subheader("Upload Multiple PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here:", accept_multiple_files=True)

    if pdf_docs:
        with st.spinner("Processing PDFs..."):
            # Extract text from the uploaded PDFs
            pdf_texts = get_pdf_text(pdf_docs)

            # Compute embedding for the hardcoded description
            description_embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base").embed_documents([hardcoded_description])[0]

            # Find the best matching PDF
            best_match_index, best_similarity = find_best_match(pdf_texts, description_embedding)

            st.success(f"The best matching PDF is: {pdf_docs[best_match_index].name} with a similarity score of {best_similarity:.2f}")

            # Provide the option to download or view the best matching PDF
            with open(pdf_docs[best_match_index].name, "rb") as best_pdf:
                st.download_button("Download Best Matching PDF", best_pdf, file_name=pdf_docs[best_match_index].name)

if __name__ == '__main__':
    main()
