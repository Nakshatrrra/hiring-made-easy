import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# Load environment variables
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_KEY")

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

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

# Create a vectorstore from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Create a conversational chain to interact with the resume
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small", 
        model_kwargs={"temperature": 0.5, "max_length": 512}, 
        huggingfacehub_api_token=huggingface_api_key
    )
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# New function to initiate the conversation with questions based on the resume
def initiate_questions():
    initial_questions = [
        "Can you walk me through your work experience in your resume?",
        "What are the key technical skills you have mentioned?",
        "Can you explain the most challenging project you worked on?",
    ]
    return initial_questions

# Function to handle user input and generate follow-up questions
def handle_resume_interaction(user_response, conversation_chain):
    # Process the user's answer and ask a follow-up question
    follow_up_response = conversation_chain({'question': user_response})
    return follow_up_response['chat_history']

# Main function to handle the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Interactive Resume Q&A", page_icon=":memo:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "initial_questions" not in st.session_state:
        st.session_state.initial_questions = None

    st.header("Interactive Resume Q&A :memo:")

    # Input: Resume as PDF
    with st.sidebar:
        st.subheader("Your Resume")
        resume_pdf = st.file_uploader("Upload your resume PDF", accept_multiple_files=False)
        if st.button("Process Resume"):
            with st.spinner("Processing resume..."):
                # Extract text from the resume
                resume_text = get_pdf_text([resume_pdf])
                text_chunks = get_text_chunks(resume_text)
                vectorstore = get_vectorstore(text_chunks)

                # Initialize the conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                # Start with initial questions
                st.session_state.initial_questions = initiate_questions()
# yes i've done 3 internships in full stack development
    # Ask questions based on resume and user responses
    if st.session_state.conversation:
        # Ask the first question
        if st.session_state.initial_questions:
            st.write(bot_template.replace("{{MSG}}", st.session_state.initial_questions[0]), unsafe_allow_html=True)
            st.session_state.initial_questions.pop(0)
        
        # Input field for user's answer
        user_answer = st.text_input("Your answer:")
        if user_answer:
            # Handle user input and get follow-up
            chat_history = handle_resume_interaction(user_answer, st.session_state.conversation)
            st.session_state.chat_history = chat_history

            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            # Continue asking questions based on user's answer
            if len(st.session_state.initial_questions) > 0:
                st.write(bot_template.replace("{{MSG}}", st.session_state.initial_questions[0]), unsafe_allow_html=True)
                st.session_state.initial_questions.pop(0)

if __name__ == '__main__':
    main()
