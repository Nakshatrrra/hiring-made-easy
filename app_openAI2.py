import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from uploaded PDFs using PyPDF2
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into smaller chunks for processing using CharacterTextSplitter
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Create a vectorstore from text chunks using OpenAIEmbeddings and FAISS vector DB
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Create a conversational chain to interact with the resume using gpt-4 model using ChatOpenAI
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.5,
        model="gpt-4"
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

def generate_new_question(chat_history, job_description):
    prompt = f"""
    You are an AI interviewer. Based on the following chat history and job description, suggest a new, context-aware question that follows logically from the conversation and switch topics in between if same topic question is asked for mmore than 3 times:
    
    Chat History:
    {chat_history}
    
    Job Description:
    {job_description}
    
    Provide only the question as output.
    """
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.7,
        model="gpt-4"
    )
    new_question = llm.predict(prompt).strip()
    return new_question


# Function to score user responses using ChatOpenAI
def score_response(question, user_response):
    scoring_prompt = f"""
    You are an expert interviewer. Score the candidate's response to the following question on a scale of 0 to 1, where:
    - 1 means the response is absolutely good according to the question for a basic interview.
    - 0 means the response is entirely incorrect, irrelevant, or unhelpful.
    
    Question: {question}
    Response: {user_response}
    
    Provide only the score as output.
    """
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model="gpt-4"
    )
    score = llm.predict(scoring_prompt)
    return float(score.strip())

def handle_resume_interaction(question, user_response, conversation_chain):
    follow_up_response = conversation_chain({'question': user_response})
    score = score_response(question, user_response)
    return follow_up_response['chat_history'], score

# Check if more questions are available
def continue_conversation():
    return len(st.session_state.initial_questions) > 0

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
    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    st.header("Interactive Resume Q&A :memo:")

    with st.sidebar:
        st.subheader("Your Resume")
        resume_pdf = st.file_uploader("Upload your resume PDF", accept_multiple_files=False)
        st.subheader("Job Description")
        job_description = st.text_area("Paste the job description here")
        if st.button("Process Resume"):
            with st.spinner("Processing resume..."):
                resume_text = get_pdf_text([resume_pdf])
                text_chunks = get_text_chunks(resume_text)
                vectorstore = get_vectorstore(text_chunks)

                # Initialize the conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                # Start with initial questions
                st.session_state.initial_questions = initiate_questions()
                st.session_state.job_description = job_description

    # Ask questions based on resume and user responses
    if st.session_state.conversation:
        if st.session_state.initial_questions and not st.session_state.current_question:
            st.session_state.current_question = st.session_state.initial_questions.pop(0)
            st.write(bot_template.replace("{{MSG}}", st.session_state.current_question), unsafe_allow_html=True)

        user_answer = st.text_input("Your answer:")
        if user_answer:
            chat_history, score = handle_resume_interaction(
                st.session_state.current_question,
                user_answer,
                st.session_state.conversation
            )
            st.session_state.chat_history = chat_history

            # for i, message in enumerate(st.session_state.chat_history):
            #     if i % 2 == 0:
            #         st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            #     else:
            #         st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

            st.write(f"**Score for your response:** {score:.2f}")

            if continue_conversation():
                st.session_state.current_question = st.session_state.initial_questions.pop(0)
            else:
                new_question = generate_new_question(st.session_state.chat_history, st.session_state.job_description)
                if new_question:
                    st.session_state.current_question = new_question
                else:
                    st.session_state.current_question = None
            st.write(bot_template.replace("{{MSG}}", st.session_state.current_question), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
