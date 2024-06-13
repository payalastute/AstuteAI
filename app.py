import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

# from dotenv import load_dotenv
import tempfile
from pymongo import MongoClient
from datetime import datetime



# Initialize MongoDB client
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
collection = db["chat_history"]

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything."]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

def save_to_mongodb(user_input, bot_response):
    # chat_entry = {
    #     "user_input": user_input,
    #     "bot_response": bot_response,
    #     "timestamp": datetime.utcnow()
    # }
    # collection.insert_one(chat_entry)
    pass

def conversation_chat(query, chain, history):
    result = chain({"input": query, "history": history})
    # print(result)
    print((query, result['response']))
    history.append((query, result['response']))
    save_to_mongodb(query, result['response'])  # Save chat to MongoDB
    return result['response']

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_area("Question:", placeholder="Ask about your documents", height=30)
            submit_button = st.form_submit_button(label="Send")
        if submit_button and user_input:
            with st.spinner("Generating response..."):
                output = conversation_chat(user_input, chain, st.session_state['history'])
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + "_user", avatar_style="thumbs")
                message(st.session_state['generated'][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store=None):
    # llm = CTransformers(model=r"C:\Users\hp\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa",
    #                     streaming=True,
    #                     callbacks=[StreamingStdOutCallbackHandler()],
    #                     model_type="ollama", config={'max_new_tokens': 5000, 'temperature': 0.01})
    llm = Ollama(model="llama3")
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    if vector_store:
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 2}), memory=memory)
    else:
        chain = ConversationChain(llm=llm, memory=memory)
    return chain



def main():
    initialize_session_state()
    st.title("Multi-Docs Chatbot using LLaMa 3 📚")
    st.sidebar.title("Document Processing")
    upload_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    vector_store = None
    if upload_files:
        text = []
        for file in upload_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name
                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)
                elif file_extension == ".csv":
                    loader = CSVLoader(temp_file_path)
                elif file_extension == ".txt":
                    loader = TextLoader(temp_file_path)
                elif file_extension == ".docx" or file_extension == ".doc":
                    loader = Docx2txtLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)


        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=1000, length_function=len)
        text_chunks = text_splitter.split_documents(text)
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
        vector_store = FAISS.from_documents(text_chunks)
    chain = create_conversational_chain(vector_store)
    display_chat_history(chain)


if __name__ == "__main__":
    main()

