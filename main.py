# Override default sqlite3 with pysqlite3-binary
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain_community.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

def generate_response(uploaded_file, openai_api_key, query_text):
    try:
        # Load document if file is uploaded
        if uploaded_file is not None:
            try:
                documents = [uploaded_file.read().decode('utf-8')]
            except UnicodeDecodeError:
                st.error("Error: Unable to decode the uploaded file. Please ensure it's a valid UTF-8 encoded text file.")
                return None
        else:
            st.error("Error: No file uploaded.")
            return None

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)

        # Select embeddings
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        # Create retriever
        retriever = db.as_retriever()
        # Initialize LLM
        llm = OpenAI(api_key=openai_api_key)
        # Define prompt template (mimicking 'stuff' chain type)
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Question: {question}
        Answer: """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        # Create retrieval chain
        qa_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": prompt, "llm": llm}
        )
        # Run query
        response = qa_chain.invoke({"question": query_text})
        return response["answer"]
    except Exception as e:
        st.error(f"Error processing the request: {str(e)}")
        return None

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            if response:
                result.append(response)
                del openai_api_key

if len(result):
    st.info(response)
