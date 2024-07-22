import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings
warnings.filterwarnings("ignore")
# ADD YOUR FILE PATH HERE
# GO TO LLM FUNCTION AND PUT API KEY
files_path=['/home/centrox/Downloads/Haider_Zaidi_Employment_Agreement_signed.pdf']

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredFileLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_community.document_loaders import TextLoader
import logging
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_together import Together

def get_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower()
def load_pdf_file(file):
    loader = PyPDFLoader(file)
    pages = loader.load()
    return pages
def load_csv_file(file):
    loader = CSVLoader(file)
    pages = loader.load()
    return pages
def load_html_file(file):
    loader = BSHTMLLoader(file)
    pages = loader.load()
    return pages
def load_txt_file(file):
    loader = TextLoader(file)
    pages = loader.load()
    return pages
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
def LLM(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.5, tokens=256):
    """
    Create and initialize a Together AI language model.

    Parameters:
    - model_name (str, optional): The name of the Together AI language model.
    - temperature (float, optional): The parameter for randomness in text generation.
    - tokens (int, optional): The maximum number of tokens to generate.

    Returns:
    - llm (Together): The initialized Together AI language model.
    """

    # ADD YOUR OWN API KEY ALSO SIMPLE CHANGE FROM TOGETHER to OpenAI and also chage model name at top IF YOU DONT HAVE THE API 
    together_api_key=''
    llm = Together(
        model=model_name,
        temperature=temperature,
        max_tokens=tokens,
        together_api_key=together_api_key
    )
    return llm

def get_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower()
def load_split(i,text_splitter,embeddings):
    extension = get_file_extension(i)
    print(f"The file extension is: {extension}")
    if extension==".txt":
        pages=load_txt_file(i)
    elif extension==".pdf":
        pages=load_pdf_file(i)
    elif extension==".csv":
        pages=load_csv_file(i)
    print(f"The number of pages are: {len(pages)}")
    doc = text_splitter.split_documents(pages)
    return doc
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=1000,
    length_function=len,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
llm=LLM()
docs=load_split(files_path[0],text_splitter,embeddings)
print(f"The length of docs are: {len(docs)}")
vectordb =  FAISS.from_documents(
documents=docs,
embedding=embeddings
)
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer the questions as accurate as possible based on context provided. Always say "thanks for asking!" at the end of the answer. The user prompt would be your supreme order and you have to strictly follow it.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(
llm,
retriever=vectordb.as_retriever(),
return_source_documents=True,
chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain({"query": '''What are the positions and duties?'''})
print(result["result"])