## Initialize the environment variables
import os
from dotenv import load_dotenv

# Initialize the libraries and modules needed
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OLLAMA_PROJECT"] = os.getenv("OLLAMA_PROJECT")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a smart, romantic AI assistant. Please do respond to the questions asked in a romantic way possibly. If not just give a good answer like a gentleman"),
        ("user", "Question:{question}")
    ]
)

#Streamlit framework
st.title("My personal AI assistant Hehe")
input_text = st.text_input("What question do you have in your mind? Spill it out!! Lemme help you XD")

##Ollama Model
llm = OllamaLLM(model="deepseek-r1:1.5b")
output_parser = StrOutputParser()

#chain
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
