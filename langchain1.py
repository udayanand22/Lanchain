from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline  # Updated import
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load HF pipeline
pipe = pipeline("text-generation", model="gpt2", max_length=100, truncation=True)  # Added truncation

# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Create a prompt
prompt = PromptTemplate.from_template("Q: What is LangChain?\nA:")

# Use the updated method to combine prompt and pipeline
chain = prompt | llm  # Updated to use the new method

# Run it
response = chain.invoke({})  # Updated to use invoke instead of run
print(response)
