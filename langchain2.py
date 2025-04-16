from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Setup for DeepSeek for text generation
deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
deepseek_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1")
deepseek_pipe = pipeline("text-generation", model=deepseek_model, tokenizer=deepseek_tokenizer)

# Use HuggingFacePipeline for LangChain v1.0+
deepseek_llm = HuggingFacePipeline(pipeline=deepseek_pipe)

# Setup for Facebook BART for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Combine both in LangChain: DeepSeek for generation, BART for summarization
def generate_and_summarize(input_text):
    # Generate text using DeepSeek
    generated_text = deepseek_llm.invoke(input_text)  # Updated to 'invoke'
    print("Generated Text:", generated_text)

    # Summarize the generated text using BART
    # Ensure the text is properly handled as the summarizer expects strings
    if isinstance(generated_text, str):
        generated_text = [generated_text]
    
    summary = summarizer(generated_text[0], max_length=100, min_length=30, do_sample=False)
    print("\nSummary:", summary[0]['summary_text'])

# Example input text
input_text = "LangChain is an open-source framework for developing applications powered by language models. It provides a set of tools and abstractions that allow developers to build pipelines for natural language processing tasks such as question answering, text summarization, translation, and more."

# Call the function
generate_and_summarize(input_text)
