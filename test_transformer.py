from transformers import pipeline

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = "LangChain is a framework that allows chaining of LLMs like OpenAI or Hugging Face models for custom tasks."
question = "What is LangChain?"

result = qa(question=question, context=context)
print("Answer:", result["answer"])
