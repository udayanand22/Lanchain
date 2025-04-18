# narad.py

import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Setup LLM and tools
cohere_api_key = os.getenv("COHERE_API_KEY")
llm = ChatCohere(cohere_api_key=cohere_api_key, model="command-r")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt
prompt = PromptTemplate.from_template(
    """You are Narada, the celestial sage, known for your dramatic flair and poetic storytelling.
{chat_history}
User: Tell me a story about {topic}
Narada:"""
)

# 🧠 Chain as a Runnable
story_chain = RunnableMap({
    "topic": lambda x: x["topic"],
    "chat_history": lambda _: "\n".join(
        [f"{m.type.title()}: {m.content}" for m in memory.chat_memory.messages]
    )
}) | prompt | llm | StrOutputParser()

# 👇 API and CLI handler
def get_narada_story(topic: str):
    try:
        raw_story = story_chain.invoke({"topic": topic})
        raw_story = f"Narayan Narayan! {raw_story.strip()}"  # <-- inject signature
        memory.chat_memory.add_user_message(f"Tell me a story about {topic}")
        memory.chat_memory.add_ai_message(raw_story)

        if raw_story.strip() == "":
            return {"error": "Narada was unable to weave a tale."}

        summary = summarizer(raw_story, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        result = qa_pipeline(question=topic, context=summary)

        if result['score'] < 0.5:
            return {"message": f"Here's a more detailed answer about {topic}:\n{raw_story}"}
        else:
            return {"message": summary}

    except Exception as e:
        return {"error": str(e)}

# 👇 Interactive CLI (optional)
if __name__ == "__main__":
    print("🦚 Narada Cohere Edition Activated — Now With Memory!")

    while True:
        topic = input("🎤 Ask Narada for a story (or type 'exit'): ")
        if topic.lower() == 'exit':
            print("Goodbye! Narada's tales will await you next time.")
            break

        result = get_narada_story(topic)

        if "error" in result:
            print("\n⚠️ Error:", result["error"])
        else:
            print("\n📜 Narada's Tale:\n", result["message"])
            print("\n🧠 Narada's Memory So Far:\n")
            for msg in memory.chat_memory.messages:
                print(f"{msg.type.title()}: {msg.content}")
