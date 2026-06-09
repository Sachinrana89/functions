from dotenv import load_dotenv

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Mistral Embedding Model
embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# Load Chroma Vector Database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# Mistral LLM
llm = ChatMistralAI(
    model="mistral-small-2506"
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),

        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("RAG system created")
print("Press 0 to exit")

while True:

    query = input("\nYou: ")

    if query == "0":
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create final prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate response
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}")