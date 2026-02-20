from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import config

# 1. Load the pre-built db
embeddings=OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
db=Chroma(persist_directory=config.DB_PATH,embedding_function=embeddings)

print("✅ Database loaded from disk!")

# 2. Vision Observation
vision_observation="Mutfak tezgahının üzerinde ahşap bir kesme tahtası duruyor."

print(f"✅ Vision Observation: '{vision_observation}'")
print("-"*50)

# 3. RAG Search(k=3 for more context)
results=db.similarity_search(vision_observation, k=config.DEFAULT_TOP_K)

if not results:
    print("❌ No results found for the vision observation.")
    exit(1)
else:
    for i, doc in enumerate(results, 1):
        print(f"Match {i}: {doc.page_content[:100]}...")
        print(f"Source: {doc.metadata.get('resource', 'unknown')}\n")

# 4. Judge AI Decision
    judge_llm=ChatOpenAI(temperature=config.LLM_TEMPERATURE, model=config.LLM_MODEL)
    laws_context="\n\n".join([doc.page_content for doc in results])

    prompt_template=ChatPromptTemplate.from_template("""
    Sen Türk Gıda Mevzuatı denetçisisin. Aşağıdaki kanıtı ve ilgili yasaları incele.
    
    KANIT (Gözlem): {observation}
    
    İLGİLİ YASALAR:
    {laws}
    
    GÖREV:
    Bu durum bir ihlal midir? Neden? Hangi mevzuata göre?
    Cevabı JSON formatında ver: {{"violation": true/false, "explanation": "...", "risk_level": "...", "mevzuat": "..."}}
    """)

    chain = prompt_template | judge_llm
    verdict = chain.invoke({
        "observation": vision_observation,
        "laws": laws_context
    })

    print("=" * 50)
    print("INSPECTION REPORT:")
    print(verdict.content)

###########
# import os


# def main():
#     print("Hello from health-inspector!")
#     print(os.getenv('OPENAI_API_KEY'))


# if __name__ == "__main__":
#     main()
