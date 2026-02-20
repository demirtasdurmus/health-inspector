from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from . import config

def load_database():
  """
  Load database from disk

  Args:
    N/A

  Returns:
    Db instance
  """
  embeddings=OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
  db=Chroma(persist_directory=config.DB_PATH,embedding_function=embeddings)

  print("✅ Database loaded from disk!")
  return db

def get_vision_observation():
  """
  Gets vision observation from the provided photos&videos
  For now returns static text for simplicity

  Args:
    A list of images or a video(s)

  Returns:
    Vision observation generated from images/video(s) by AI model
  """
  vision_observation="Mutfak tezgahının üzerinde ahşap bir kesme tahtası duruyor."

  print(f"✅ Vision Observation: '{vision_observation}'")

  return vision_observation

def search_laws(db, vision_observation):
  """
  Gets similar results through RAG similarity search

  Args:
    db: DB instance
    vision_observation: text to search

  Returns:
    Similar results
  """
  # 3. k=3 for more context
  results=db.similarity_search(vision_observation, k=config.DEFAULT_TOP_K)

  print(f"✅ Found {len(results)} matches")

  return results

def run_judge(observation, laws):
  """
  Runs LLM for each of the matched laws

  Args:
    observation: Vision observation for the provided media
    laws: Matched laws after RAG search

  Returns
    verdict: Verdict of the judge regarding violated laws
  """
  for i, doc in enumerate(laws, 1):
    print(f"Match {i}: {doc.page_content[:100]}...")
    print(f"Source: {doc.metadata.get('resource', 'unknown')}\n")

  judge_llm=ChatOpenAI(temperature=config.LLM_TEMPERATURE, model=config.LLM_MODEL)
  laws_context="\n\n".join([doc.page_content for doc in laws])

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
      "observation": observation,
      "laws": laws_context
  })

  return verdict.content