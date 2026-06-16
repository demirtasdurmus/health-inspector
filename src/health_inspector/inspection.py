import base64
import io
from pathlib import Path

import pillow_heif
from PIL import Image
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

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

DEFAULT_IMAGE_PATHS = [
    "resources/images/IMG_4047.jpg",
    "resources/images/IMG_4048.jpg",
    "resources/images/IMG_4049.jpg",
]

VISION_PROMPT = """Sen bir gıda denetimi tutanağı yazan denetçisisin. Sana verilen görüntüleri incele ve gördüklerini nesnel biçimde kaydet.

GÖREV:
- Görüntülerde ne gördüğünü gıda güvenliği ve hijyen açısından önemli olacak şekilde tarafsız ve olgusal olarak açıkla: tezgahlar, ekipmanlar (kesme tahtası, bıçak, ocak vb.), yüzeyler, malzemeler (ahşap, plastik, cam vb.), nesneler ve genel düzen.
- Yorum yapma, değerlendirme yapma, öneri verme veya iyi/kötü gibi nitelendirmeler kullanma.
- Sadece gözle görüleni yaz; çıkarım veya varsayım ekleme.
- Cevabını Türkçe, düz metin (tek paragraf) olarak ver."""

def _load_image_as_base64(image_path: str) -> str:
  """Loads a HEIC or standard image and returns a base64-encoded JPEG string."""
  path = Path(image_path)
  if path.suffix.upper() == ".HEIC":
    heif_file = pillow_heif.read_heif(str(path))
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
  else:
    image = Image.open(path)

  buffer = io.BytesIO()
  image.convert("RGB").save(buffer, format="JPEG")
  return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_vision_observation(image_paths: list[str] = None) -> str:
  """
  Sends images to a vision-capable LLM and returns an inspection-context description.

  Args:
    image_paths: List of paths to image files (HEIC or standard formats).
                 Defaults to the three sample images in resources/images/.

  Returns:
    Vision observation text describing the scene from a food safety perspective.
  """
  if image_paths is None:
    image_paths = DEFAULT_IMAGE_PATHS

  image_contents = []
  for path in image_paths:
    b64 = _load_image_as_base64(path)
    image_contents.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    })

  image_contents.append({"type": "text", "text": VISION_PROMPT})

  vision_llm = ChatOpenAI(model=config.VISION_MODEL, temperature=config.LLM_TEMPERATURE)
  message = HumanMessage(content=image_contents)
  response = vision_llm.invoke([message])

  vision_observation = response.content
  print(f"✅ Vision observation received:\n {vision_observation})")

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