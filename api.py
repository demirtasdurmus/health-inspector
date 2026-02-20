from dotenv import load_dotenv
load_dotenv()

import sys
import json
import re
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from health_inspector.inspection import load_database, get_vision_observation, search_laws, run_judge

app = FastAPI(title="Health Inspector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the vector DB once at startup and reuse across requests
db = load_database()

# Sync pipeline runs in a thread pool so the async event loop stays unblocked
executor = ThreadPoolExecutor()


def _parse_verdict(raw: str) -> dict:
    """Strip markdown code fences from run_judge() output and parse to dict."""
    # Remove ```json ... ``` or ``` ... ``` wrappers if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _run_pipeline(image_paths: list[str]) -> dict:
    """Execute the full inspection pipeline synchronously."""
    observation = get_vision_observation(image_paths)
    matched_laws = search_laws(db, observation)

    if not matched_laws:
        raise ValueError("No matching regulations found for the given images.")

    verdict_raw = run_judge(observation, matched_laws)
    verdict = _parse_verdict(verdict_raw)

    return {
        "vision_observation": observation,
        "matched_laws": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("resource", "unknown"),
            }
            for doc in matched_laws
        ],
        "verdict": verdict,
    }


@app.post("/inspect")
async def inspect(images: list[UploadFile] = File(...)):
    """
    Accept one or more image files (HEIC, JPEG, PNG, etc.) as multipart/form-data.
    Returns a structured JSON inspection report.
    """
    import asyncio

    # Write each upload to a named temp file so pillow / pillow-heif can read it
    tmp_paths: list[str] = []
    tmp_files = []
    try:
        for upload in images:
            suffix = Path(upload.filename or "image").suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_files.append(tmp)
            contents = await upload.read()
            tmp.write(contents)
            tmp.flush()
            tmp_paths.append(tmp.name)

        # Run the blocking pipeline off the async event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _run_pipeline, tmp_paths)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for tmp in tmp_files:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)

    return result
