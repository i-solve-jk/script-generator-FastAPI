"""
AI Video Script Generator — Backend API.

Provides: prompt option extraction (Groq), prompt enhancement, script generation,
and video generation via Leonardo AI or Replicate. Long scripts are condensed to
fit Leonardo's 1500-char limit; duration options map to 4/6/8s for Leonardo VEO3.
"""
import os
import json
import re
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai-video")

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
LEONARDO_API_KEY = os.getenv("LEONARDO_API_KEY")
REPLICATE_API_BASE = "https://api.replicate.com/v1"
LEONARDO_API_BASE = "https://cloud.leonardo.ai/api/rest/v1"
GROQ_MODEL = "llama-3.1-8b-instant"
MAX_PROMPT_LENGTH = 5000  # max chars for prompt/script in our API
LEONARDO_MAX_PROMPT_LENGTH = 1500  # Leonardo text-to-video API limit
VIDEO_POLL_INTERVAL = int(os.getenv("VIDEO_POLL_INTERVAL", "5"))  # seconds between status checks
VIDEO_MAX_WAIT = int(os.getenv("VIDEO_MAX_WAIT", "900"))  # max seconds to wait for video (default 15 min)

# Leonardo dimension presets: size label -> (width, height)
# MOTION2 uses these; VEO3/VEO3FAST use fixed 1280x720
LEONARDO_DIMENSIONS: dict[str, tuple[int, int]] = {
    "Landscape (16:9)": (832, 480),
    "Vertical (9:16)": (480, 832),
    "Square (1:1)": (480, 480),
}
# Leonardo only supports 4, 6, or 8 seconds. VEO3/VEO3FAST support duration; MOTION2 is ~5s fixed.
LEONARDO_DURATION_SECONDS: dict[str, int] = {
    "15 seconds": 4,
    "30 seconds": 6,
    "60 seconds": 8,
    "90 seconds": 8,
    "2 minutes": 8,
    "3 minutes": 8,
    "5 minutes": 8,
}

# Allowed values for extracted options; used to normalize LLM output and validate UI.
VALID_OPTIONS: dict[str, list[str]] = {
    "duration": ["15 seconds", "30 seconds", "60 seconds", "90 seconds", "2 minutes", "3 minutes", "5 minutes"],
    "language": ["English","Tamil", "Telugu", "Kannada", "Malayalam", "Hindi", "Spanish", "French", "German", "Japanese", "Chinese", "Arabic", "Portuguese"],
    "platform": ["YouTube", "Instagram", "TikTok", "LinkedIn", "Facebook", "Twitter/X"],
    "size": ["Landscape (16:9)", "Vertical (9:16)", "Square (1:1)"],
    "category": ["Kids", "Education", "Entertainment", "Marketing", "Tutorial", "Vlog", "Documentary", "Travel", "Fitness", "Cooking", "Tech"],
}

# ── Singleton clients ────────────────────────────────────────────────────────
_groq_client: Groq | None = None
_http_client: httpx.AsyncClient | None = None


def get_groq_client() -> Groq:
    """Lazy-init Groq client; raises if GROQ_API_KEY is missing."""
    global _groq_client
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured. Add it to backend/.env")
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_http_client() -> httpx.AsyncClient:
    """Lazy-init shared HTTP client for Leonardo/Replicate (reused across requests)."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=60.0))
    return _http_client


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting AI Video Script Generator")
    yield
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
    logger.info("Shut down cleanly")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Video Script Generator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalize_value(value: str, valid_options: list[str]) -> str:
    """Match user/LLM text to one of the allowed option strings (e.g. 'portrait' -> 'Vertical (9:16)')."""
    if not value:
        return ""
    val_lower = value.strip().lower()
    for opt in valid_options:
        if opt.lower() == val_lower or opt.lower() in val_lower or val_lower in opt.lower():
            return opt
    return ""


def normalize_options(raw: dict[str, Any]) -> dict[str, str]:
    """Normalize each option key from raw LLM JSON to a valid value from VALID_OPTIONS."""
    return {key: normalize_value(raw.get(key, ""), opts) for key, opts in VALID_OPTIONS.items()}


def build_options_context(opts: "ExtractedOptions", label: str = "Options") -> str:
    """Format options for inclusion in LLM prompts (extract/enhance/script)."""
    return (
        f"\n{label}:\n"
        f"- Duration: {opts.duration or 'Not specified'}\n"
        f"- Language: {opts.language or 'Not specified'}\n"
        f"- Platform: {opts.platform or 'Not specified'}\n"
        f"- Size: {opts.size or 'Not specified'}\n"
        f"- Category: {opts.category or 'Not specified'}\n"
    )


def groq_chat(system: str, user: str, *, temperature: float = 0.5, max_tokens: int = 500) -> str:
    """Single round-trip chat with Groq (system + user message); returns assistant text."""
    client = get_groq_client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def extract_video_url(output: Any) -> str:
    """Get a single video URL from Replicate output (string, list of URLs, or other)."""
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        return str(output[0])
    return str(output)


def condense_for_video_prompt(long_text: str, max_chars: int = 1400) -> str:
    """Use Groq to condense a long script into one short video prompt (for Leonardo's 1500-char limit)."""
    condensed = groq_chat(
        CONDENSE_SCRIPT_FOR_VIDEO_PROMPT,
        long_text,
        temperature=0.4,
        max_tokens=600,
    )
    if len(condensed) > max_chars:
        condensed = condensed[:max_chars].rsplit(" ", 1)[0] or condensed[:max_chars]
    return condensed.strip()


# ── Request / Response Models ────────────────────────────────────────────────
class PromptRequest(BaseModel):
    """Input for /api/extract-options."""
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)


class ExtractedOptions(BaseModel):
    """Duration, language, platform, size, category — normalized to VALID_OPTIONS values."""
    duration: str = ""
    language: str = ""
    platform: str = ""
    size: str = ""
    category: str = ""


class EnhanceRequest(BaseModel):
    """Input for /api/enhance-prompt."""
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    options: ExtractedOptions


class GenerateScriptRequest(BaseModel):
    """Input for /api/generate-script."""
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    options: ExtractedOptions


class GenerateVideoRequest(BaseModel):
    """Input for /api/generate-video. Prompt can be enhanced prompt or full script (condensed if long)."""
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    options: ExtractedOptions
    provider: str = Field(default="leonardo", pattern="^(replicate|leonardo)$")


class ExtractResponse(BaseModel):
    """Normalized options dict + original prompt."""
    options: dict[str, str]
    original_prompt: str


class EnhanceResponse(BaseModel):
    enhanced_prompt: str


class ScriptResponse(BaseModel):
    script: str


class VideoResponse(BaseModel):
    """Returned by /api/generate-video: URL of generated video and provider used."""
    video_url: str
    status: str
    provider: str = ""


# ── Routes & LLM Prompts ─────────────────────────────────────────────────────
EXTRACT_SYSTEM_PROMPT = """You are an expert at analyzing video creation prompts. Extract the following fields from the user's prompt. If a field is not mentioned or unclear, return an empty string for that field.

Return ONLY valid JSON with these exact keys and ONLY use the allowed values listed below:

{
  "duration": "<one of: '15 seconds','30 seconds','60 seconds','90 seconds','2 minutes','3 minutes','5 minutes', or ''>",
  "language": "<one of: 'English','Hindi','Spanish','French','German','Japanese','Chinese','Arabic','Portuguese', or ''>",
  "platform": "<one of: 'YouTube','Instagram','TikTok','LinkedIn','Facebook','Twitter/X', or ''>",
  "size": "<one of: 'Landscape (16:9)','Vertical (9:16)','Square (1:1)', or ''. Map landscape/widescreen to 'Landscape (16:9)', portrait/vertical/reels/shorts to 'Vertical (9:16)', square to 'Square (1:1)'>",
  "category": "<one of: 'Kids','Education','Entertainment','Marketing','Tutorial','Vlog','Documentary','Travel','Fitness','Cooking','Tech', or ''>"
}

Do not include any text outside the JSON object."""

ENHANCE_SYSTEM_PROMPT = """You are an expert cinematic video prompt enhancer. Given a user's video prompt and extracted options, enhance the prompt to be more detailed, vivid, and cinematic while preserving the original intent.

Guidelines:
- Make the prompt more descriptive and visually rich
- Add cinematic language (camera angles, lighting, transitions)
- Incorporate the provided options naturally into the prompt
- Keep the enhanced prompt concise but powerful (2-4 sentences max)
- Return ONLY the enhanced prompt text, nothing else"""

CONDENSE_SCRIPT_FOR_VIDEO_PROMPT = """You are an expert at turning long video scripts into a single, vivid text-to-video prompt.

Given a long script (with scenes, descriptions, etc.), output ONE concise video prompt (under 1400 characters) that captures the main visual story, mood, and key moments. Use vivid, cinematic language. No scene numbers or section headers—just one flowing description a video AI can generate from. Return ONLY that prompt, nothing else."""

SCRIPT_SYSTEM_PROMPT = """You are an expert cinematic video script writer. Generate a detailed, scene-by-scene video script based on the provided prompt and options.

Format the script as follows:
- Title
- Overview (1-2 lines)
- Scene-by-scene breakdown with:
  - Scene number and title
  - Visual description (camera angle, lighting, setting)
  - Action/narration
  - Duration of scene
  - Transition to next scene
- Closing notes (music/mood suggestions)

Make it cinematic, professional, and production-ready. Use vivid visual language."""

# Fallback when extract-options returns invalid JSON
EMPTY_OPTIONS = {"duration": "", "language": "", "platform": "", "size": "", "category": ""}


@app.post("/api/extract-options", response_model=ExtractResponse)
async def extract_options(request: PromptRequest):
    """Extract duration, language, platform, size, category from user prompt via Groq."""
    logger.info("Extracting options from prompt (%d chars)", len(request.prompt))
    try:
        response_text = groq_chat(EXTRACT_SYSTEM_PROMPT, request.prompt, temperature=0.1, max_tokens=300)
        json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
        parsed = json.loads(json_match.group() if json_match else response_text)
        options = normalize_options(parsed)
        logger.info("Extracted options: %s", options)
        return ExtractResponse(options=options, original_prompt=request.prompt)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON; returning empty options")
        return ExtractResponse(options=EMPTY_OPTIONS, original_prompt=request.prompt)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Extract failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/enhance-prompt", response_model=EnhanceResponse)
async def enhance_prompt(request: EnhanceRequest):
    """Enhance user prompt with cinematic language using Groq."""
    logger.info("Enhancing prompt (%d chars)", len(request.prompt))
    try:
        ctx = build_options_context(request.options)
        user_msg = f"Original prompt: {request.prompt}\n{ctx}\n\nEnhance this prompt:"
        enhanced = groq_chat(ENHANCE_SYSTEM_PROMPT, user_msg, temperature=0.7, max_tokens=500)
        return EnhanceResponse(enhanced_prompt=enhanced)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Enhance failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-script", response_model=ScriptResponse)
async def generate_script(request: GenerateScriptRequest):
    """Generate a scene-by-scene cinematic script from prompt + options via Groq."""
    logger.info("Generating script (%d chars)", len(request.prompt))
    try:
        ctx = build_options_context(request.options, label="Video Specifications")
        user_msg = f"Create a cinematic video script for:\n\nPrompt: {request.prompt}\n{ctx}"
        script = groq_chat(SCRIPT_SYSTEM_PROMPT, user_msg, temperature=0.8, max_tokens=2000)
        return ScriptResponse(script=script)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Script generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Leonardo Video Generation ────────────────────────────────────────────────
async def leonardo_create_video(prompt: str, size: str, duration_option: str = "") -> str:
    """Create a text-to-video job on Leonardo; returns generationId for polling.
    If duration_option is set (e.g. '30 seconds'), uses VEO3FAST with 4/6/8s; otherwise MOTION2 (~5s).
    """
    client = get_http_client()
    duration_sec = LEONARDO_DURATION_SECONDS.get(duration_option) if duration_option else None
    use_veo = duration_sec is not None  # VEO3FAST supports 4/6/8s; MOTION2 is fixed ~5s

    if use_veo:
        # VEO3FAST: explicit duration 4/6/8 sec; output is fixed 1280x720 per Leonardo API
        width, height = 1280, 720
        payload = {
            "prompt": prompt,
            "model": "VEO3FAST",
            "duration": duration_sec,
            "width": width,
            "height": height,
            "resolution": "RESOLUTION_720",
            "frameInterpolation": True,
            "isPublic": False,
            "promptEnhance": True,
        }
        logger.info("Leonardo create (VEO3FAST, %ds) — %dx%d — %.80s…", duration_sec, width, height, prompt)
    else:
        # MOTION2: no duration param; output ~5s; use size presets for dimensions
        width, height = LEONARDO_DIMENSIONS.get(size, (832, 480))
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "resolution": "RESOLUTION_720",
            "frameInterpolation": True,
            "isPublic": False,
            "promptEnhance": True,
        }
        logger.info("Leonardo create (MOTION2, ~5s) — %dx%d — %.80s…", width, height, prompt)
    resp = await client.post(
        f"{LEONARDO_API_BASE}/generations-text-to-video",
        headers={
            "Authorization": f"Bearer {LEONARDO_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
    )
    if resp.status_code not in (200, 201):
        detail = resp.text
        try:
            body = resp.json()
            detail = body.get("error", body.get("detail", resp.text))
        except Exception:
            pass
        logger.error("Leonardo create failed (%d): %s", resp.status_code, detail)
        raise HTTPException(status_code=resp.status_code, detail=f"Leonardo error: {detail}")

    data = resp.json()

    # Some API versions wrap the payload in a "data" key
    if "data" in data and isinstance(data["data"], dict) and len(data) == 1:
        data = data["data"]

    # Leonardo can return generationId in different shapes; support all for compatibility
    generation_id = None
    job = data.get("motionVideoGenerationJob") or data.get("motion_video_generation_job")
    if job and isinstance(job, dict):
        generation_id = job.get("generationId") or job.get("generation_id")
    if not generation_id and (data.get("sdGenerationJob") or data.get("sd_generation_job")):
        job = data.get("sdGenerationJob") or data.get("sd_generation_job")
        if isinstance(job, dict):
            generation_id = job.get("generationId") or job.get("generation_id")
    if not generation_id:
        generation_id = data.get("generationId") or data.get("generation_id")
    if not generation_id and (data.get("textToVideoGeneration") or data.get("text_to_video_generation")):
        job = data.get("textToVideoGeneration") or data.get("text_to_video_generation")
        if isinstance(job, dict):
            generation_id = job.get("generationId") or job.get("generation_id")

    if not generation_id:
        # No known key contained generationId — return helpful error with API message if any
        err_msg = data.get("error") or data.get("message") or data.get("detail")
        keys_preview = list(data.keys())
        logger.error("Leonardo response missing generationId. Keys: %s. Response: %s", keys_preview, data)
        detail = f"Leonardo did not return a generationId. Response keys: {keys_preview}"
        if err_msg:
            detail += f". API message: {err_msg}"
        raise HTTPException(status_code=500, detail=detail)
    
    logger.info("Leonardo generationId: %s", generation_id)
    return generation_id


async def leonardo_poll_generation(generation_id: str) -> str:
    """Poll GET /generations/{id} until status is COMPLETE or FAILED; return video URL on success."""
    logger.info("Polling Leonardo generation %s (max %ds)", generation_id, VIDEO_MAX_WAIT)
    client = get_http_client()
    elapsed = 0
    while elapsed < VIDEO_MAX_WAIT:
        resp = await client.get(
            f"{LEONARDO_API_BASE}/generations/{generation_id}",
            headers={
                "Authorization": f"Bearer {LEONARDO_API_KEY}",
                "Accept": "application/json",
            },
        )
        if resp.status_code != 200:
            logger.error("Leonardo poll error %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=500, detail=f"Failed to check Leonardo generation: {resp.status_code}")

        data = resp.json()
        generation = data.get("generations_by_pk", {})
        status = generation.get("status")
        logger.info("Leonardo %s — %s (%ds)", generation_id, status, elapsed)

        if status == "COMPLETE":
            # Prefer motion MP4 URL when present (video); fall back to static image URL
            videos = generation.get("generated_images", [])
            if videos:
                video_url = videos[0].get("url", "")
                motion_url = videos[0].get("motionMP4URL", "")
                final_url = motion_url or video_url
                if final_url:
                    return final_url
            raise HTTPException(status_code=500, detail="Leonardo generation completed but no video URL found")

        if status == "FAILED":
            logger.error("Leonardo generation failed: %s", generation)
            raise HTTPException(status_code=500, detail="Leonardo video generation failed")

        await asyncio.sleep(VIDEO_POLL_INTERVAL)
        elapsed += VIDEO_POLL_INTERVAL

    raise HTTPException(status_code=504, detail=f"Leonardo video generation timed out ({VIDEO_MAX_WAIT}s)")


# ── Replicate Video Generation ───────────────────────────────────────────────
async def replicate_create_prediction(prompt: str) -> dict:
    """Start a Minimax video-01 prediction on Replicate; returns prediction object with id for polling."""
    client = get_http_client()
    resp = await client.post(
        f"{REPLICATE_API_BASE}/models/minimax/video-01/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json={"input": {"prompt": prompt}},
    )
    if resp.status_code not in (200, 201):
        detail = resp.text
        try:
            body = resp.json()
            detail = body.get("detail", body.get("error", resp.text))
        except Exception:
            pass
        logger.error("Replicate create failed (%d): %s", resp.status_code, detail)
        raise HTTPException(status_code=resp.status_code, detail=f"Replicate error: {detail}")
    return resp.json()


async def replicate_poll_prediction(prediction_id: str) -> str:
    """Poll Replicate prediction until succeeded/failed/canceled; return output video URL."""
    logger.info("Polling prediction %s (max %ds)", prediction_id, VIDEO_MAX_WAIT)
    client = get_http_client()
    elapsed = 0
    while elapsed < VIDEO_MAX_WAIT:
        resp = await client.get(
            f"{REPLICATE_API_BASE}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
        )
        if resp.status_code != 200:
            logger.error("Poll error %d: %s", resp.status_code, resp.text)
            raise HTTPException(status_code=500, detail=f"Failed to check prediction: {resp.status_code}")

        data = resp.json()
        status = data.get("status")
        logger.info("Prediction %s — %s (%ds)", prediction_id, status, elapsed)

        if status == "succeeded":
            return extract_video_url(data.get("output"))
        if status in ("failed", "canceled"):
            err = data.get("error", "Unknown error")
            logger.error("Prediction %s: %s", status, err)
            raise HTTPException(status_code=500, detail=f"Video generation {status}: {err}")

        await asyncio.sleep(VIDEO_POLL_INTERVAL)
        elapsed += VIDEO_POLL_INTERVAL

    raise HTTPException(status_code=504, detail=f"Video generation timed out ({VIDEO_MAX_WAIT}s)")


@app.post("/api/generate-video", response_model=VideoResponse)
async def generate_video(request: GenerateVideoRequest):
    """Generate video from prompt (or script) via Leonardo or Replicate. Leonardo: long prompts condensed with Groq; duration option → 4/6/8s with VEO3FAST."""
    provider = request.provider
    video_prompt = request.prompt.strip()
    if request.options.category:
        video_prompt = f"{request.options.category} style: {video_prompt}"
    logger.info("Generating video [%s] — prompt: %.80s…", provider, video_prompt)

    if provider == "leonardo":
        if not LEONARDO_API_KEY:
            raise HTTPException(status_code=500, detail="LEONARDO_API_KEY not configured. Add it to backend/.env")
        if not video_prompt:
            raise HTTPException(status_code=400, detail="Prompt is required for video generation")
        # Leonardo allows max 1500 chars: condense long scripts with Groq, else truncate
        if len(video_prompt) > LEONARDO_MAX_PROMPT_LENGTH:
            if GROQ_API_KEY:
                logger.info("Condensing long script (%d chars) into video prompt for Leonardo", len(video_prompt))
                try:
                    video_prompt = condense_for_video_prompt(video_prompt, LEONARDO_MAX_PROMPT_LENGTH)
                    logger.info("Condensed to %d chars", len(video_prompt))
                except Exception as e:
                    logger.warning("Condense failed, truncating: %s", e)
                    video_prompt = (video_prompt[: LEONARDO_MAX_PROMPT_LENGTH].rsplit(" ", 1)[0] or video_prompt[: LEONARDO_MAX_PROMPT_LENGTH])
            else:
                video_prompt = video_prompt[: LEONARDO_MAX_PROMPT_LENGTH].rsplit(" ", 1)[0] or video_prompt[: LEONARDO_MAX_PROMPT_LENGTH]
                logger.info("Prompt truncated to %d chars for Leonardo (no Groq for condensing)", len(video_prompt))
        try:
            generation_id = await leonardo_create_video(video_prompt, request.options.size, request.options.duration)
            video_url = await leonardo_poll_generation(generation_id)
            return VideoResponse(video_url=video_url, status="completed", provider="leonardo")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Leonardo video generation failed")
            raise HTTPException(status_code=500, detail=f"Leonardo video generation failed: {str(e)}")

    elif provider == "replicate":
        if not REPLICATE_API_TOKEN:
            raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not configured. Add it to backend/.env")
        try:
            prediction = await replicate_create_prediction(video_prompt)
            if prediction.get("status") == "succeeded":
                return VideoResponse(video_url=extract_video_url(prediction.get("output")), status="completed", provider="replicate")
            prediction_id = prediction.get("id")
            if not prediction_id:
                raise HTTPException(status_code=500, detail="No prediction ID returned")
            video_url = await replicate_poll_prediction(prediction_id)
            return VideoResponse(video_url=video_url, status="completed", provider="replicate")
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Replicate video generation failed")
            raise HTTPException(status_code=500, detail=f"Replicate video generation failed: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


@app.get("/api/health")
async def health():
    """Health check; reports which API keys are configured (no secrets)."""
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "replicate_configured": bool(REPLICATE_API_TOKEN),
        "leonardo_configured": bool(LEONARDO_API_KEY),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
