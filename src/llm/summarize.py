import os
import json
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # voice2notes/
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-2.5-flash-001"  

SYSTEM_PROMPT = (
    "You are an internal corporate meeting assistant.\n"
    "The transcript may include Arabic dialects, mixed Arabic+English, and ASR noise.\n"
    "The transcript lines are time-coded like: [MM:SS-MM:SS] text.\n\n"
    "Your job:\n"
    "1) Understand the meaning even if slang/mixed language.\n"
    "2) Extract meeting notes (summary, key points, decisions, risks, actions, highlights).\n"
    "3) For EVERY extracted item, attach evidence_timestamp using the closest matching time-coded line.\n\n"
    "Evidence rules:\n"
    "- Use the START time from the chosen line: [MM:SS-..] => evidence_timestamp = MM:SS.\n"
    "- If you cannot find supporting evidence, set evidence_timestamp to null.\n"
    "- Do NOT invent facts.\n"
    "- If owner is not explicit, set owner = null.\n"
    "- If due date is not explicit, set due_date = null.\n\n"
    "Output rules:\n"
    "- Return ONLY valid JSON.\n"
    "- No markdown, no comments, no extra keys, no explanations.\n"
)

def _build_prompt(transcript: str) -> str:
    return f"""
{SYSTEM_PROMPT}

Return JSON with EXACTLY this schema:
{{
  "short_summary": "string",
  "key_points": ["string"],
  "action_items": [
    {{
      "task": "string",
      "owner": "string or null",
      "due_date": "string or null",
      "evidence_timestamp": "MM:SS or null"
    }}
  ],
  "decisions": ["string"],
  "risks_or_issues": ["string"],
  "highlights": [
    {{
      "text": "string",
      "timestamp": "MM:SS"
    }}
  ]
}}

Transcript:
{transcript}
""".strip()

def summarize(transcript_text: str, model_name: str | None = None) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

    client = genai.Client(api_key=api_key)
    model_to_use = model_name or MODEL_NAME

    prompt = _build_prompt(transcript_text)

    resp = client.models.generate_content(
        model=model_to_use,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    raw = (resp.text or "").strip()
    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last == -1:
        raise ValueError(f"Gemini did not return JSON. Raw output:\n{raw[:500]}")

    return json.loads(raw[first:last + 1])