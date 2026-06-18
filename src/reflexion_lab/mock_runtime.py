from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

@dataclass
class RuntimeAnswer:
    text: str
    token_estimate: int
    latency_ms: int


def _runtime_mode() -> str:
    return os.getenv("REFLEXION_RUNTIME_MODE", "mock").strip().lower()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) + max(1, len(text) // 4))


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)


@lru_cache(maxsize=1)
def _gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def _call_gemini(prompt: str, text_payload: str) -> tuple[str, int, int]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini runtime mode")
    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{_gemini_model_name()}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": f"{prompt.strip()}\n\n{text_payload}"}]}],
        "generationConfig": {"temperature": 0.0},
    }
    start = time.perf_counter()
    response = requests.post(url, params={"key": api_key}, json=payload, timeout=60)
    latency_ms = int((time.perf_counter() - start) * 1000)
    response.raise_for_status()
    data = response.json()
    text = "".join(part.get("text", "") for candidate in data.get("candidates", []) for part in candidate.get("content", {}).get("parts", []))
    usage = data.get("usageMetadata", {})
    token_estimate = int(usage.get("totalTokenCount") or 0) or _estimate_tokens(text)
    return text.strip(), token_estimate, latency_ms


def _parse_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _gemini_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> RuntimeAnswer:
    context = "\n".join(f"[{chunk.title}] {chunk.text}" for chunk in example.context)
    memory = "\n".join(f"- {item}" for item in reflection_memory) or "- None"
    payload = f"QUESTION:\n{example.question}\n\nCONTEXT:\n{context}\n\nREFLECTION_MEMORY:\n{memory}\n\nATTEMPT_ID: {attempt_id}\nAGENT_TYPE: {agent_type}"
    text, tokens, latency = _call_gemini(ACTOR_SYSTEM, payload)
    return RuntimeAnswer(text=text, token_estimate=tokens, latency_ms=latency)


def _gemini_evaluator(example: QAExample, answer: str) -> JudgeResult:
    payload = json.dumps({
        "question": example.question,
        "context": [{"title": chunk.title, "text": chunk.text} for chunk in example.context],
        "gold_answer": example.gold_answer,
        "predicted_answer": answer,
    }, ensure_ascii=False, indent=2)
    raw_text, _, _ = _call_gemini(EVALUATOR_SYSTEM, payload)
    data = _parse_json_response(raw_text)
    return JudgeResult.model_validate(data)


def _gemini_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    payload = json.dumps({
        "attempt_id": attempt_id,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "failure_reason": judge.reason,
        "missing_evidence": judge.missing_evidence,
        "spurious_claims": judge.spurious_claims,
    }, ensure_ascii=False, indent=2)
    raw_text, _, _ = _call_gemini(REFLECTOR_SYSTEM, payload)
    data = _parse_json_response(raw_text)
    if "attempt_id" not in data:
        data["attempt_id"] = attempt_id
    return ReflectionEntry.model_validate(data)


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> RuntimeAnswer:
    if _runtime_mode() == "gemini":
        return _gemini_actor_answer(example, attempt_id, agent_type, reflection_memory)
    text = _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)
    return RuntimeAnswer(text=text, token_estimate=_estimate_tokens(text), latency_ms=8)


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if _runtime_mode() == "gemini":
        try:
            return _gemini_evaluator(example, answer)
        except Exception:
            pass
    return _mock_evaluator(example, answer)


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    if _runtime_mode() == "gemini":
        try:
            return _gemini_reflector(example, attempt_id, judge)
        except Exception:
            pass
    return _mock_reflector(example, attempt_id, judge)
