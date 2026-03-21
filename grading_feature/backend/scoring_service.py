import asyncio
import json
import os
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime based on env
    OpenAI = None


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MOCK_SCORER = os.getenv("MOCK_SCORER", "0") == "1"


def round_num(value: float, digits: int = 2) -> float:
    if not isinstance(value, (int, float)):
        return 0
    return round(float(value), digits)


def safe_number(value: Any, fallback: float = 0) -> float:
    try:
        number = float(value)
        if number != number:  # NaN guard
            return fallback
        return number
    except (TypeError, ValueError):
        return fallback


def decorate_score(question: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    parsed = {
        "relevance": safe_number(raw.get("relevance")),
        "politeness": safe_number(raw.get("politeness")),
        "on_topic": safe_number(raw.get("on_topic")),
        "neutrality": safe_number(raw.get("neutrality")),
        "non_imperative": safe_number(raw.get("non_imperative")),
        "clarity_optional": safe_number(raw.get("clarity_optional")),
        "privacy_minimization_optional": safe_number(raw.get("privacy_minimization_optional")),
    }

    score_total = (
        parsed["relevance"]
        + parsed["politeness"]
        + parsed["on_topic"]
        + parsed["neutrality"]
        + parsed["non_imperative"]
        + parsed["clarity_optional"]
        + parsed["privacy_minimization_optional"]
    )

    verdict = "ok"
    if score_total >= 11:
        verdict = "good"
    elif score_total <= 5:
        verdict = "needs_work"

    return {
        "id": question.get("id"),
        "question": question.get("text", ""),
        "studentId": question.get("studentId", "unknown"),
        "studentName": question.get("studentName", "Unknown"),
        "score_total": round_num(score_total),
        "verdict": verdict,
        **parsed,
        "notes": raw.get("notes", ""),
    }


def mock_score(question: dict[str, Any]) -> dict[str, Any]:
    text = (question.get("text") or "").strip()
    text_len = len(text)
    base = 2 if text_len > 80 else 1.5 if text_len > 40 else 1

    raw = {
        "relevance": base,
        "politeness": base,
        "on_topic": base,
        "neutrality": 1,
        "non_imperative": 1,
        "clarity_optional": 1.5 if text_len > 100 else 1,
        "privacy_minimization_optional": 0 if any(k in text.lower() for k in ["password", "account", "id", "ssn", "social"]) else 1,
        "notes": "mock score",
    }
    return decorate_score(question, raw)


def _openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Add 'openai' to requirements.txt")
    return OpenAI(api_key=OPENAI_API_KEY or None, base_url=OPENAI_BASE_URL)


def score_with_openai(question: dict[str, Any]) -> dict[str, Any]:
    client = _openai_client()
    system_prompt = (
        "You are a careful rater for student questions in a financial-planning interview practice. "
        "Given ONE question from the student, score each dimension from 0 to 2 (0=poor,1=mixed,2=good). "
        "Return strict JSON only with keys: relevance, politeness, on_topic, neutrality, non_imperative, "
        "clarity_optional, privacy_minimization_optional, notes."
    )

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.get("text", "")},
        ],
    )

    payload = completion.choices[0].message.content
    parsed = json.loads(payload)
    return decorate_score(question, parsed)


def compute_abi_for_question(scored: dict[str, Any]) -> dict[str, Any]:
    rel = safe_number(scored.get("relevance"), 0)
    on_topic = safe_number(scored.get("on_topic"), 0)
    clarity = safe_number(scored.get("clarity_optional"), 0)
    polite = safe_number(scored.get("politeness"), 0)
    neutral = safe_number(scored.get("neutrality"), 0)
    non_imp = safe_number(scored.get("non_imperative"), 0)
    privacy = safe_number(scored.get("privacy_minimization_optional"), 0)

    def norm(v: float) -> float:
        return v / 2

    subs = {
        "knowledge_consistency": norm(rel),
        "professional_tone": norm(clarity),
        "rationality": norm(neutral),
        "calibrated_confidence": 0.6,
        "politeness": norm(polite),
        "human_care": norm(polite),
        "care_my_interest": norm(on_topic),
        "shared_interest": 0.5,
        "legality": norm(privacy),
        "morality": norm(neutral),
        "contract": norm(on_topic),
        "inducement": 1 - norm(non_imp),
    }

    ability = (
        0.35 * subs["knowledge_consistency"]
        + 0.25 * subs["professional_tone"]
        + 0.2 * subs["rationality"]
        + 0.2 * subs["calibrated_confidence"]
    )

    benevolence = (
        0.25 * subs["politeness"]
        + 0.3 * subs["human_care"]
        + 0.25 * subs["care_my_interest"]
        + 0.2 * subs["shared_interest"]
    )

    integrity = (
        0.3 * subs["legality"]
        + 0.25 * subs["morality"]
        + 0.3 * subs["contract"]
        + 0.15 * (1 - subs["inducement"])
    )

    return {
        "ability": round_num(ability),
        "benevolence": round_num(benevolence),
        "integrity": round_num(integrity),
        "abi_total": round_num((ability + benevolence + integrity) / 3),
        "subs": {k: round_num(v) for k, v in subs.items()},
    }


def aggregate_abi(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None

    count = len(entries)
    ability = sum(safe_number(x.get("ability")) for x in entries) / count
    benevolence = sum(safe_number(x.get("benevolence")) for x in entries) / count
    integrity = sum(safe_number(x.get("integrity")) for x in entries) / count

    sub_keys = list(entries[0].get("subs", {}).keys())
    subs = {}
    for key in sub_keys:
        subs[key] = round_num(sum(safe_number(x.get("subs", {}).get(key)) for x in entries) / count)

    return {
        "ability": round_num(ability),
        "benevolence": round_num(benevolence),
        "integrity": round_num(integrity),
        "abi_total": round_num((ability + benevolence + integrity) / 3),
        "subs": subs,
    }


def build_aggregate(results: list[dict[str, Any]], abi_enabled: bool) -> dict[str, Any]:
    count = len(results)

    def avg_of(key: str) -> float:
        return sum(safe_number(r.get(key)) for r in results) / count if count else 0

    avg_total = avg_of("score_total")
    overall_100 = (avg_total / 14) * 100

    dims_list = [
        "relevance",
        "politeness",
        "on_topic",
        "neutrality",
        "non_imperative",
        "clarity_optional",
        "privacy_minimization_optional",
    ]
    dims = {d: round_num(avg_of(d)) for d in dims_list}

    bins = [0, 0, 0, 0]
    for item in results:
        total = safe_number(item.get("score_total"))
        if total <= 3:
            bins[0] += 1
        elif total <= 6:
            bins[1] += 1
        elif total <= 10:
            bins[2] += 1
        else:
            bins[3] += 1

    habits = []
    if dims["relevance"] < 1.3:
        habits.append("Stay closer to the client scenario and be more specific.")
    if dims["politeness"] < 1.3:
        habits.append("Use more polite, tentative phrasing instead of direct blame.")
    if dims["on_topic"] < 1.3:
        habits.append("Keep questions focused on finances and planning, not side topics.")
    if dims["privacy_minimization_optional"] < 1:
        habits.append("Avoid asking for detailed IDs, passwords, or account numbers unless strictly necessary.")
    if not habits:
        habits.append("Good habits overall - keep asking clear, polite, and focused questions!")

    per_student: dict[str, dict[str, Any]] = {}
    for row in results:
        sid = row.get("studentId") or "unknown"
        if sid not in per_student:
            per_student[sid] = {
                "studentId": sid,
                "studentName": row.get("studentName") or sid,
                "count": 0,
                "sum_total": 0,
                "dims_sum": {d: 0 for d in dims_list},
                "abi_list": [],
            }

        target = per_student[sid]
        target["count"] += 1
        target["sum_total"] += safe_number(row.get("score_total"))
        for d in dims_list:
            target["dims_sum"][d] += safe_number(row.get(d))
        if abi_enabled and row.get("abi"):
            target["abi_list"].append(row["abi"])

    per_student_out: dict[str, Any] = {}
    for sid, stu in per_student.items():
        c = stu["count"]
        dims_avg = {d: round_num(stu["dims_sum"][d] / c if c else 0) for d in dims_list}
        avg_score = stu["sum_total"] / c if c else 0
        per_student_out[sid] = {
            "studentId": sid,
            "studentName": stu["studentName"],
            "count": c,
            "avg_total_0_14": round_num(avg_score),
            "overall_0_100": round_num((avg_score / 14) * 100),
            "dims": dims_avg,
            "abi_avg": aggregate_abi(stu["abi_list"]) if abi_enabled else None,
        }

    abi_all = aggregate_abi([r["abi"] for r in results if r.get("abi")]) if abi_enabled else None

    return {
        "count": count,
        "avg_total_0_14": round_num(avg_total),
        "overall_0_100": round_num(overall_100),
        "dims": dims,
        "distribution": {
            "labels": ["0-3", "4-6", "7-10", "11-14"],
            "counts": bins,
        },
        "habits": habits,
        "perStudent": per_student_out,
        "abi_global": abi_all,
    }


async def _score_one(question: dict[str, Any]) -> dict[str, Any]:
    if MOCK_SCORER or not OPENAI_API_KEY:
        return mock_score(question)
    return await asyncio.to_thread(score_with_openai, question)


async def score_questions(questions: list[dict[str, Any]], use_abi: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = []
    for idx, item in enumerate(questions):
        if isinstance(item, str):
            normalized.append(
                {
                    "id": idx,
                    "text": item,
                    "studentId": "unknown",
                    "studentName": "Unknown",
                }
            )
            continue

        normalized.append(
            {
                "id": item.get("id", idx),
                "text": str(item.get("text", "")),
                "studentId": item.get("studentId") or "unknown",
                "studentName": item.get("studentName") or item.get("studentId") or "Unknown",
            }
        )

    scored = await asyncio.gather(*[_score_one(q) for q in normalized])
    if use_abi:
        scored = [{**row, "abi": compute_abi_for_question(row)} for row in scored]

    aggregate = build_aggregate(scored, use_abi)
    return scored, aggregate