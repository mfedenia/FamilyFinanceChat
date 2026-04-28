"""
title: ABI Trust Pipe
author: tony-xu
version: 0.4
"""

from __future__ import annotations

import os, json, math, statistics
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from pydantic import BaseModel
import requests

# ----------------------------- Enums / Config ----------------------------- #


class Stage(Enum):
    CONTRACT = "contract"
    KNOWLEDGE = "knowledge"
    EMPATHY = "empathy"


def _lvl(x: str) -> float:
    return {"低": 0.20, "中": 0.50, "高": 1.00}[x]


STAGE_SENS = {
    Stage.CONTRACT: (_lvl("中"), _lvl("低"), _lvl("高")),
    Stage.KNOWLEDGE: (_lvl("中"), _lvl("中"), _lvl("低")),
    Stage.EMPATHY: (_lvl("低"), _lvl("高"), _lvl("低")),
}

THRESH = {"I_high": 0.70, "B_high": 0.70, "A_conv_std": 0.05}
WINDOW = 5

# ----------------------------- Atomic Scores ----------------------------- #


@dataclass
class AbilitySubs:
    knowledge_consistency: float
    professional_tone: float
    rationality: float
    calibrated_confidence: float

    def to_score(self, w=(0.35, 0.25, 0.20, 0.20)) -> float:
        a = (
            self.knowledge_consistency * w[0]
            + self.professional_tone * w[1]
            + self.rationality * w[2]
            + self.calibrated_confidence * w[3]
        )
        return max(0.0, min(1.0, a))


@dataclass
class BenevolenceSubs:
    politeness: float
    human_care: float
    care_my_interest: float
    shared_interest: float

    def to_score(self, w=(0.25, 0.30, 0.25, 0.20)) -> float:
        b = (
            self.politeness * w[0]
            + self.human_care * w[1]
            + self.care_my_interest * w[2]
            + self.shared_interest * w[3]
        )
        return max(0.0, min(1.0, b))


@dataclass
class IntegritySubs:
    legality: float
    morality: float
    contract: float
    inducement: float

    def to_score(self, w=(0.30, 0.25, 0.30, 0.15)) -> float:
        i = (
            self.legality * w[0]
            + self.morality * w[1]
            + self.contract * w[2]
            + (1.0 - self.inducement) * w[3]
        )
        return max(0.0, min(1.0, i))


@dataclass
class UtteranceABISubs:
    ability: AbilitySubs
    benevolence: BenevolenceSubs
    integrity: IntegritySubs

    def to_abi(self):
        return (
            self.ability.to_score(),
            self.benevolence.to_score(),
            self.integrity.to_score(),
        )


# ----------------------------- State & Update ----------------------------- #


@dataclass
class TrustState:
    A: float = 0.50
    B: float = 0.50
    I: float = 0.80
    stage: Stage = Stage.CONTRACT
    history_A: List[float] = field(default_factory=list)
    history_B: List[float] = field(default_factory=list)
    history_I: List[float] = field(default_factory=list)

    def stage_weights(self):
        return STAGE_SENS[self.stage]

    def trust_weighted(self) -> float:
        wa, wb, wi = self.stage_weights()
        s = wa + wb + wi
        return (wa * self.A + wb * self.B + wi * self.I) / (s if s > 0 else 1.0)


@dataclass
class UpdateResult:
    A_local: float
    B_local: float
    I_local: float
    dA: float
    dB: float
    dI: float
    stage_before: Stage
    stage_after: Stage
    A_after: float
    B_after: float
    I_after: float
    notes: Dict[str, str]


class ABIEngine:
    def __init__(
        self, gamma: float = 1.0, clip_min: float = 0.0, clip_max: float = 1.0
    ):
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max

    @staticmethod
    def _clip(x, lo, hi):
        return min(hi, max(lo, x))

    def update(self, state: TrustState, subs: UtteranceABISubs) -> UpdateResult:
        A_loc, B_loc, I_loc = subs.to_abi()
        wa, wb, wi = state.stage_weights()

        # 原始增量
        dA = wa * (A_loc - 0.5) * (state.A**self.gamma)
        dB = wb * (B_loc - 0.5) * (state.B**self.gamma)
        dI = wi * (I_loc - 0.5) * (state.I**self.gamma)

        # 如果当前已经在上下界，就视为没有变化（尤其是上限时不再继续正向累加）
        def freeze_if_capped(cur: float, delta: float) -> float:
            eps = 1e-6
            if delta > 0 and cur >= self.clip_max - eps:
                return 0.0
            if delta < 0 and cur <= self.clip_min + eps:
                return 0.0
            return delta

        dA = freeze_if_capped(state.A, dA)
        dB = freeze_if_capped(state.B, dB)
        dI = freeze_if_capped(state.I, dI)

        A_new = self._clip(state.A + dA, self.clip_min, self.clip_max)
        B_new = self._clip(state.B + dB, self.clip_min, self.clip_max)
        I_new = self._clip(state.I + dI, self.clip_min, self.clip_max)

        stage_before = state.stage
        notes: Dict[str, str] = {}

        state.history_A.append(dA)
        state.history_B.append(dB)
        state.history_I.append(dI)
        if len(state.history_A) > WINDOW:
            state.history_A.pop(0)
        if len(state.history_B) > WINDOW:
            state.history_B.pop(0)
        if len(state.history_I) > WINDOW:
            state.history_I.pop(0)

        stage_after = stage_before

        if stage_before == Stage.CONTRACT:
            A_std = (
                statistics.pstdev(state.history_A) if len(state.history_A) >= 2 else 1.0
            )
            if I_new >= THRESH["I_high"] and A_std < THRESH["A_conv_std"]:
                stage_after = Stage.KNOWLEDGE
                notes["stage_transition"] = "CONTRACT→KNOWLEDGE"
        elif stage_before == Stage.KNOWLEDGE:
            if B_new >= THRESH["B_high"] and I_new >= 0.55:
                stage_after = Stage.EMPATHY
                notes["stage_transition"] = "KNOWLEDGE→EMPATHY"

        state.A, state.B, state.I = A_new, B_new, I_new
        state.stage = stage_after

        return UpdateResult(
            A_loc,
            B_loc,
            I_loc,
            dA,
            dB,
            dI,
            stage_before,
            stage_after,
            A_new,
            B_new,
            I_new,
            notes,
        )


# ----------------------------- Gating ----------------------------- #


@dataclass
class GateInputs:
    is_private: int
    is_polite: int
    has_preamble: int
    trust_weighted: float


@dataclass
class GateResult:
    p_answer: float
    should_answer: bool
    rationale: Dict[str, float]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Gate:
    def __init__(
        self, w_trust=2.2, w_privacy=0.70, w_polite=0.30, w_preamble=0.30, bias=2.0
    ):
        self.w_trust = w_trust
        self.w_privacy = w_privacy
        self.w_polite = w_polite
        self.w_preamble = w_preamble
        self.bias = bias

    def decide(self, x: GateInputs, threshold: float = 0.5) -> GateResult:
        score = (
            self.w_trust * x.trust_weighted
            + self.w_privacy * (1 - x.is_private)
            + self.w_polite * x.is_polite
            + self.w_preamble * x.has_preamble
            - self.bias
        )
        p = sigmoid(score)
        return GateResult(
            p,
            p >= threshold,
            dict(
                trust=x.trust_weighted,
                non_private=(1 - x.is_private),
                polite=x.is_polite,
                preamble=x.has_preamble,
                linear_score=score,
            ),
        )


# ----------------------------- API Scorer ----------------------------- #


class APIScorer:
    SYSTEM_PROMPT = (
        "You are a careful rater. Given ONE user utterance, score the following sub-dimensions in [0,1]. "
        "Return STRICT JSON with keys: ability, benevolence, integrity; each contains its sub-keys.\n"
        "Scoring rubric:\n"
        "- ability.knowledge_consistency: 1 if consistent with known facts, 0 if contradicts.\n"
        "- ability.professional_tone: 1 if simple and clear for laypeople; penalize excessive jargon or slang.\n"
        "- ability.rationality: 1 if neutral and reasoned; penalize partisan or emotional tone.\n"
        "- ability.calibrated_confidence: confident but not arrogant.\n"
        "- benevolence.politeness; benevolence.human_care; benevolence.care_my_interest; benevolence.shared_interest.\n"
        "- integrity.legality; integrity.morality; integrity.contract; integrity.inducement (higher if persuasive/inducing).\n"
        "Only output JSON."
    )

    @staticmethod
    def default_user_prompt(text: str) -> str:
        return f"Utterance:\n{text}\nReturn only JSON."

    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai

    def score(self, text: str, mock: bool = False) -> UtteranceABISubs:
        if mock or not (os.getenv("OPENAI_API_KEY") or os.getenv("SCORE_API_URL")):
            return self._mock_scores(text)

        if self.use_openai and os.getenv("OPENAI_API_KEY"):
            return self._score_openai(text)
        else:
            return self._score_generic_endpoint(text)

    def _score_openai(self, text: str) -> UtteranceABISubs:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.default_user_prompt(text)},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            return self._parse_abi_json(data)
        except Exception:
            return self._mock_scores(text)

    def _score_generic_endpoint(self, text: str) -> UtteranceABISubs:
        url = os.getenv("SCORE_API_URL")
        key = os.getenv("SCORE_API_KEY", "")
        payload = {
            "system_prompt": self.SYSTEM_PROMPT,
            "user_prompt": self.default_user_prompt(text),
        }
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            return self._parse_abi_json(data)
        except Exception:
            return self._mock_scores(text)

    @staticmethod
    def _parse_abi_json(data: dict) -> UtteranceABISubs:
        a = data.get("ability", {})
        b = data.get("benevolence", {})
        i = data.get("integrity", {})
        return UtteranceABISubs(
            ability=AbilitySubs(
                knowledge_consistency=float(a.get("knowledge_consistency", 0.5)),
                professional_tone=float(a.get("professional_tone", 0.5)),
                rationality=float(a.get("rationality", 0.5)),
                calibrated_confidence=float(a.get("calibrated_confidence", 0.5)),
            ),
            benevolence=BenevolenceSubs(
                politeness=float(b.get("politeness", 0.5)),
                human_care=float(b.get("human_care", 0.5)),
                care_my_interest=float(b.get("care_my_interest", 0.5)),
                shared_interest=float(b.get("shared_interest", 0.5)),
            ),
            integrity=IntegritySubs(
                legality=float(i.get("legality", 0.5)),
                morality=float(i.get("morality", 0.5)),
                contract=float(i.get("contract", 0.5)),
                inducement=float(i.get("inducement", 0.5)),
            ),
        )

    @staticmethod
    def _mock_scores(text: str) -> UtteranceABISubs:
        base = 0.7 if len(text) > 20 else 0.55
        return UtteranceABISubs(
            ability=AbilitySubs(
                knowledge_consistency=base,
                professional_tone=0.75,
                rationality=0.72,
                calibrated_confidence=0.70,
            ),
            benevolence=BenevolenceSubs(
                politeness=0.78,
                human_care=0.70,
                care_my_interest=0.68,
                shared_interest=0.62,
            ),
            integrity=IntegritySubs(
                legality=0.92,
                morality=0.85,
                contract=0.88,
                inducement=0.12,
            ),
        )


# ----------------------------- 高层封装 function ----------------------------- #

GLOBAL_TRUST_STATES: Dict[str, TrustState] = {}


def get_session_id(session_meta) -> str:
    if isinstance(session_meta, dict):
        return str(
            session_meta.get("id")
            or session_meta.get("session_id")
            or session_meta.get("uuid")
            or "default"
        )
    return "default"


def get_trust_state(session_id: str) -> TrustState:
    st = GLOBAL_TRUST_STATES.get(session_id)
    if st is None:
        st = TrustState()
        GLOBAL_TRUST_STATES[session_id] = st
    return st


def abi_update_for_utterance(
    session_id: str, text: str, scorer: APIScorer, engine: ABIEngine, mock: bool
):
    state = get_trust_state(session_id)
    subs = scorer.score(text, mock=mock)
    result = engine.update(state, subs)
    GLOBAL_TRUST_STATES[session_id] = state
    return state, result


def gate_for_session(
    session_id: str,
    gate: Gate,
    is_private: int,
    is_polite: int,
    has_preamble: int,
    threshold: float,
):
    state = get_trust_state(session_id)
    gi = GateInputs(
        is_private=is_private,
        is_polite=is_polite,
        has_preamble=has_preamble,
        trust_weighted=state.trust_weighted(),
    )
    gr = gate.decide(gi, threshold=threshold)
    return state, gr


def make_response(text: str) -> dict:
    return {
        "id": "abi-pipe",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def call_base_model(messages: List[Dict], model: Optional[str] = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model_name, "messages": messages}

    r = requests.post(
        f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ----------------------------- Pipe 实现 ----------------------------- #


class BasePipe:
    class Valves(BaseModel):
        use_mock_scores: bool = False
        gate_threshold: float = 0.5
        assume_private: int = 1
        assume_polite: int = 1
        assume_preamble: int = 1
        base_model: Optional[str] = None

    def __init__(self):
        self.valves = self.Valves()
        self.scorer = APIScorer(use_openai=True)
        self.engine = ABIEngine()
        self.gate = Gate()

    def pipe(self, body: dict, __user__=None, __session__=None) -> dict:
        session_id = get_session_id(__session__)
        messages: List[Dict] = body.get("messages", [])
        if not messages:
            return make_response("No messages provided.")

        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else ""

        state, update_res = abi_update_for_utterance(
            session_id=session_id,
            text=last_user,
            scorer=self.scorer,
            engine=self.engine,
            mock=self.valves.use_mock_scores,
        )

        stage_for_display = update_res.stage_before.value

        _, gate_res = gate_for_session(
            session_id=session_id,
            gate=self.gate,
            is_private=self.valves.assume_private,
            is_polite=self.valves.assume_polite,
            has_preamble=self.valves.assume_preamble,
            threshold=self.valves.gate_threshold,
        )

        if not gate_res.should_answer:
            txt = (
                "I won't answer this."
                f" (p_answer={gate_res.p_answer:.2f}, trust={state.trust_weighted():.2f})"
            )
            return make_response(txt)

        base_reply = call_base_model(messages, model=self.valves.base_model)

        score_suffix = (
            f"\n\n[ABI local: A={update_res.A_local:.2f}, "
            f"B={update_res.B_local:.2f}, I={update_res.I_local:.2f}; "
            f"global: A={state.A:.2f}, B={state.B:.2f}, I={state.I:.2f}, "
            f"stage={stage_for_display}, p_answer={gate_res.p_answer:.2f}]"
        )

        final = base_reply + score_suffix
        return make_response(final)


class PipeMedium(BasePipe):
    """
    中等模式：回答意愿相对宽松，和原始 Pipe 行为接近。
    """

    def __init__(self):
        super().__init__()
        self.valves.gate_threshold = 0.5
        self.engine = ABIEngine(gamma=1.0, clip_min=0.0, clip_max=1.0)
        self.gate = Gate(
            w_trust=2.2,
            w_privacy=0.70,
            w_polite=0.30,
            w_preamble=0.30,
            bias=2.0,
        )


class PipeHard(BasePipe):
    """
    困难模式：更保守的 gating，同样 ABI 更新逻辑，但更难通过 gate。
    """

    def __init__(self):
        super().__init__()
        self.valves.gate_threshold = 0.7
        self.engine = ABIEngine(gamma=1.2, clip_min=0.0, clip_max=1.0)
        self.gate = Gate(
            w_trust=2.5,
            w_privacy=1.0,
            w_polite=0.30,
            w_preamble=0.20,
            bias=2.8,
        )


class PipeMediumHard(BasePipe):
    """
    介于 Medium 和 Hard 之间的模式：
    - gate 比 Medium 严一些，比 Hard 松
    - 输出展示 ΔA/ΔB/ΔI 以及变化原因，而不是当前全局数值
    """

    def __init__(self):
        super().__init__()
        self.valves.gate_threshold = 0.60
        self.engine = ABIEngine(gamma=1.1, clip_min=0.0, clip_max=1.0)
        self.gate = Gate(
            w_trust=2.35,
            w_privacy=0.85,
            w_polite=0.30,
            w_preamble=0.25,
            bias=2.50,
        )

    @staticmethod
    def _weight_text(w: float) -> str:
        if w >= 0.9:
            return "strongly weighted"
        if w >= 0.6:
            return "moderately weighted"
        return "weakly weighted"

    def _delta_reason_text(self, update_res: UpdateResult) -> str:
        wa, wb, wi = STAGE_SENS[update_res.stage_before]
        parts: List[str] = []

        def one(name_short: str, name_full: str, delta: float, local: float, w: float):
            if abs(delta) < 1e-3:
                trend = "stayed roughly stable"
            elif delta > 0:
                trend = "increased"
            else:
                trend = "decreased"
            base_reason = (
                f"{name_full} {trend} ({delta:+.3f}) because local {name_full}="
                f"{local:.2f} relative to neutral 0.50"
            )
            weight_reason = (
                f"and it is {self._weight_text(w)} in stage {update_res.stage_before.value}"
            )
            parts.append(f"{name_short}: {base_reason}, {weight_reason}")

        one("A", "ability", update_res.dA, update_res.A_local, wa)
        one("B", "benevolence", update_res.dB, update_res.B_local, wb)
        one("I", "integrity", update_res.dI, update_res.I_local, wi)

        return " ; ".join(parts)

    def pipe(self, body: dict, __user__=None, __session__=None) -> dict:
        session_id = get_session_id(__session__)
        messages: List[Dict] = body.get("messages", [])
        if not messages:
            return make_response("No messages provided.")

        user_msgs = [m for m in messages if m.get("role") == "user"]
        last_user = user_msgs[-1]["content"] if user_msgs else ""

        state, update_res = abi_update_for_utterance(
            session_id=session_id,
            text=last_user,
            scorer=self.scorer,
            engine=self.engine,
            mock=self.valves.use_mock_scores,
        )

        stage_for_display = update_res.stage_before.value

        _, gate_res = gate_for_session(
            session_id=session_id,
            gate=self.gate,
            is_private=self.valves.assume_private,
            is_polite=self.valves.assume_polite,
            has_preamble=self.valves.assume_preamble,
            threshold=self.valves.gate_threshold,
        )

        if not gate_res.should_answer:
            txt = (
                "I won't answer this."
                f" (p_answer={gate_res.p_answer:.2f}, trust={state.trust_weighted():.2f})"
            )
            return make_response(txt)

        base_reply = call_base_model(messages, model=self.valves.base_model)

        delta_text = self._delta_reason_text(update_res)
        r = gate_res.rationale
        gate_reason = (
            f"p_answer={gate_res.p_answer:.2f} "
            f"(thr={self.valves.gate_threshold:.2f}, "
            f"trust={r.get('trust', 0.0):.2f}, "
            f"non_private={r.get('non_private', 0.0):.2f}, "
            f"polite={r.get('polite', 0.0):.2f}, "
            f"preamble={r.get('preamble', 0.0):.2f})"
        )

        score_suffix = (
            f"\n\n[ABI delta: ΔA={update_res.dA:+.3f}, "
            f"ΔB={update_res.dB:+.3f}, ΔI={update_res.dI:+.3f}; "
            f"stage={stage_for_display}; reasons: {delta_text}; "
            f"gate: {gate_reason}]"
        )

        final = base_reply + score_suffix
        return make_response(final)


# 默认还是中等模式，需要的话你可以改成 Pipe = PipeMediumHard
Pipe = PipeMedium
