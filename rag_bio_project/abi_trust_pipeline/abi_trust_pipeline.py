
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Tuple, Optional
import os, json, math, statistics, time

# ----------------------------- Enums / Config ----------------------------- #

class Stage(Enum):
    CONTRACT = "contract"
    KNOWLEDGE = "knowledge"
    EMPATHY = "empathy"

def _lvl(x: str) -> float:
    return {"低":0.20, "中":0.50, "高":1.00}[x]

STAGE_SENS = {
    Stage.CONTRACT:  (_lvl("中"), _lvl("低"), _lvl("高")),
    Stage.KNOWLEDGE: (_lvl("中"), _lvl("中"), _lvl("低")),
    Stage.EMPATHY:   (_lvl("低"), _lvl("高"), _lvl("低")),
}

THRESH = {"I_high": 0.70, "B_high": 0.70, "A_conv_std": 0.05}
WINDOW = 5

# ----------------------------- Atomic Scores ----------------------------- #

from dataclasses import dataclass

@dataclass
class AbilitySubs:
    knowledge_consistency: float
    professional_tone: float
    rationality: float
    calibrated_confidence: float
    def to_score(self, w=(0.35,0.25,0.20,0.20)) -> float:
        a = (self.knowledge_consistency*w[0] + self.professional_tone*w[1] +
             self.rationality*w[2] + self.calibrated_confidence*w[3])
        return max(0.0, min(1.0, a))

@dataclass
class BenevolenceSubs:
    politeness: float
    human_care: float
    care_my_interest: float
    shared_interest: float
    def to_score(self, w=(0.25,0.30,0.25,0.20)) -> float:
        b = (self.politeness*w[0] + self.human_care*w[1] +
             self.care_my_interest*w[2] + self.shared_interest*w[3])
        return max(0.0, min(1.0, b))

@dataclass
class IntegritySubs:
    legality: float
    morality: float
    contract: float
    inducement: float
    def to_score(self, w=(0.30,0.25,0.30,0.15)) -> float:
        i = (self.legality*w[0] + self.morality*w[1] + self.contract*w[2] +
             (1.0 - self.inducement)*w[3])
        return max(0.0, min(1.0, i))

@dataclass
class UtteranceABISubs:
    ability: AbilitySubs
    benevolence: BenevolenceSubs
    integrity: IntegritySubs
    def to_abi(self):
        return (self.ability.to_score(), self.benevolence.to_score(), self.integrity.to_score())

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
        return (wa*self.A + wb*self.B + wi*self.I) / (s if s>0 else 1.0)

@dataclass
class UpdateResult:
    A_local: float; B_local: float; I_local: float
    dA: float; dB: float; dI: float
    stage_before: Stage; stage_after: Stage
    A_after: float; B_after: float; I_after: float
    notes: Dict[str,str]

class ABIEngine:
    def __init__(self, gamma: float = 1.0, clip_min: float = 0.0, clip_max: float = 1.0):
        self.gamma = gamma; self.clip_min = clip_min; self.clip_max = clip_max
    @staticmethod
    def _clip(x, lo, hi): return min(hi, max(lo, x))
    def update(self, state: TrustState, subs: UtteranceABISubs) -> UpdateResult:
        A_loc, B_loc, I_loc = subs.to_abi()
        wa, wb, wi = state.stage_weights()
        dA = wa * (A_loc - 0.5) * (state.A ** self.gamma)
        dB = wb * (B_loc - 0.5) * (state.B ** self.gamma)
        dI = wi * (I_loc - 0.5) * (state.I ** self.gamma)
        A_new = self._clip(state.A + dA, self.clip_min, self.clip_max)
        B_new = self._clip(state.B + dB, self.clip_min, self.clip_max)
        I_new = self._clip(state.I + dI, self.clip_min, self.clip_max)
        stage_before = state.stage; notes = {}

        state.history_A.append(dA); state.history_B.append(dB); state.history_I.append(dI)
        if len(state.history_A) > WINDOW: state.history_A.pop(0)
        if len(state.history_B) > WINDOW: state.history_B.pop(0)
        if len(state.history_I) > WINDOW: state.history_I.pop(0)

        stage_after = stage_before
        if stage_before == Stage.CONTRACT:
            A_std = statistics.pstdev(state.history_A) if len(state.history_A)>=2 else 1.0
            if I_new >= THRESH["I_high"] and A_std < THRESH["A_conv_std"]:
                stage_after = Stage.KNOWLEDGE
                notes["stage_transition"] = "CONTRACT→KNOWLEDGE"
        elif stage_before == Stage.KNOWLEDGE:
            if B_new >= THRESH["B_high"] and I_new >= 0.55:
                stage_after = Stage.EMPATHY
                notes["stage_transition"] = "KNOWLEDGE→EMPATHY"

        state.A, state.B, state.I = A_new, B_new, I_new
        state.stage = stage_after
        return UpdateResult(A_loc,B_loc,I_loc,dA,dB,dI,stage_before,stage_after,A_new,B_new,I_new,notes)

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
    rationale: Dict[str,float]

def sigmoid(x: float) -> float:
    return 1.0/(1.0+math.exp(-x))

class Gate:
    def __init__(self, w_trust=2.2,w_privacy=0.70,w_polite=0.30,w_preamble=0.30,bias=2.0):
        self.w_trust=w_trust; self.w_privacy=w_privacy; self.w_polite=w_polite; self.w_preamble=w_preamble; self.bias=bias
    def decide(self, x: GateInputs, threshold: float = 0.5) -> GateResult:
        score = self.w_trust*x.trust_weighted + self.w_privacy*(1-x.is_private) + self.w_polite*x.is_polite + self.w_preamble*x.has_preamble - self.bias
        p = sigmoid(score)
        return GateResult(p, p>=threshold, dict(trust=x.trust_weighted, non_private=(1-x.is_private), polite=x.is_polite, preamble=x.has_preamble, linear_score=score))

# ----------------------------- API Scorer ----------------------------- #

class APIScorer:
    """
    API + Prompt 方式对一句文本进行子维度打分，返回 UtteranceABISubs。
    默认支持 OpenAI Chat Completions 风格（通过环境变量 OPENAI_API_KEY, OPENAI_MODEL）。
    也可通过自定义 HTTP Endpoint（设置 SCORE_API_URL, SCORE_API_KEY）。
    在无 API Key 情况下可使用 mock=True 返回固定示例。
    """
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
                response_format={"type":"json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            return self._parse_abi_json(data)
        except Exception as e:
            return self._mock_scores(text)

    def _score_generic_endpoint(self, text: str) -> UtteranceABISubs:
        import requests
        url = os.getenv("SCORE_API_URL")
        key = os.getenv("SCORE_API_KEY","")
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
        except Exception as e:
            return self._mock_scores(text)

    @staticmethod
    def _parse_abi_json(data: dict) -> UtteranceABISubs:
        a = data.get("ability",{})
        b = data.get("benevolence",{})
        i = data.get("integrity",{})
        return UtteranceABISubs(
            ability=AbilitySubs(
                knowledge_consistency=float(a.get("knowledge_consistency",0.5)),
                professional_tone=float(a.get("professional_tone",0.5)),
                rationality=float(a.get("rationality",0.5)),
                calibrated_confidence=float(a.get("calibrated_confidence",0.5)),
            ),
            benevolence=BenevolenceSubs(
                politeness=float(b.get("politeness",0.5)),
                human_care=float(b.get("human_care",0.5)),
                care_my_interest=float(b.get("care_my_interest",0.5)),
                shared_interest=float(b.get("shared_interest",0.5)),
            ),
            integrity=IntegritySubs(
                legality=float(i.get("legality",0.5)),
                morality=float(i.get("morality",0.5)),
                contract=float(i.get("contract",0.5)),
                inducement=float(i.get("inducement",0.5)),
            ),
        )

    @staticmethod
    def _mock_scores(text: str) -> UtteranceABISubs:
        # Simple heuristic mock for offline runs
        base = 0.7 if len(text) > 20 else 0.55
        return UtteranceABISubs(
            ability=AbilitySubs(
                knowledge_consistency=base,
                professional_tone=0.75,
                rationality=0.72,
                calibrated_confidence=0.70
            ),
            benevolence=BenevolenceSubs(
                politeness=0.78, human_care=0.70, care_my_interest=0.68, shared_interest=0.62
            ),
            integrity=IntegritySubs(
                legality=0.92, morality=0.85, contract=0.88, inducement=0.12
            )
        )
