"""A21: fixed examples for the canonical seven-dimension rubric and ABI math.

Expected values are literal acceptance fixtures, independent of the implementation's
coefficients. A21_SCORING_SOURCE lets the mutation check load a disposable copy.
"""

import importlib.util
import os
from pathlib import Path
import unittest


SOURCE = Path(os.environ.get(
    "A21_SCORING_SOURCE",
    Path(__file__).resolve().parents[1] / "grading_feature/backend/scoring_service.py",
))
spec = importlib.util.spec_from_file_location("a21_scoring_service", SOURCE)
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)

DIMS = (
    "relevance",
    "politeness",
    "on_topic",
    "neutrality",
    "non_imperative",
    "clarity_optional",
    "privacy_minimization_optional",
)


def scores(**overrides):
    return {name: overrides.get(name, 0) for name in DIMS}


class RubricScoringTests(unittest.TestCase):
    def test_each_of_seven_dimensions_contributes_two_points(self):
        question = {"id": "q1", "text": "What is your goal?", "studentId": "s1"}
        for dimension in DIMS:
            with self.subTest(dimension=dimension):
                result = scoring.decorate_score(question, scores(**{dimension: 2}))
                self.assertEqual(result["score_total"], 2)
                self.assertEqual(result["verdict"], "needs_work")
                self.assertEqual(result[dimension], 2)

    def test_rubric_boundaries_and_full_scale(self):
        question = {"text": "How can I help?"}
        fixtures = (
            (scores(), 0, "needs_work"),
            (scores(relevance=2, politeness=2, on_topic=1), 5, "needs_work"),
            (scores(relevance=2, politeness=2, on_topic=2), 6, "ok"),
            (scores(relevance=2, politeness=2, on_topic=2, neutrality=2,
                    non_imperative=2), 10, "ok"),
            (scores(relevance=2, politeness=2, on_topic=2, neutrality=2,
                    non_imperative=2, clarity_optional=1), 11, "good"),
            ({name: 2 for name in DIMS}, 14, "good"),
        )
        for raw, total, verdict in fixtures:
            with self.subTest(total=total):
                result = scoring.decorate_score(question, raw)
                self.assertEqual(result["score_total"], total)
                self.assertEqual(result["verdict"], verdict)

    def test_missing_or_non_numeric_dimension_scores_count_as_zero(self):
        result = scoring.decorate_score({}, {"relevance": "2", "politeness": "invalid"})
        self.assertEqual(result["score_total"], 2)
        self.assertEqual(result["relevance"], 2)
        self.assertEqual(result["politeness"], 0)
        self.assertEqual(result["privacy_minimization_optional"], 0)


class AbiScoringTests(unittest.TestCase):
    def test_baseline_and_full_scale(self):
        for raw, expected in (
            (scores(), (0.12, 0.10, 0.00, 0.07)),
            ({name: 1 for name in DIMS}, (0.52, 0.50, 0.50, 0.51)),
            ({name: 2 for name in DIMS}, (0.92, 0.90, 1.00, 0.94)),
        ):
            with self.subTest(raw=raw):
                abi = scoring.compute_abi_for_question(raw)
                self.assertEqual(
                    tuple(abi[key] for key in ("ability", "benevolence", "integrity", "abi_total")),
                    expected,
                )

    def test_each_rubric_signal_has_its_current_abi_weight(self):
        # All other rubric inputs are zero. The 0.12/0.10 baselines come from
        # fixed calibrated_confidence=0.6 and shared_interest=0.5.
        expected = {
            "relevance": (0.47, 0.10, 0.00, 0.19),
            "politeness": (0.12, 0.65, 0.00, 0.26),
            "on_topic": (0.12, 0.35, 0.30, 0.26),
            "neutrality": (0.32, 0.10, 0.25, 0.22),
            "non_imperative": (0.12, 0.10, 0.15, 0.12),
            "clarity_optional": (0.37, 0.10, 0.00, 0.16),
            "privacy_minimization_optional": (0.12, 0.10, 0.30, 0.17),
        }
        for dimension, fixture in expected.items():
            with self.subTest(dimension=dimension):
                abi = scoring.compute_abi_for_question(scores(**{dimension: 2}))
                self.assertEqual(
                    tuple(abi[key] for key in ("ability", "benevolence", "integrity", "abi_total")),
                    fixture,
                )

    def test_subdimension_mapping_and_constants(self):
        subs = scoring.compute_abi_for_question({name: 1 for name in DIMS})["subs"]
        self.assertEqual(subs, {
            "knowledge_consistency": 0.5,
            "professional_tone": 0.5,
            "rationality": 0.5,
            "calibrated_confidence": 0.6,
            "politeness": 0.5,
            "human_care": 0.5,
            "care_my_interest": 0.5,
            "shared_interest": 0.5,
            "legality": 0.5,
            "morality": 0.5,
            "contract": 0.5,
            "inducement": 0.5,
        })

    def test_group_averages_and_100_point_conversion(self):
        low = scoring.decorate_score({"studentId": "low"}, scores())
        high = scoring.decorate_score({"studentId": "high"}, {name: 2 for name in DIMS})
        for row in (low, high):
            row["abi"] = scoring.compute_abi_for_question(row)
        aggregate = scoring.build_aggregate([low, high], abi_enabled=True)
        self.assertEqual(aggregate["avg_total_0_14"], 7)
        self.assertEqual(aggregate["overall_0_100"], 50)
        self.assertEqual(aggregate["distribution"]["counts"], [1, 0, 0, 1])
        self.assertEqual(aggregate["dims"], {name: 1 for name in DIMS})
        self.assertEqual(aggregate["abi_global"]["abi_total"], 0.51)
        self.assertEqual(aggregate["perStudent"]["low"]["avg_total_0_14"], 0)
        self.assertEqual(aggregate["perStudent"]["high"]["avg_total_0_14"], 14)


if __name__ == "__main__":
    unittest.main()
