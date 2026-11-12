import asyncio
import os
import sys
import tempfile
import pytest

# Ensure grading_feature/backend is on the import path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grading_feature", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import scoring_service


@pytest.fixture(autouse=True)
def setup_temp_cache(tmp_path):
    test_db = str(tmp_path / "test_score_cache.sqlite3")
    os.environ["SCORE_CACHE_PATH"] = test_db
    os.environ["MOCK_SCORER"] = "1"
    scoring_service.clear_score_cache()
    scoring_service.init_score_cache()
    yield
    scoring_service.clear_score_cache()


def test_scoring_cache_hit_and_miss():
    async def _test():
        test_questions = [
            {"id": "q1", "text": "What is the client's current income?", "studentId": "s1", "studentName": "Alice"},
            {"id": "q2", "text": "How do you plan to handle long-term debt?", "studentId": "s2", "studentName": "Bob"},
        ]

        # First call: cache miss
        stats_before = scoring_service.get_cache_stats()
        assert stats_before["cached_entries"] == 0

        scored, aggregate = await scoring_service.score_questions(test_questions, use_abi=True)
        assert len(scored) == 2
        assert aggregate["count"] == 2

        stats_after_miss = scoring_service.get_cache_stats()
        assert stats_after_miss["cached_entries"] == 2
        assert stats_after_miss["misses"] == 2

        # Second call with same questions: cache hit!
        scored_again, aggregate_again = await scoring_service.score_questions(test_questions, use_abi=True)
        assert len(scored_again) == 2

        stats_after_hit = scoring_service.get_cache_stats()
        assert stats_after_hit["hits"] == 2
        # Ensure scores match
        assert scored[0]["score_total"] == scored_again[0]["score_total"]
        assert scored[0]["abi"] == scored_again[0]["abi"]

    asyncio.run(_test())


def test_progress_callback():
    async def _test():
        test_questions = [
            {"id": f"q{i}", "text": f"Question number {i}?", "studentId": "s1", "studentName": "Alice"}
            for i in range(5)
        ]
        calls = []

        def on_progress(completed, total):
            calls.append((completed, total))

        await scoring_service.score_questions(test_questions, use_abi=False, progress_callback=on_progress)
        assert len(calls) == 5
        assert calls[-1] == (5, 5)

    asyncio.run(_test())


def test_bypass_cache():
    async def _test():
        test_question = [{"id": "q1", "text": "Can we bypass cache?", "studentId": "s1"}]
        await scoring_service.score_questions(test_question, use_abi=False)

        stats1 = scoring_service.get_cache_stats()
        assert stats1["misses"] == 1

        # Call with bypass_cache=True
        await scoring_service.score_questions(test_question, use_abi=False, bypass_cache=True)
        stats2 = scoring_service.get_cache_stats()
        # hits should NOT have incremented
        assert stats2["hits"] == 0

    asyncio.run(_test())


def test_mock_and_real_cache_keys_are_isolated(monkeypatch):
    async def _test():
        question = [{"id": "q1", "text": "How do you budget monthly expenses?", "studentId": "s1"}]

        # 1. Run in mock mode (or missing API key)
        monkeypatch.setattr(scoring_service, "MOCK_SCORER", True)
        monkeypatch.setattr(scoring_service, "OPENAI_API_KEY", "")
        mock_scored, _ = await scoring_service.score_questions(question, use_abi=False)
        assert mock_scored[0]["notes"] == "mock score"

        # 2. Restore real credentials and stub score_with_openai
        monkeypatch.setattr(scoring_service, "MOCK_SCORER", False)
        monkeypatch.setattr(scoring_service, "OPENAI_API_KEY", "sk-real-test-key")
        real_calls = []

        def fake_score_with_openai(q):
            real_calls.append(q["text"])
            return scoring_service.decorate_score(
                q,
                {
                    "relevance": 2,
                    "politeness": 2,
                    "on_topic": 2,
                    "neutrality": 2,
                    "non_imperative": 2,
                    "clarity_optional": 2,
                    "privacy_minimization_optional": 2,
                    "notes": "real score",
                },
            )

        monkeypatch.setattr(scoring_service, "score_with_openai", fake_score_with_openai)

        real_scored, _ = await scoring_service.score_questions(question, use_abi=False)
        assert len(real_calls) == 1
        assert real_scored[0]["notes"] == "real score"

        # 3. Second real call should hit the real cache, not call OpenAI again
        real_scored_cached, _ = await scoring_service.score_questions(question, use_abi=False)
        assert len(real_calls) == 1
        assert real_scored_cached[0]["notes"] == "real score"

    asyncio.run(_test())

