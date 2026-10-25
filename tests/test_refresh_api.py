import asyncio
import os
import sys
import tempfile
import json
import pytest
from starlette.testclient import TestClient

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grading_feature", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import main
import scoring_service


@pytest.fixture(autouse=True)
def setup_environment(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_cache.sqlite3")
    test_data = str(tmp_path / "test_data.json")

    # Sample user and chat data
    sample_users = [
        {
            "user_id": "u1",
            "name": "Student One",
            "email": "student1@example.com",
            "chats": [
                {
                    "title": "Advising Chat 1",
                    "message_pairs": [
                        {"question": "What is your investment horizon?", "answer": "10 years."},
                        {"question": "How much debt do you have?", "answer": "None."},
                    ],
                }
            ],
        }
    ]
    with open(test_data, "w", encoding="utf-8") as f:
        json.dump(sample_users, f)

    monkeypatch.setenv("SCORE_CACHE_PATH", test_db)
    monkeypatch.setenv("DATA_PATH", test_data)
    monkeypatch.setenv("OUTPUT_PATH", test_data)
    monkeypatch.setenv("MOCK_SCORER", "1")

    # Reset in-memory states
    scoring_service.clear_score_cache()
    scoring_service.init_score_cache()
    main.DATA_PATH = test_data
    main.refresh_manager = main.RefreshJobManager()

    # Mock extract_data to avoid needing live OpenWebUI
    def mock_extract():
        import time
        time.sleep(0.3)
        return {
            "users_processed": 1,
            "chat_entries_processed": 1,
            "message_pairs_processed": 2,
            "latest_message_timestamp_found": "10/25/2026 15:00",
            "output_file_path": test_data,
        }


    monkeypatch.setattr(main, "extract_data", mock_extract)
    yield
    scoring_service.clear_score_cache()


def test_initial_refresh_status():
    client = TestClient(main.app)
    response = client.get("/refresh/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "idle"
    assert data["progress"] == 0


def test_sync_refresh():
    client = TestClient(main.app)
    response = client.get("/refresh?sync=true")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "refresh_metadata" in data
    assert data["refresh_metadata"]["users_processed"] == 1


def test_async_refresh_flow():
    async def _test():
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            # 1. Trigger background refresh
            post_res = await client.post("/refresh")
            assert post_res.status_code == 202
            post_data = post_res.json()
            assert post_data["status"] == "in_progress"
            assert "job_id" in post_data

            # 2. Triggering while running should return 409 Conflict
            conflict_res = await client.post("/refresh")
            assert conflict_res.status_code == 409
            assert conflict_res.json()["detail"]["status"] == "in_progress"

            # 3. Poll status until completed
            max_wait = 5.0
            elapsed = 0.0
            completed = False
            while elapsed < max_wait:
                status_res = await client.get("/refresh/status")
                assert status_res.status_code == 200
                current = status_res.json()
                if current["status"] == "completed":
                    completed = True
                    assert current["progress"] == 100
                    assert current["refresh_metadata"] is not None
                    assert current["scoring_stats"]["total_questions"] == 2
                    break
                await asyncio.sleep(0.05)
                elapsed += 0.05

            assert completed, "Refresh job did not complete within timeout"

    asyncio.run(_test())



def test_scoring_cache_stats_and_clear():
    client = TestClient(main.app)
    # Check stats endpoint
    stats_res = client.get("/api/cache/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["ok"] is True

    # Clear endpoint
    clear_res = client.post("/api/cache/clear")
    assert clear_res.status_code == 200
    assert clear_res.json()["ok"] is True
