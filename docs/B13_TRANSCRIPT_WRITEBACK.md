# B13 — Avatar transcript write-back (WF-77)

## Decision and scope

**Preferred path implemented: Open WebUI is the canonical transcript store.** The avatar's speech pipeline supplies *finalized* student STT text and avatar reply text to the `AvatarWriteback` integration. It uses only public Open WebUI REST APIs to create a student-owned chat and append each turn. The existing `grading_feature/backend/extract_chats.py` reads these persisted chats via `/api/v1/users/all` and `/api/v1/chats/all/db`, groups by `user_id`, and pairs the flat `chat.messages` list of user/assistant messages. No second database or modified scoring logic is required.

**Not established yet:** Whether a bare `POST /api/chat/completions` automatically persists a chat on your particular v0.11.4 stack. The API is primarily a completion endpoint; don't assume persistence. Run the optional live probe (below) against the test stack to capture evidence. The write-back integration does not rely on the result.

## Avatar prototype recording

The supplied Nov 13 screen recording shows a **LiveAvatar sandbox** streaming an avatar with session controls. It demonstrates a candidate session UI, **not** an actual Open WebUI transcript integration, student authentication, or a scored grading dashboard record. The final avatar platform and event names are not yet known. Hook `append_turn_async(...)` into the event handler that receives a finalized student transcription and the finalized avatar response; do not write microphone audio or video to the grading chat.

## Integration code

1. Copy `tools/avatar_writeback/` and `tests/test_avatar_writeback.py` from this package into the matching paths at the repository root.
2. Run offline contract tests: `python -m unittest discover -s tests -p 'test_avatar_writeback.py' -v`.
3. The avatar backend must use a **student-scoped** Open WebUI bearer token. Using an administrator/service token creates a chat owned by that administrator, which the extractor will attribute to the wrong person. Do not accept a forged student_id from the browser or log tokens.
4. Start a session after appropriate consent and text transcription choices have been confirmed; save returned `chat_id` in the avatar session state. Reuse it for each finalized pair. Use a stable unique `turn_id` for retries.

```python
from tools.avatar_writeback import AvatarWriteback
from tools.avatar_writeback.writer import JsonAPI

writer = AvatarWriteback(
    JsonAPI('http://open-webui:8080', student_scoped_openwebui_token),
    model='your-configured-model',
    expected_user_id=authenticated_openwebui_user_id,
)
chat_id = writer.start_session(title='Avatar advising practice')
# After speech recognition and avatar's answer are finalized:
await writer.append_turn_async(
    chat_id, turn_id=avatar_turn_id,
    student_text=final_student_transcription,
    avatar_text=final_avatar_reply_text,
)
```

This is an integration module rather than a complete LiveAvatar SDK adapter. Do not use one shared `AvatarWriteback` instance for different student accounts. Serialize concurrent writes to the **same chat** in the avatar backend; otherwise two read/modify/write operations can lose a turn. If the app runs with multiple replicas, use a distributed per-chat lock or append queue.

### Live synthetic test

Use a dedicated synthetic test user. Generate that user's Open WebUI API key through the supported user interface, or obtain a user-scoped token through the normal auth flow. Do **not** commit credentials or real transcripts. In Git Bash:

```bash
export AVATAR_STUDENT_API_KEY='YOUR_TEST_STUDENT_TOKEN'
python -m tools.avatar_writeback.demo \
  --base-url http://localhost:3000 \
  --model gpt-4o-mini \
  --student-id 'THE_TEST_STUDENT_OPENWEBUI_ID' \
  --confirm-synthetic-write
```

The `--model` value is metadata only for this write-back test; it does **not** contact the model. It should still be the model your avatar actually uses.

Expected output: `B13 synthetic write-back PASS`, `message_count=4`, and the returned `chat_id`. Then open the same **test student account** in Open WebUI and verify the chat exists and has two question/answer pairs.

### Test the story's open question separately (optional, may cost model tokens)

```bash
python -m tools.avatar_writeback.probe_completion \
  --base-url http://localhost:3000 \
  --model gpt-4o-mini \
  --confirm-model-call
```

This compares the synthetic student's saved-chat IDs before and after a bare completion call. If the model isn't configured or a separate process writes a chat concurrently, treat the output as inconclusive. Explicit write-back avoids relying on autosave behavior.

### Confirm grader attribution and normal scoring (mandatory acceptance test)

1. On an **isolated test stack with synthetic data**, finish the write-back demo and record `chat_id`, `student_id`, and the unique synthetic title.
2. Confirm this chat appears in the Open WebUI UI under **that same student**, *not under the admin*. `GET /api/v1/chats/{chat_id}` (using the student's token) should return `user_id=<student_id>` and a `chat.messages` list of length 4 alternating `user`, `assistant`, `user`, `assistant`.
3. Through the grading backend, run `POST http://localhost:9500/api/refresh` (only after it is configured against the same Open WebUI test instance) or click the instructor dashboard's refresh button. `GET /api/refresh/status` can show the job status. Beware: refreshing your **real grading dashboard** may read real educational records; limit this step to synthetic test data and an authorized environment.
4. In the grading dashboard, find the synthetic student and confirm the new avatar session is present with the two question/answer pairs. Check the score calculation through the existing normal grading UI/API; record the resulting rubric and ABI scores or an approved synthetic example. A persisted chat alone is *not* enough to satisfy the scoring acceptance criterion.
5. Ensure the team has the B2 instructor-approved privacy/consent policy **before collecting or recording any actual student speech**. Keep the default text transcript only unless another retention policy is formally approved.

### Acceptance evidence for Jira

| Evidence | Current status |
|---|---|
| Public API write-back implementation | Prepared in this package |
| Offline tests for transcript structure, attribution, idempotency, failure | Covered |
| `POST /api/chat/completions` autosave experiment | Must run against local v0.11.4 instance |
| Live synthetic student chat visible in Open WebUI | Must run against local instance |
| Synthetic session visible under correct student in grading dashboard | Must run against local stack |
| Synthetic session scored normally | Must run against grading service |

**Do not mark B13 Done** until the final three acceptance checks are demonstrated and documented with screenshots/IDs. This package does not include a live avatar SDK or access to your VM's running Open WebUI/grading service.

## Caveats

- The current extractor reads `chat.messages` (flat list), not only the newer `chat.history.messages` map. The write-back service updates both structures.
- An interrupted network response from *chat creation* may create an unknown duplicate chat; start the chat once per session and save its returned ID immediately. Stable per-turn IDs make **turn writes** retry-safe.
- The writer assumes one ordered writer per chat. Protect concurrent writers with an application-level lock/queue. A `POST` of the full chat payload is not a transactional append.
- If multiple model calls or TTS chunks are produced for one student message, save **one finalized assistant reply** instead of each chunk.
- Never put bearer tokens in browser JavaScript or Git. Only a trusted server-side avatar application should handle the token.
- Open WebUI's public chat routes and the grading extractor were inspected against currently published source and documentation, but only live integration can establish your exact deployed behavior.

## References

- Open WebUI v0.11.4 routes: https://github.com/open-webui/open-webui/blob/v0.11.4/backend/open_webui/routers/chats.py
- Backend-controlled UI-compatible API flow (community example; originally v0.6.15): https://docs.openwebui.com/reference/api-flow/
- Existing extractor: https://github.com/mfedenia/FamilyFinanceChat/blob/main/grading_feature/backend/extract_chats.py
- Existing grading refresh endpoints: https://github.com/mfedenia/FamilyFinanceChat/blob/main/grading_feature/backend/main.py

### Automated verification after grader refresh

After the refresh job finishes successfully, run this optional verifier (only against **synthetic test data**):

```bash
python -m tools.avatar_writeback.verify_grading \
  --base-url http://localhost:9500 \
  --student-id 'THE_TEST_STUDENT_OPENWEBUI_ID' \
  --confirm-score-api
```

It checks that the correct synthetic student has the avatar chat with at least two question/answer pairs and that their grading feedback contains scored results. This is a live integration check, *not* an offline proof that the grading system was run. Scores may use a paid API.
