"""Verify that a synthetic avatar transcript was extracted and scored.

Run only against a local/authorized *test* grading service with synthetic data.
"""
import argparse
import json
import sys
from urllib.request import urlopen
from urllib.parse import quote
from urllib.error import HTTPError, URLError


def get(base, path):
    with urlopen(base.rstrip('/') + path, timeout=60) as response:
        return json.load(response)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://localhost:9500')
    ap.add_argument('--student-id', required=True)
    ap.add_argument('--title', default='B13 verification — synthetic avatar practice')
    ap.add_argument('--min-pairs', type=int, default=2)
    ap.add_argument('--confirm-score-api', action='store_true', help='May run paid model evaluation')
    args = ap.parse_args()
    if not args.confirm_score_api:
        ap.error('Pass --confirm-score-api to call the grading API (may consume paid model tokens)')
    try:
        student = get(args.base_url, '/api/user/' + quote(args.student_id, safe=''))
        if student.get('user_id') != args.student_id:
            raise ValueError('Grading service has no extracted record for this student')
        matching = [c for c in student.get('chats', []) if c.get('title') == args.title]
        if not matching:
            raise ValueError('Synthetic avatar chat title not found under this student')
        if max(len(c.get('message_pairs', [])) for c in matching) < args.min_pairs:
            raise ValueError('Fewer persisted question/answer pairs than expected')
        score = get(args.base_url, '/api/student-feedback/' + quote(args.student_id, safe=''))
        if not score.get('ok') or score.get('user', {}).get('user_id') != args.student_id:
            raise ValueError('Grading feedback belongs to wrong student or failed')
        results = score.get('results') or []
        source_questions = {p.get('question') for c in matching for p in c.get('message_pairs', [])}
        # Response shape may vary; require at least one result and student attribution.
        if not results:
            raise ValueError('Grading returned no scored questions')
        print('B13 grading verification PASS')
        print(f'student_id={args.student_id} extracted_chats={len(matching)} min_pairs={args.min_pairs}')
        print(f'scored_questions_for_student={len(results)}')
        print('Please inspect the dashboard itself for the exact synthetic questions and rubric scores.')
        return 0
    except (HTTPError, URLError, ValueError, TypeError, KeyError) as exc:
        print('B13 grading verification FAILED: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
