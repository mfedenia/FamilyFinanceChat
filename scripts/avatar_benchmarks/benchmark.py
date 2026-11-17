
import os
import csv
import json
import time
import statistics
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ["OPENWEBUI_URL"].rstrip("/")
API_KEY = os.environ["OPENWEBUI_API_KEY"]
MODEL = os.environ["OPENWEBUI_MODEL"]

QUESTIONS = [
    "What are your family's main financial goals?",
    "What is your family's monthly income?",
    "What are your biggest monthly expenses?",
    "Do you currently have any debts?",
    "How much do you save every month?"
]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

results = []

for index, question in enumerate(QUESTIONS, 1):
    payload = {
        "model": MODEL,
        "stream": True,
        "messages": [
            {"role": "user", "content": question}
        ]
    }

    start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    

    with requests.post(
        f"{BASE_URL}/api/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=120
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line.startswith(b"data: "):
                continue

            data = line[6:]
            if data == b"[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    now = time.perf_counter()

                    if first_token_time is None:
                        first_token_time = now

                    last_token_time = now
                    
    end = time.perf_counter()

    if first_token_time is not None and last_token_time is not None:
        print(
            f"Text streaming duration: "
            f"{last_token_time - first_token_time:.3f}s"
        )


    ttft = (
        first_token_time - start
        if first_token_time is not None
        else None
    )
    total = end - start

    results.append({
        "question_number": index,
        "ttft_seconds": ttft,
        "total_seconds": total
    })

    print(f"Question {index}: {question}")
    print(f"  First text: {ttft}")
    print(f"  Total time: {total:.3f}s\n")

with open("b4_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

times = [
    r["ttft_seconds"]
    for r in results
    if r["ttft_seconds"] is not None
]

if times:
    times.sort()

    def percentile(values, p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        return values[lower] + (
            values[upper] - values[lower]
        ) * fraction

    print("First-text latency:")
    print(f"p50: {statistics.median(times):.3f}s")
    print(f"p95: {percentile(times, 0.95):.3f}s")

print("Results saved to b4_results.csv")
