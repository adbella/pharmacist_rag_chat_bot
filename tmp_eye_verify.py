import json
import os
import subprocess
import sys
import time

import requests


def wait_ready(base_url: str, timeout_sec: int = 360):
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            r = requests.get(f"{base_url}/health", timeout=1.5)
            if r.status_code == 200:
                return True, r.json()
        except Exception:
            pass
        time.sleep(1)
    return False, None


def run_test():
    api = "http://127.0.0.1:8012"
    result = {
        "server_ready": False,
        "health": None,
        "chat_http": None,
        "statuses": [],
        "done": None,
        "error": None,
    }

    env = os.environ.copy()
    env.setdefault("OOS_GUARD_ENABLED", "true")
    env.setdefault("OOS_MIN_RELEVANCE", "0.55")
    env.setdefault("OOS_MIN_TOP_SCORE", "0.002")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8012"],
        env=env,
    )

    try:
        ok, health = wait_ready(api)
        result["server_ready"] = ok
        result["health"] = health
        if not ok:
            result["error"] = "server_not_ready"
            return result

        payload = {
            "query": "눈이 건조한데 어떻게 하지",
            "model": "gpt-5",
            "top_k": 5,
            "ensemble_k": 20,
            "weight_bm25": 0.8,
            "use_self_correction": True,
        }
        resp = requests.post(f"{api}/chat", json=payload, stream=True, timeout=480)
        result["chat_http"] = resp.status_code

        ev = None
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event: "):
                ev = line[7:]
                continue
            if not line.startswith("data: "):
                continue

            data = json.loads(line[6:])
            if ev == "status":
                result["statuses"].append(data.get("step"))
            elif ev == "error":
                result["error"] = data
                break
            elif ev == "done":
                result["done"] = {
                    "verify_result": data.get("verify_result"),
                    "answer_preview": (data.get("answer") or "").replace("\n", " ")[:500],
                    "metrics": data.get("metrics"),
                }
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

    return result


if __name__ == "__main__":
    out = run_test()
    with open("tmp_eye_verify_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("WROTE tmp_eye_verify_result.json")
