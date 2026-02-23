from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

from dotenv import load_dotenv

from external_pharma_api import fetch_external_pharma_docs


def main() -> None:
    load_dotenv()

    os.environ["EXTERNAL_LLM_QUERY_ROUTER"] = "true"
    os.environ["EXTERNAL_LLM_QUERY_ROUTER_MODEL"] = os.getenv(
        "EXTERNAL_LLM_QUERY_ROUTER_MODEL", "gpt-5-nano"
    )
    os.environ["EXTERNAL_LLM_QUERY_ROUTER_MAX_KEYWORDS"] = os.getenv(
        "EXTERNAL_LLM_QUERY_ROUTER_MAX_KEYWORDS", "6"
    )
    os.environ["EXTERNAL_AUTO_TRANSLATE"] = "false"

    queries = [
        "\ub208\uc774 \uac74\uc870\ud558\uace0 \ucda9\ud608\ub420 \ub54c \uc4f8 \uc218 \uc788\ub294 \uc57d",
        "\uc54c\ub808\ub974\uae30 \ube44\uc5fc \uc57d \uc131\ubd84",
    ]

    result_payload: dict = {"ok": True, "items": []}
    try:
        for q in queries:
            docs = fetch_external_pharma_docs(
                query=q,
                provider="korea_hybrid",
                top_k=4,
                timeout_sec=12.0,
            )
            result_payload["items"].append(
                {
                    "query": q,
                    "count": len(docs),
                    "top_sources": [d.metadata.get("source") for d in docs[:5]],
                    "top_providers": [d.metadata.get("provider") for d in docs[:5]],
                }
            )
    except Exception as e:
        result_payload = {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    out = json.dumps(result_payload, ensure_ascii=False, indent=2)
    print(out)
    Path("external_test_result.json").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
