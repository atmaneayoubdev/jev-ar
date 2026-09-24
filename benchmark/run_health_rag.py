"""Score a model on health-rag, the Jev-AR main benchmark: one health-insurance RAG app, 6 routes, 20 questions.

    python benchmark/run_health_rag.py                        # atmaneayoub/jev-ar (gated: accept its licence first)
    python benchmark/run_health_rag.py --model <local folder or repo id of a Jev-AR model>

Each question is routed over the English and the Arabic route list, in listed order and reversed. The main
score is the number of correct answers over the two listed lists (40 decisions); "flips" counts answers that
change when the list is reversed. The benchmark's content hash is checked first, so a modified copy can't be
scored by mistake.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from jev_ar_route import MODEL_ID, JevAR  # noqa: E402

BENCH = HERE / "health_rag_v1.json"
SHA256 = "d1c8fb26b512460b06576014865a949d2e0717095df7971983068236ec700340"  # of the parsed JSON, key-sorted


def load():
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(bench, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                            .encode("utf-8")).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{BENCH} was modified (content sha256 {digest[:12]}..., expected {SHA256[:12]}...)")
    return bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()
    bench, router = load(), JevAR(args.model)
    correct, flips, rows = {}, {}, []
    for lang in ("en", "ar"):
        routes = [(r["id"], r[f"name_{lang}"], r[f"desc_{lang}"]) for r in bench["routes"]]
        to_id = {name: rid for rid, name, _ in routes}
        listed = {name: desc for _, name, desc in routes}
        reverse = {name: desc for _, name, desc in routes[::-1]}
        correct[lang] = flips[lang] = 0
        for q in bench["questions"]:
            a = router.route(q["text"], listed, bench["instruction"][lang])
            b = router.route(q["text"], reverse, bench["instruction"][lang])
            ok = to_id[a["route"]] == q["route"]
            correct[lang] += ok
            flips[lang] += a["route"] != b["route"]
            rows.append((q["id"], lang, q["slice"], q["route"], to_id[a["route"]], a["confidence"], ok))
    n = len(bench["questions"])
    print(f"model: {args.model}")
    print(f"main score: {correct['en'] + correct['ar']}/{2 * n}  (English list {correct['en']}/{n}, "
          f"Arabic list {correct['ar']}/{n}; flips when reversed: {flips['en']} / {flips['ar']})")
    for qid, lang, sl, gold, pred, conf, ok in rows:
        if not ok:
            print(f"  miss: #{qid} [{lang}, {sl}] gold {gold} -> {pred} ({conf:.2f})")


if __name__ == "__main__":
    main()
