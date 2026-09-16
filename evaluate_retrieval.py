import argparse
import json
import os

from rag.vector_store import VectorStoreService


RECALL_K_LIST = [1, 3, 5]


def normalize_path_name(path: str) -> str:
    return os.path.basename(str(path).replace("\\", "/"))


def is_hit(doc, expected_source: str, expected_keywords: list[str]) -> bool:
    source = normalize_path_name(doc.metadata.get("source", ""))
    source_hit = source == expected_source

    if not expected_keywords:
        return source_hit

    keyword_hit = any(keyword in doc.page_content for keyword in expected_keywords)
    return source_hit and keyword_hit


def search_documents(vector_store: VectorStoreService, query: str, mode: str, top_k: int):
    if mode == "vector":
        return vector_store.vector_search(query, top_k=top_k)

    if mode == "hybrid":
        return vector_store.hybrid_search(query, final_k=top_k, use_rerank=False)

    if mode == "hybrid_rerank":
        return vector_store.hybrid_search(query, final_k=top_k, use_rerank=True)

    raise ValueError(f"不支持的评估模式：{mode}")


def evaluate(test_set_path: str, mode: str):
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    vector_store = VectorStoreService()
    counters = {k: 0 for k in RECALL_K_LIST}
    rows = []

    for case in test_cases:
        docs = search_documents(vector_store, case["question"], mode, max(RECALL_K_LIST))
        hit_at = {}

        for k in RECALL_K_LIST:
            hit_at[k] = any(
                is_hit(doc, case["expected_source"], case.get("expected_keywords", []))
                for doc in docs[:k]
            )
            if hit_at[k]:
                counters[k] += 1

        rows.append({
            "id": case["id"],
            "question": case["question"],
            "expected_source": case["expected_source"],
            "hit_at_1": hit_at[1],
            "hit_at_3": hit_at[3],
            "hit_at_5": hit_at[5],
            "top_sources": [normalize_path_name(doc.metadata.get("source", "")) for doc in docs],
        })

    total = len(test_cases)
    recalls = {f"Recall@{k}": counters[k] / total if total else 0 for k in RECALL_K_LIST}
    return {
        "mode": mode,
        "total": total,
        "recalls": recalls,
        "rows": rows,
    }


def print_report(result: dict):
    print(f"评估模式：{result['mode']}")
    print(f"测试题数量：{result['total']}")

    for name, value in result["recalls"].items():
        print(f"{name}: {value:.2%}")

    print("\n逐题结果：")
    for row in result["rows"]:
        print(
            f"{row['id']} | "
            f"@1={'Y' if row['hit_at_1'] else 'N'} "
            f"@3={'Y' if row['hit_at_3'] else 'N'} "
            f"@5={'Y' if row['hit_at_5'] else 'N'} | "
            f"期望={row['expected_source']} | "
            f"Top来源={row['top_sources']}"
        )


def main():
    parser = argparse.ArgumentParser(description="评估RAG检索Recall@1/3/5")
    parser.add_argument("--test-set", default="test_set_heritage_fire.json", help="测试集JSON路径")
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "hybrid_rerank"],
        default="hybrid",
        help="检索模式：vector=纯向量，hybrid=向量+BM25+RRF，hybrid_rerank=混合检索+Rerank",
    )
    parser.add_argument("--output", default=None, help="可选：保存详细评估结果JSON")
    args = parser.parse_args()

    result = evaluate(args.test_set, args.mode)
    print_report(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
