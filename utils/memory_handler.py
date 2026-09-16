import csv
import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, List

from model.factory import chat_model
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


MEMORY_PATH = get_abs_path("data/memory/conversation_memory.csv")
MEMORY_FIELDS = [
    "memory_id",
    "user_id",
    "created_at",
    "topic",
    "site_name",
    "heritage_type",
    "summary",
    "key_facts",
    "risk_points",
    "suggestions",
    "source_question",
]


def _ensure_memory_file():
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    if not os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MEMORY_FIELDS)
            writer.writeheader()


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def _fallback_summary(question: str, answer: str) -> dict:
    content = f"{question}\n{answer}".strip()
    return {
        "topic": "历史对话",
        "site_name": "",
        "heritage_type": "",
        "summary": content[:300],
        "key_facts": "；".join(re.findall(r"[^。；;\n]*\d+[^。；;\n]*", content)[:8]),
        "risk_points": "",
        "suggestions": "",
    }


def summarize_conversation(question: str, answer: str) -> dict:
    prompt = f"""
你是文物安防智能助手的长期记忆提取器。请从本轮对话中提取可复用的长期记忆。

要求：
1. 只输出JSON对象，不要输出Markdown或解释。
2. 保留关键数字数据，例如温度、湿度、巡检次数、月份、点位编号、比例、风险等级等。
3. 如果本轮只是寒暄、无意义输入、纯页面操作问题，则summary可以为空字符串。
4. 不要编造用户没有提供、工具没有返回的信息。

JSON字段：
{{
  "topic": "一句话主题",
  "site_name": "点位、场馆或对象；没有则为空",
  "heritage_type": "文物类型；没有则为空",
  "summary": "本轮对话摘要，80到200字",
  "key_facts": "关键事实和数字，用分号分隔",
  "risk_points": "风险点，用分号分隔",
  "suggestions": "建议，用分号分隔"
}}

用户问题：
{question}

助手回答：
{answer}
""".strip()

    try:
        response = chat_model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        memory = _extract_json(content)
    except Exception as e:
        logger.warning(f"[memory]长期记忆摘要生成失败，使用兜底摘要：{e}")
        memory = _fallback_summary(question, answer)

    return {field: str(memory.get(field, "")).strip() for field in MEMORY_FIELDS if field not in {
        "memory_id", "user_id", "created_at", "source_question"
    }}


def save_conversation_memory(user_id: str, question: str, answer: str) -> bool:
    user_id = (user_id or "").strip()
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not user_id or not question or not answer:
        return False

    memory = summarize_conversation(question, answer)
    if not memory.get("summary"):
        logger.info("[memory]本轮对话未产生可保存的长期记忆")
        return False

    row = {
        "memory_id": uuid.uuid4().hex,
        "user_id": user_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_question": question,
        **memory,
    }

    _ensure_memory_file()
    with open(MEMORY_PATH, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MEMORY_FIELDS)
        writer.writerow(row)

    logger.info(f"[memory]已写入用户{user_id}的长期记忆：{row['memory_id']}")
    return True


def search_conversation_memory(user_id: str, query: str, limit: int = 5) -> List[Dict[str, str]]:
    user_id = (user_id or "").strip()
    query = (query or "").strip()
    if not user_id or not query or not os.path.exists(MEMORY_PATH):
        return []

    query_terms = [term for term in re.split(r"\s+|，|,|。|；|;|：|:", query) if term]
    scored_rows = []

    with open(MEMORY_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("user_id", "").strip() != user_id:
                continue

            haystack = " ".join(row.get(field, "") for field in MEMORY_FIELDS if field != "user_id")
            score = sum(1 for term in query_terms if term in haystack)
            if query in haystack:
                score += 3
            if score > 0:
                scored_rows.append((score, row))

    scored_rows.sort(key=lambda item: (item[0], item[1].get("created_at", "")), reverse=True)
    return [row for _, row in scored_rows[:limit]]


def format_memory_rows(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return ""

    blocks = []
    for index, row in enumerate(rows, start=1):
        blocks.append(
            f"【历史记忆{index}】\n"
            f"时间：{row.get('created_at', '')}\n"
            f"主题：{row.get('topic', '')}\n"
            f"对象：{row.get('site_name', '')}\n"
            f"文物类型：{row.get('heritage_type', '')}\n"
            f"摘要：{row.get('summary', '')}\n"
            f"关键事实：{row.get('key_facts', '')}\n"
            f"风险点：{row.get('risk_points', '')}\n"
            f"建议：{row.get('suggestions', '')}"
        )

    return "\n\n".join(blocks)
