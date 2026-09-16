import os
import csv
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.memory_handler import search_conversation_memory, format_memory_rows

rag = RagSummarizeService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010", ]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}
current_user_id = None


def set_current_user_id(user_id: str):
    global current_user_id
    current_user_id = user_id


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    api_key = os.getenv("SENIVERSE_API_KEY") or agent_conf.get("weather_api_key", "")
    if not api_key:
        logger.warning("[get_weather]未配置心知天气API密钥")
        return "天气查询失败：未配置天气API密钥，请在环境变量SENIVERSE_API_KEY或config/agent.yml的weather_api_key中配置。"

    city = city.strip()
    if not city:
        return "天气查询失败：城市名称不能为空。"

    params = {
        "key": api_key,
        "location": city,
        "language": agent_conf.get("weather_language", "zh-Hans"),
        "unit": agent_conf.get("weather_unit", "c"),
        "start": str(agent_conf.get("weather_start", -1)),
        "days": str(agent_conf.get("weather_days", 5)),
    }
    api_url = agent_conf.get("weather_api_url", "https://api.seniverse.com/v3/weather/daily.json")
    request_url = f"{api_url}?{urlencode(params)}"

    try:
        with urlopen(request_url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        logger.exception(f"[get_weather]天气API请求失败，HTTP状态码：{e.code}")
        return f"天气查询失败：天气API返回HTTP状态码{e.code}。"
    except URLError as e:
        logger.exception(f"[get_weather]天气API网络请求失败：{e.reason}")
        return f"天气查询失败：无法连接天气API，原因：{e.reason}。"
    except Exception as e:
        logger.exception(f"[get_weather]天气API解析失败：{e}")
        return f"天气查询失败：{e}"

    try:
        result = payload["results"][0]
        location = result["location"]["name"]
        daily_items = result["daily"]
    except (KeyError, IndexError, TypeError):
        logger.warning(f"[get_weather]天气API返回结构异常：{payload}")
        return "天气查询失败：天气API返回数据结构异常。"

    lines = [f"{location}未来{len(daily_items)}天天气："]
    for item in daily_items:
        lines.append(
            f"{item.get('date', '')}：白天{item.get('text_day', '未知')}，夜间{item.get('text_night', '未知')}，"
            f"{item.get('low', '未知')}~{item.get('high', '未知')}℃，"
            f"湿度{item.get('humidity', '未知')}%，降水概率{item.get('precip', '未知')}%。"
        )

    return "\n".join(lines)


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    if current_user_id:
        return current_user_id

    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


@tool(description="检索当前登录用户自己的长期对话记忆，以纯字符串形式返回；只能查询当前用户历史，不允许跨用户查询")
def search_user_memory(query: str) -> str:
    if not current_user_id:
        logger.warning("[search_user_memory]当前没有登录用户，无法检索长期记忆")
        return ""

    rows = search_conversation_memory(current_user_id, query)
    if not rows:
        logger.info(f"[search_user_memory]用户{current_user_id}未检索到相关长期记忆，query={query}")
        return ""

    return format_memory_rows(rows)


def generate_external_data():
    """
    {
        "user_id": {
            "month": [
                {"点位ID": xxx, "点位信息": xxx, "文物类型": xxx, ...},
                ...
            ]
        }
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id: str = row["用户ID"].strip()
                month: str = row["时间"].strip()

                if user_id not in external_data:
                    external_data[user_id] = {}

                if month not in external_data[user_id]:
                    external_data[user_id][month] = []

                external_data[user_id][month].append({
                    "点位ID": row.get("点位ID", "").strip(),
                    "点位信息": row.get("点位信息", "").strip(),
                    "文物类型": row.get("文物类型", "").strip(),
                    "巡检情况": row.get("巡检情况", "").strip(),
                    "风险问题": row.get("风险问题", "").strip(),
                    "整改建议": row.get("整改建议", "").strip(),
                    "时间": month,
                })


@tool(description="从外部系统中获取指定用户在指定月份、指定点位或文物类型的巡检/风险记录，以纯字符串形式返回；site_name为空时返回该用户当月全部记录，未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str, site_name: str = "") -> str:
    generate_external_data()

    try:
        records = external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的文物巡检记录数据")
        return ""

    site_name = site_name.strip()
    if site_name:
        records = [
            record for record in records
            if site_name in record["点位ID"]
            or site_name in record["点位信息"]
            or site_name in record["文物类型"]
        ]

    if not records:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}关于{site_name}的文物巡检记录数据")
        return ""

    return json.dumps(records, ensure_ascii=False, indent=2)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"

