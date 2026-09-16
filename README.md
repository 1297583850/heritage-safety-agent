<div align="center">

# 智慧文保安全规范智能问答与风险预警助手

面向文物信息化保护与文物安防场景的 RAG Agent 原型系统

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Agent-1C3C3C?style=flat-square)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Vector_DB-Chroma-6B4EFF?style=flat-square)](https://www.trychroma.com/)
[![RAG](https://img.shields.io/badge/RAG-Hybrid_Retrieval-0A7B83?style=flat-square)](#检索评估)

</div>

---

## 项目简介

本项目面向文物信息化保护、博物馆安全管理和不可移动文物安防场景，构建了一个基于本地知识库的智能问答与风险辅助分析系统。

系统以文物保护规范、火灾风险检查指引、藏品防护资料等文档为知识库，结合 LangChain ReAct Agent、Chroma 向量检索、BM25 关键词检索、RRF 融合排序和 Rerank 精排，实现文物安防知识问答、巡检记录查询、风险报告生成、网页端知识库管理和检索效果评估。

当前项目定位为原型系统，适合用于展示 RAG 检索增强、Agent 工具调用、结构化业务数据查询和测试评估能力。

---

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 文物安防知识问答 | 基于本地文物保护资料回答消防安全、藏品防护、风险源识别、整改建议等问题 |
| 混合检索增强 | Chroma 向量检索 + BM25 关键词检索 + RRF 融合，兼顾语义召回和精确词匹配 |
| Rerank 精排 | 使用 `gte-rerank-v2` 对候选文档二次排序，提升首条命中质量 |
| 知识库管理 | 网页端上传、删除 `txt/pdf` 文档，并支持一键重建索引 |
| 用户登录 | 支持本地用户注册、登录和退出，基于用户 ID 绑定业务数据 |
| 巡检数据查询 | 根据用户 ID、月份、点位或文物类型查询本地结构化巡检记录 |
| 风险报告生成 | 结合巡检记录和知识库依据，生成文物安全检查、风险研判和整改建议报告 |
| 天气环境分析 | 调用心知天气 API 获取温度、湿度、降水概率等环境信息，辅助文物风险分析 |
| 长期记忆实验 | 将部分对话摘要写入本地 CSV，并按当前登录用户 ID 隔离检索 |
| 检索效果评估 | 使用 25 道文物安防测试题评估 Recall@1/3/5，支持三组消融对比 |

---

## 系统架构

```text
┌──────────────────────────────────────────────┐
│                 Streamlit Web 页面            │
│  登录注册 / 知识库管理 / 对话交互 / 报告生成     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              LangChain ReAct Agent            │
│  意图判断 / 工具调用 / 动态提示词 / 流式输出      │
└───────┬──────────────┬──────────────┬────────┘
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ RAG 知识库    │ │ 结构化巡检数据 │ │ 外部环境工具   │
│ Chroma       │ │ records.csv   │ │ Weather API  │
│ BM25 + RRF   │ │ user_id过滤   │ │ 城市天气查询   │
│ Rerank       │ │ 点位/月查询    │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────────────────────────────────────┐
│        文物安全问答 / 巡检分析 / 风险报告       │
└──────────────────────────────────────────────┘
```

---

## 技术栈

| 模块 | 技术 |
| --- | --- |
| Web 页面 | Streamlit |
| Agent 框架 | LangChain `create_agent` |
| 大语言模型 | Qwen `qwen3-max` |
| Embedding 模型 | `text-embedding-v3` |
| 向量数据库 | Chroma |
| 关键词检索 | jieba + rank_bm25 |
| 检索融合 | RRF，`bm25_weight=0.75` |
| 精排模型 | gte-rerank-v2 |
| 文档解析 | TextLoader / PyPDFLoader |
| 文本切分 | RecursiveCharacterTextSplitter |
| 用户认证 | 本地 JSON + SHA256 加盐哈希 |
| 业务数据 | CSV 巡检记录 |
| 长期记忆 | CSV 摘要记忆 |
| 天气接口 | 心知天气 API |
| 配置管理 | YAML |
| 日志记录 | Python logging |
| 评估指标 | Recall@1 / Recall@3 / Recall@5 |

---

## 检索评估

项目构建了 25 道文物安防领域测试题，覆盖博物馆火灾风险、文物建筑消防安全、藏品防护、木质文物保护、泥质佛像雕塑保护、文物保护原则与保护措施等知识库内容。

在 `bm25_weight=0.75` 配置下，三种检索模式评估结果如下：

| 检索模式 | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: |
| vector | 64.00% | 76.00% | 92.00% |
| hybrid | 72.00% | 84.00% | 88.00% |
| hybrid_rerank | 80.00% | 88.00% | 92.00% |

结果说明：

- 混合检索相比纯向量检索提升了 Recall@1 和 Recall@3，说明 BM25 对专业术语和精确关键词有补充作用。
- `hybrid_rerank` 的 Recall@1 达到 80.00%，比纯向量检索提升 16 个百分点，说明 Rerank 对首条排序质量提升明显。
- Recall@5 已达到 92.00%，说明多数正确资料已经能够进入候选集合，后续优化重点主要是排序质量和查询改写。

运行评估：

```bash
# 纯向量检索
python evaluate_retrieval.py --mode vector

# 向量检索 + BM25 + RRF
python evaluate_retrieval.py --mode hybrid

# 向量检索 + BM25 + RRF + Rerank
python evaluate_retrieval.py --mode hybrid_rerank
```

---

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repository-url>
cd 扫地机器人agent实战
```

### 2. 安装依赖

```bash
pip install streamlit langchain langchain-core langchain-community langchain-chroma chromadb langchain-text-splitters pypdf pyyaml jieba rank_bm25 dashscope
```

### 3. 配置 API Key

模型、Embedding 和 Rerank 使用 DashScope：

```powershell
$env:DASHSCOPE_API_KEY="你的DashScope API Key"
```

天气查询使用心知天气：

```powershell
$env:SENIVERSE_API_KEY="你的心知天气 API Key"
```

也可以使用 Windows 永久环境变量：

```powershell
setx DASHSCOPE_API_KEY "你的DashScope API Key"
setx SENIVERSE_API_KEY "你的心知天气 API Key"
```

### 4. 启动应用

```bash
streamlit run app.py
```

启动后在浏览器访问：

```text
http://localhost:8501
```

首次使用流程：

```text
注册用户 -> 登录 -> 上传或重建知识库 -> 开始对话
```

---

## 配置说明

### 模型配置

`config/rag.yml`

```yaml
chat_model_name: qwen3-max
embedding_model_name: text-embedding-v3
rerank_model_name: gte-rerank-v2
```

### 检索配置

`config/chroma.yml`

```yaml
k: 3
vector_top_k: 8
bm25_top_k: 8
rrf_k: 60
bm25_weight: 0.75
rerank_top_n: 3
enable_rerank: true
data_path: data
allow_knowledge_file_type: ["txt", "pdf"]
```

如需关闭在线 Rerank：

```yaml
enable_rerank: false
```

### 业务数据与天气配置

`config/agent.yml`

```yaml
external_data_path: data/external/records.csv
weather_api_url: https://api.seniverse.com/v3/weather/daily.json
weather_api_key: ""
weather_language: zh-Hans
weather_unit: c
weather_start: -1
weather_days: 5
```

---

## 项目结构

```text
.
├── app.py                          # Streamlit 主页面
├── evaluate_retrieval.py            # 检索评估脚本
├── test_set_heritage_fire.json      # 25 道文物安防测试题
├── agent/
│   ├── react_agent.py               # ReAct Agent 创建与执行
│   └── tools/
│       ├── agent_tools.py           # Agent 工具函数
│       └── middleware.py            # 工具监控、日志、提示词切换
├── rag/
│   ├── rag_service.py               # RAG 总结链路
│   └── vector_store.py              # 向量检索、BM25、RRF、Rerank、索引管理
├── model/
│   └── factory.py                   # LLM 与 Embedding 模型工厂
├── utils/
│   ├── auth_handler.py              # 注册登录与密码哈希
│   ├── config_handler.py            # YAML 配置加载
│   ├── file_handler.py              # 文档加载与 MD5
│   ├── logger_handler.py            # 日志配置
│   ├── memory_handler.py            # 轻量长期记忆
│   ├── path_tool.py                 # 路径工具
│   └── prompt_loader.py             # 提示词加载
├── config/
│   ├── agent.yml                    # 工具与外部数据配置
│   ├── chroma.yml                   # 检索参数配置
│   ├── prompts.yml                  # 提示词路径配置
│   └── rag.yml                      # 模型配置
├── prompts/
│   ├── main_prompt.txt              # Agent 主提示词
│   ├── rag_summarize.txt            # RAG 总结提示词
│   └── report_prompt.txt            # 报告生成提示词
├── data/
│   ├── external/records.csv         # 模拟巡检记录
│   ├── memory/                      # 长期记忆 CSV
│   ├── users.json                   # 本地用户数据
│   └── *.txt                        # 文物保护知识库
├── chroma_db/                       # Chroma 持久化数据
└── logs/                            # 运行日志
```

---

## 知识库与数据

当前知识库示例：

- 博物馆火灾风险防范指南
- 博物馆火灾风险检查指引
- 文物建筑火灾风险防范指南
- 文物建筑火灾风险检查指引
- 藏品防护
- 木质文物保存与保护
- 泥质佛像雕塑文物保存与保护
- 文物保护原则
- 文物保护措施

结构化巡检数据位于：

```text
data/external/records.csv
```

主要字段：

```text
用户ID, 点位ID, 点位信息, 文物类型, 巡检情况, 风险问题, 整改建议, 时间
```

---

## 示例问题

```text
博物馆火灾风险现场实体抽查重点有哪些？
```

```text
帮我生成一份壁画文物安全检查、风险研判和整改建议报告。
```

```text
合肥未来几天湿度变化会不会影响文物建筑巡检？
```

```text
查询 2025 年 11 月泥质佛像展厅的巡检风险记录。
```

---

## 当前边界

- `get_user_location` 目前为模拟位置，不是真实定位。
- `get_current_month` 目前为模拟月份，后续可改为系统真实月份。
- `records.csv` 是本地模拟业务数据，真实项目中应接入巡检系统、传感器平台或业务数据库。
- 长期记忆为轻量实验版，仍需进一步优化记忆写入判断、记忆类型、数量限制和检索排序。
- `gte-rerank-v2` 与天气 API 为在线服务，涉及敏感数据时建议替换为本地模型或关闭相关能力。
- 系统输出仅供辅助分析，最终风险结论需由专业人员复核。

---

## 免责声明

本项目知识库和结构化巡检数据用于学习、测试和原型展示。系统生成的风险判断、整改建议和报告内容仅供辅助参考，不替代文物保护、消防安全或安防领域专业人员的现场核验与最终结论。
