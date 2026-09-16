from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf, rag_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os

try:
    import jieba
    from rank_bm25 import BM25Okapi
except ImportError:
    jieba = None
    BM25Okapi = None

try:
    import dashscope
except ImportError:
    dashscope = None


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )
        #文本分割器
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )
        self.bm25_documents = self._load_split_documents()
        self.bm25_index = self._build_bm25_index(self.bm25_documents)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def _get_file_documents(self, read_path: str):
        if read_path.endswith(".txt"):
            return txt_loader(read_path)

        if read_path.endswith(".pdf"):
            return pdf_loader(read_path)

        return []

    def _get_allowed_files_path(self) -> tuple[str]:
        return listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

    def _load_split_documents(self) -> list[Document]:
        split_documents = []

        for path in self._get_allowed_files_path():
            try:
                documents = self._get_file_documents(path)
                split_documents.extend(self.spliter.split_documents(documents))
            except Exception as e:
                logger.warning(f"[BM25索引]{path}加载失败：{str(e)}")

        return split_documents

    def _tokenize(self, text: str) -> list[str]:
        if jieba is None:
            return list(text)

        return [token for token in jieba.lcut(text) if token.strip()]

    def _build_bm25_index(self, documents: list[Document]):
        if BM25Okapi is None or not documents:
            return None

        corpus = [self._tokenize(doc.page_content) for doc in documents]
        return BM25Okapi(corpus)

    def _refresh_bm25_index(self):
        self.bm25_documents = self._load_split_documents()
        self.bm25_index = self._build_bm25_index(self.bm25_documents)

    def vector_search(self, query: str, top_k: int = None) -> list[Document]:
        top_k = top_k or chroma_conf.get("vector_top_k", chroma_conf["k"])
        documents = self.vector_store.similarity_search(query, k=top_k)
        logger.info(f"[向量检索]query={query}，召回{len(documents)}条")
        return documents

    def bm25_search(self, query: str, top_k: int = None) -> list[Document]:
        if self.bm25_index is None:
            logger.warning("[BM25检索]jieba或rank_bm25未安装，跳过关键词检索")
            return []

        top_k = top_k or chroma_conf.get("bm25_top_k", chroma_conf["k"])
        scores = self.bm25_index.get_scores(self._tokenize(query))
        ranked_indexes = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        return [
            self.bm25_documents[idx]
            for idx in ranked_indexes[:top_k]
            if scores[idx] > 0
        ]

    def _document_key(self, doc: Document) -> tuple[str, str]:
        return doc.page_content, str(doc.metadata)

    def rrf_fusion(self, vector_docs: list[Document], bm25_docs: list[Document]) -> list[Document]:
        rrf_k = chroma_conf.get("rrf_k", 60)
        bm25_weight = chroma_conf.get("bm25_weight", 0.25)
        doc_map = {}
        scores = {}

        def add_scores(docs: list[Document], weight: float):
            for rank, doc in enumerate(docs, start=1):
                key = self._document_key(doc)
                doc_map[key] = doc
                scores[key] = scores.get(key, 0) + weight / (rrf_k + rank)

        add_scores(vector_docs, 1.0)
        add_scores(bm25_docs, bm25_weight)

        ranked_keys = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[key] for key in ranked_keys]

    def rerank(self, query: str, documents: list[Document], top_n: int = None) -> list[Document]:
        top_n = top_n or chroma_conf.get("rerank_top_n", chroma_conf["k"])

        if not chroma_conf.get("enable_rerank", True):
            return documents[:top_n]

        if dashscope is None:
            logger.warning("[Rerank]dashscope未安装，跳过gte-rerank-v2精排")
            return documents[:top_n]

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("[Rerank]未配置DASHSCOPE_API_KEY，跳过gte-rerank-v2精排")
            return documents[:top_n]

        try:
            dashscope.api_key = api_key
            response = dashscope.TextReRank.call(
                model=rag_conf.get("rerank_model_name", "gte-rerank-v2"),
                query=query,
                documents=[doc.page_content for doc in documents],
                top_n=top_n,
                return_documents=False,
            )

            results = response.output["results"]
            return [documents[item["index"]] for item in results]
        except Exception as e:
            logger.warning(f"[Rerank]gte-rerank-v2调用失败，降级为RRF结果：{str(e)}")
            return documents[:top_n]

    def hybrid_search(self, query: str, final_k: int = None, use_rerank: bool = True) -> list[Document]:
        final_k = final_k or chroma_conf["k"]
        recall_k = max(final_k, chroma_conf.get("vector_top_k", final_k), chroma_conf.get("bm25_top_k", final_k))
        vector_docs = self.vector_search(query, top_k=recall_k)
        bm25_docs = self.bm25_search(query, top_k=recall_k)
        logger.info(f"[BM25检索]query={query}，召回{len(bm25_docs)}条")
        fused_docs = self.rrf_fusion(vector_docs, bm25_docs)
        logger.info(f"[RRF融合]向量{len(vector_docs)}条，BM25 {len(bm25_docs)}条，融合后{len(fused_docs)}条")

        if not fused_docs:
            return []

        if use_rerank:
            return self.rerank(query, fused_docs, top_n=final_k)

        return fused_docs[:final_k]

    def list_knowledge_files(self) -> list[str]:
        data_path = get_abs_path(chroma_conf["data_path"])
        allowed_types = tuple(chroma_conf["allow_knowledge_file_type"])
        if not os.path.isdir(data_path):
            os.makedirs(data_path, exist_ok=True)
            return []

        return sorted(
            f for f in os.listdir(data_path)
            if os.path.isfile(os.path.join(data_path, f)) and f.endswith(allowed_types)
        )

    def save_uploaded_file(self, filename: str, content: bytes) -> str:
        data_path = get_abs_path(chroma_conf["data_path"])
        os.makedirs(data_path, exist_ok=True)

        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith(tuple(chroma_conf["allow_knowledge_file_type"])):
            raise ValueError(f"仅支持上传：{', '.join(chroma_conf['allow_knowledge_file_type'])}")

        target_path = os.path.join(data_path, safe_filename)
        with open(target_path, "wb") as f:
            f.write(content)

        return target_path

    def delete_knowledge_file(self, filename: str) -> str:
        data_path = get_abs_path(chroma_conf["data_path"])
        target_path = os.path.abspath(os.path.join(data_path, os.path.basename(filename)))
        data_root = os.path.abspath(data_path)

        if os.path.commonpath([data_root, target_path]) != data_root:
            raise ValueError("文件路径不合法")

        if not os.path.isfile(target_path):
            raise FileNotFoundError(f"知识库文件不存在：{filename}")

        os.remove(target_path)
        return target_path

    def reset_vector_store(self):
        try:
            self.vector_store.delete_collection()
        except Exception as e:
            logger.warning(f"[重建知识库]删除旧集合失败，将继续尝试重新创建集合：{str(e)}")

        md5_store = get_abs_path(chroma_conf["md5_hex_store"])
        if os.path.exists(md5_store):
            os.remove(md5_store)

        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )
        self._refresh_bm25_index()

    def rebuild_vector_store(self):
        self.reset_vector_store()
        return self.load_document()

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False  # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True  # md5 处理过

                return False  # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        allowed_files_path: list[str] = self._get_allowed_files_path()

        result = {
            "loaded": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0,
            "details": [],
        }

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                result["skipped"] += 1
                result["details"].append(f"跳过：{os.path.basename(path)}，原因：MD5已存在")
                continue

            try:
                documents: list[Document] = self._get_file_documents(path)
                logger.info(f"[加载知识库]{path}读取到{len(documents)}个原始文档片段")

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    result["skipped"] += 1
                    result["details"].append(f"跳过：{os.path.basename(path)}，原因：未读取到有效文本")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                logger.info(f"[加载知识库]{path}切分得到{len(split_document)}个知识片段")

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    result["skipped"] += 1
                    result["details"].append(f"跳过：{os.path.basename(path)}，原因：切分后无有效片段")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
                result["loaded"] += 1
                result["chunks"] += len(split_document)
                result["details"].append(f"成功：{os.path.basename(path)}，写入{len(split_document)}个知识片段")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                result["failed"] += 1
                result["details"].append(f"失败：{os.path.basename(path)}，原因：{str(e)}")
                continue

        self._refresh_bm25_index()
        return result


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    res = vs.hybrid_search("火灾检查")
    for r in res:
        print(r.page_content)
        print("-" * 20)
