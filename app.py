import time

import streamlit as st
from agent.react_agent import ReactAgent
import agent.tools.agent_tools as agent_tools
from rag.rag_service import RagSummarizeService
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.auth_handler import register_user, verify_user
from utils.memory_handler import save_conversation_memory

# 标题
st.title("智慧文保安全规范智能问答与风险预警助手")
st.divider()


def refresh_agent():
    agent_tools.rag = RagSummarizeService()
    st.session_state["agent"] = ReactAgent()


def rebuild_knowledge_base(vector_store: VectorStoreService):
    result = vector_store.rebuild_vector_store()
    refresh_agent()
    return result


def show_rebuild_result(result, success_title: str):
    message = (
        f"{success_title}：加载 {result['loaded']} 个文件，"
        f"写入 {result.get('chunks', 0)} 个知识片段，"
        f"跳过 {result['skipped']} 个，失败 {result['failed']} 个"
    )

    if result["failed"] > 0:
        st.error(message)
    elif result["loaded"] == 0:
        st.warning(message)
    else:
        st.success(message)

    details = result.get("details", [])
    if details:
        with st.expander("查看知识库构建详情"):
            for detail in details:
                st.write(detail)


if "login_user_id" not in st.session_state:
    st.session_state["login_user_id"] = None

if not st.session_state["login_user_id"]:
    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        with st.form("login_form"):
            user_id = st.text_input("用户ID")
            password = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", use_container_width=True)

            if submitted:
                if verify_user(user_id, password):
                    st.session_state["login_user_id"] = user_id.strip()
                    agent_tools.set_current_user_id(user_id.strip())
                    st.success("登录成功")
                    st.rerun()
                else:
                    st.error("用户ID或密码错误")

    with register_tab:
        with st.form("register_form"):
            new_user_id = st.text_input("注册用户ID")
            new_password = st.text_input("注册密码", type="password")
            confirm_password = st.text_input("确认密码", type="password")
            submitted = st.form_submit_button("注册", use_container_width=True)

            if submitted:
                if new_password != confirm_password:
                    st.error("两次输入的密码不一致")
                else:
                    try:
                        register_user(new_user_id, new_password)
                        st.success("注册成功，请返回登录页登录")
                    except Exception as e:
                        st.error(f"注册失败：{str(e)}")

    st.stop()


agent_tools.set_current_user_id(st.session_state["login_user_id"])
vector_store_service = VectorStoreService()

with st.sidebar:
    st.caption(f"当前用户：{st.session_state['login_user_id']}")
    if st.button("退出登录", use_container_width=True):
        st.session_state["login_user_id"] = None
        st.session_state.pop("agent", None)
        st.session_state.pop("message", None)
        agent_tools.set_current_user_id(None)
        st.rerun()

    st.divider()
    st.header("知识库管理")
    knowledge_files = vector_store_service.list_knowledge_files()

    if knowledge_files:
        st.caption(f"当前知识库文件：{len(knowledge_files)} 个")
        for filename in knowledge_files:
            st.write(f"- {filename}")
    else:
        st.caption("当前没有可用知识库文件")

    uploaded_files = st.file_uploader(
        "上传知识库文件",
        type=chroma_conf["allow_knowledge_file_type"],
        accept_multiple_files=True,
    )

    if st.button("保存上传并重建索引", use_container_width=True):
        if not uploaded_files:
            st.warning("请先选择要上传的文件")
        else:
            try:
                for uploaded_file in uploaded_files:
                    vector_store_service.save_uploaded_file(uploaded_file.name, uploaded_file.getvalue())

                result = rebuild_knowledge_base(vector_store_service)
                show_rebuild_result(result, "知识库已更新")
            except Exception as e:
                st.error(f"上传或重建失败：{str(e)}")

    file_for_delete = st.selectbox(
        "选择要删除的文件",
        options=knowledge_files,
        index=None,
        placeholder="选择文件",
    )

    if st.button("删除选中文件并重建索引", use_container_width=True):
        if not file_for_delete:
            st.warning("请先选择要删除的文件")
        else:
            try:
                vector_store_service.delete_knowledge_file(file_for_delete)
                result = rebuild_knowledge_base(vector_store_service)
                show_rebuild_result(result, "已删除并重建")
            except Exception as e:
                st.error(f"删除或重建失败：{str(e)}")

    if st.button("仅重建索引", use_container_width=True):
        try:
            result = rebuild_knowledge_base(vector_store_service)
            show_rebuild_result(result, "索引已重建")
        except Exception as e:
            st.error(f"重建索引失败：{str(e)}")

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)


        def capture(generator, cache_list):

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char


        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        assistant_answer = response_messages[-1] if response_messages else ""
        st.session_state["message"].append({"role": "assistant", "content": assistant_answer})
        try:
            save_conversation_memory(st.session_state["login_user_id"], prompt, assistant_answer)
        except Exception as e:
            st.warning(f"长期记忆保存失败：{str(e)}")
        st.rerun()
