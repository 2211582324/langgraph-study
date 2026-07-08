# 1.定义全局状态
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str      # 经过LLM理解后的用户需求总结
    search_query: str    # 优化后用于Tavily API的搜索查询
    search_results: str  # Tavily搜索返回的结果
    final_answer: str    # 最终生成的答案
    step: str            # 标记当前步骤

# 2.定义工作流节点
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from tavily import TavilyClient

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化模型
# 我们将使用这个 llm 实例来驱动所有节点的智能
llm = ChatOpenAI(
    model="gpt-5.5",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.ccb233.cn/v1"),
    temperature=0.7
)
# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# 3.创建核心节点
# 3.1 理解与查询节点
def understand_query_node(state: SearchState) -> dict:
    """步骤1：理解用户查询并生成搜索关键词"""
    user_message = state["messages"][-1].content

    understand_prompt = f"""分析用户的查询："{user_message}"
请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    response = llm.invoke([SystemMessage(content=understand_prompt)])
    response_text = response.content

    # 解析LLM的输出，提取搜索关键词
    search_query = user_message  # 默认使用原始查询
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()

    return {
        "user_query": response_text,
        "search_query": search_query,
        "step": "understood",
        "messages": [AIMessage(content=f"我将为您搜索：{search_query}")]
    }

#3.2 搜索节点
def tavily_search_node(state: SearchState) -> dict:
    """步骤2：使用Tavily API进行真实搜索"""
    search_query = state["search_query"]
    try:
        print(f"🔍 正在搜索: {search_query}")
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )
        # ... (处理和格式化搜索结果) ...
        # --- 替换开始：提取并格式化 Tavily 的搜索结果 ---
        search_results_list = []

        # 1. 优先尝试获取 Tavily 智能提炼出的直接回答
        if isinstance(response, dict) and response.get("answer"):
            search_results_list.append(f"【智能总结】: {response['answer']}\n")

        # 2. 遍历网页搜索结果，提取标题、内容片段和URL，拼接成可读文本
        if isinstance(response, dict) and "results" in response:
            search_results_list.append("【详细参考来源】:")
            for i, item in enumerate(response["results"], 1):
                title = item.get("title", "未知标题")
                content = item.get("content", "无内容介绍")
                url = item.get("url", "#")
                search_results_list.append(f"[{i}] {title}\n   内容: {content}\n   链接: {url}")

        # 3. 将所有部分用换行符连接起来。如果上面两步都没取到，安全兜底直接转为字符串
        search_results = "\n".join(search_results_list) if search_results_list else str(response)
        # --- 替换结束 ---

        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content="✅ 搜索完成！正在整理答案...")]
        }
    except Exception as e:
        # ... (处理错误) ...
        return {
            "search_results": f"搜索失败：{e}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题...")]
        }

# 3.3 回答节点
def generate_answer_node(state: SearchState) -> dict:
    """步骤3：基于搜索结果生成最终答案"""
    if state["step"] == "search_failed":
        # 如果搜索失败，执行回退策略，基于LLM自身知识回答
        fallback_prompt = f"搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：{state['user_query']}"
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
    else:
        # 搜索成功，基于搜索结果生成答案
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：
用户问题：{state['user_query']}
搜索结果：\n{state['search_results']}
请综合搜索结果，提供准确、有用的回答..."""
        response = llm.invoke([SystemMessage(content=answer_prompt)])

    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)]
    }

# 4.构建图
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


def create_search_assistant():
    workflow = StateGraph(SearchState)

    # 添加节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)

    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)

    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


if __name__ == "__main__":
    app = create_search_assistant()

    # 模拟用户提问
    user_input = "帮我查一下LangGraph相比普通LangChain有哪些核心优势？"

    print(f"用户提问: {user_input}")

    # 初始化输入状态（传入一条 HumanMessage）
    from langchain_core.messages import HumanMessage

    inputs = {
        "messages": [HumanMessage(content=user_input)]
    }

    # 配置 thread_id 以便维持对话上下文
    config = {"configurable": {"thread_id": "user_session_1"}}

    # 使用流式（stream）运行，观察图的每一步变化
    for event in app.stream(inputs, config=config):
        # event 是一个字典，键为当前执行的节点名称，值为该节点返回的状态增量
        for node_name, state_update in event.items():
            print(f"==> 节点【{node_name}】执行完毕")
            if "final_answer" in state_update:
                print("\n【最终回答】:")
                print(state_update["final_answer"])