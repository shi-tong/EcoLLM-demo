"""
Reasoning 生成辅助函数
用于动态构建 Prompt 和处理 Tool Response
"""

import json
import re
from typing import Dict, List, Any


def summarize_tool_response(tool_name: str, content: str) -> str:
    """
    简化 tool response，提取关键信息（显示完整信息）
    
    Args:
        tool_name: 工具名称
        content: tool response 的完整内容
        
    Returns:
        简化后的摘要
    """
    try:
        data = json.loads(content)
    except:
        # 如果不是 JSON，直接返回前 500 字符
        return content[:500]
    
    if tool_name == "search_document":
        results = data.get("results", [])
        if not results:
            return "No results found"
        
        # 🔥 显示全部 chunks（不截断）
        summaries = []
        for result in results:
            chunk_id = result.get("chunk_id", "?")
            content_text = result.get("content", "")
            # 🔥 显示完整内容（不截断）
            summaries.append(f"  - Chunk {chunk_id}: {content_text}")
        
        return f"Found {len(results)} chunks:\n" + "\n".join(summaries)
    
    elif tool_name == "process_document":
        session_id = data.get("session_id", "unknown")
        return f"Document processed successfully, session: {session_id}"
    
    elif tool_name == "record_process_flow":
        action_id = data.get("data", {}).get("action_id", "unknown")
        return f"Flow recorded successfully, action_id: {action_id}"
    
    elif tool_name == "define_lca_scope":
        action_id = data.get("data", {}).get("action_id", "unknown")
        param_name = data.get("data", {}).get("parameter_name", "unknown")
        return f"Scope defined: {param_name}, action_id: {action_id}"
    
    elif tool_name == "get_session_summary":
        completeness = data.get("completeness_score", 0)
        missing = data.get("missing_categories", [])
        return f"Completeness: {completeness}%, Missing: {missing}"
    
    # 默认：返回前 500 字符
    return str(data)[:500]


def describe_next_action(tool_call: Dict) -> str:
    """
    描述下一个要执行的动作（显示完整信息）
    
    Args:
        tool_call: tool_calls 数组中的一个元素
        
    Returns:
        动作描述
    """
    tool_name = tool_call.get("name")
    args = tool_call.get("arguments", {})
    
    if tool_name == "search_document":
        queries = args.get("queries", [])
        # 🔥 显示全部 queries（不截断）
        return f"Search the document for: {', '.join(queries)}"
    
    elif tool_name == "record_process_flow":
        category = args.get("category")
        name = args.get("name")
        value = args.get("value")
        unit = args.get("unit")
        flow_type = args.get("flow_type", "")
        return f"Record {flow_type} flow - {category}: {name} ({value} {unit})"
    
    elif tool_name == "define_lca_scope":
        param_name = args.get("parameter_name")
        description = args.get("description", "")
        return f"Define LCA scope parameter: {param_name} ({description})"
    
    elif tool_name == "get_session_summary":
        return "Check the session summary to assess extraction progress"
    
    elif tool_name == "record_parameter":
        param_name = args.get("parameter_name")
        value = args.get("value")
        unit = args.get("unit", "")
        return f"Record parameter: {param_name} = {value} {unit}"
    
    elif tool_name == "execute_calculation":
        formula = args.get("formula", "")
        return f"Execute calculation: {formula}"
    
    return f"Call tool: {tool_name}"


def build_conversation_history(previous_messages: List[Dict]) -> str:
    """
    构建对话历史摘要（Agent 可见的部分）
    
    Args:
        previous_messages: 当前位置之前的所有 messages
        
    Returns:
        对话历史的文本摘要
    """
    history_lines = []
    
    for msg in previous_messages:
        role = msg.get("role")
        
        if role == "user":
            content = msg.get("content", "")
            if content:
                # 只显示前 200 字符（避免过长）
                content_preview = content[:200] + "..." if len(content) > 200 else content
                history_lines.append(f"**User**: {content_preview}")
        
        elif role == "tool":
            # Tool Response
            tool_name = msg.get("name", "unknown")
            content = msg.get("content", "")
            
            # 简化 tool response
            summary = summarize_tool_response(tool_name, content)
            history_lines.append(f"**Tool Response ({tool_name})**:\n{summary}")
        
        elif role == "assistant":
            # 前面的 Assistant 消息
            reasoning = msg.get("reasoning_content", "")
            tool_calls = msg.get("tool_calls", [])
            
            if reasoning:
                # 清理 placeholder
                reasoning_clean = re.sub(
                    r'\[SMART_SKIP_PLACEHOLDER:.*?\]', '', reasoning
                ).strip()
                if reasoning_clean:
                    history_lines.append(f"**Your Previous Reasoning**: {reasoning_clean}")
            
            if tool_calls:
                for tc in tool_calls:
                    tool_desc = describe_next_action(tc)
                    history_lines.append(f"**Your Previous Action**: {tool_desc}")
    
    return "\n\n".join(history_lines) if history_lines else "(No previous actions)"


def build_dynamic_prompt(
    previous_messages: List[Dict],
    current_tool_call: Dict,
    user_query: str = ""
) -> str:
    """
    动态构建 Prompt（核心函数）
    
    Args:
        previous_messages: 当前位置之前的所有 messages
        current_tool_call: 当前要执行的 tool_call
        user_query: 用户的初始请求（可选）
        
    Returns:
        完整的 prompt
    """
    # 1. 构建对话历史
    history_text = build_conversation_history(previous_messages)
    
    # 2. 描述下一个动作
    next_action_text = describe_next_action(current_tool_call)
    
    # 🔥 检测需要评价搜索结果的场景
    need_search_evaluation = False
    evaluation_context = ""
    
    if previous_messages and len(previous_messages) >= 2:
        last_msg = previous_messages[-1]
        second_last_msg = previous_messages[-2]
        
        # 如果最后一条是 tool (search_document)，倒数第二条是 assistant (search_document)
        if (last_msg.get("role") == "tool" and 
            second_last_msg.get("role") == "assistant" and
            second_last_msg.get("tool_calls", [{}])[0].get("name") == "search_document"):
            
            # 场景 1: 连续搜索（Verification/Pivot）
            if current_tool_call.get("name") == "search_document":
                need_search_evaluation = True
                evaluation_context = "search_to_search"
            
            # 场景 2: 搜索后检查进度（Search → Summary）
            elif current_tool_call.get("name") == "get_session_summary":
                need_search_evaluation = True
                evaluation_context = "search_to_summary"
            
            # 场景 3: 搜索后记录（Search → Record）
            elif current_tool_call.get("name") in ["record_process_flow", "record_parameter", "define_lca_scope"]:
                need_search_evaluation = True
                evaluation_context = "search_to_record"
    
    # 3. 组装 prompt
    prompt_parts = []
    
    # 添加用户请求（如果有且是前几条消息）
    if user_query and len(previous_messages) <= 3:
        prompt_parts.append(f"**Initial Task**: {user_query}")
        prompt_parts.append("")
    
    # 添加对话历史
    prompt_parts.append("## Conversation So Far:")
    prompt_parts.append(history_text)
    prompt_parts.append("")
    
    # 添加下一步动作
    prompt_parts.append("## Next Action You Will Take:")
    prompt_parts.append(next_action_text)
    prompt_parts.append("")
    
    # 添加任务说明
    prompt_parts.append("## Your Task:")
    
    # 🔥 需要评价搜索结果的场景：特殊引导
    if need_search_evaluation:
        if evaluation_context == "search_to_search":
            # 连续搜索（Verification/Pivot）
            prompt_parts.append("Generate your internal reasoning that:")
            prompt_parts.append("1. Briefly comments on the search results you just received")
            prompt_parts.append("2. Explains why you need to search for something else")
            prompt_parts.append("")
            prompt_parts.append("Be natural and vary your expression.")
        
        elif evaluation_context == "search_to_summary":
            # 搜索后检查进度
            prompt_parts.append("Generate your internal reasoning that:")
            prompt_parts.append("1. Briefly comments on the search results you just received")
            prompt_parts.append("2. Explains why you want to check the session summary now")
            prompt_parts.append("")
            prompt_parts.append("Be natural and vary your expression.")
        
        elif evaluation_context == "search_to_record":
            # 搜索后记录
            prompt_parts.append("Generate your internal reasoning that:")
            prompt_parts.append("1. Briefly comments on the search results you just received")
            prompt_parts.append("2. Explains what you're going to record based on these results")
            prompt_parts.append("")
            prompt_parts.append("Be natural and vary your expression.")
    else:
        prompt_parts.append("Generate your internal reasoning explaining WHY you chose this action.")
    
    prompt_parts.append("")
    prompt_parts.append("**Remember**:")
    prompt_parts.append("- Write in first person (\"I need to...\", \"I found...\")")
    prompt_parts.append("- Be natural and concise (50-180 characters)")
    prompt_parts.append("- Reference previous actions if relevant")
    prompt_parts.append("- Focus on your thought process, not the action details")
    prompt_parts.append("")
    prompt_parts.append("Generate only the reasoning:")
    
    return "\n".join(prompt_parts)
