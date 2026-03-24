"""
Qwen-Agent 服务 V2
真正使用 Qwen-Agent 框架，通过 vLLM API 进行推理

架构：
┌─────────────────────────────────────────┐
│           QwenAgentServiceV2            │
│  ┌─────────────────────────────────┐    │
│  │     Qwen-Agent Assistant        │    │
│  │   (对话管理、工具调用解析)        │    │
│  └─────────────────────────────────┘    │
│                   │                      │
│                   ▼                      │
│  ┌─────────────────────────────────┐    │
│  │     vLLM API (OpenAI 兼容)       │    │
│  │   (高性能推理引擎)               │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘

使用方式：
1. 先启动 vLLM: ./start_vllm.sh
2. 设置环境变量: LLM_SERVICE=qwen_agent
3. 启动后端: ./restart_services.sh
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 尝试导入 qwen-agent
try:
    from qwen_agent.agents import Assistant
    from qwen_agent.llm import get_chat_model
    from qwen_agent.tools.base import BaseTool, register_tool
    QWEN_AGENT_AVAILABLE = True
except ImportError:
    QWEN_AGENT_AVAILABLE = False
    logger.warning("qwen-agent 未安装，将使用简化模式")

# 导入我们的 LCA 工具
from backend.services.qwen_agent_tools import (
    set_tool_service, 
    get_lca_tool_names,
    QWEN_AGENT_AVAILABLE as TOOLS_AVAILABLE
)


class QwenAgentServiceV2:
    """
    基于 Qwen-Agent 框架的 LLM 服务
    
    通过 vLLM API 进行推理，使用 Qwen-Agent 管理对话和工具调用
    """
    
    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        api_key: str = "EMPTY",
        model_name: str = "qwen-lca",
        session_manager = None
    ):
        """
        初始化服务
        
        Args:
            api_base: vLLM API 地址
            api_key: API 密钥
            model_name: 模型名称
            session_manager: 会话管理器
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.session_manager = session_manager
        
        self.llm = None
        self.assistant = None
        self.is_initialized = False
        
        # PDF 处理器引用
        self.pdf_processor = None
        
        # 工具执行器（由 LLMChatService 设置）
        self.tool_executor = None
        
        logger.info(f"QwenAgentServiceV2 初始化，API: {self.api_base}")
        
    def set_pdf_processor(self, pdf_processor):
        """设置 PDF 处理器"""
        self.pdf_processor = pdf_processor
        
    def set_tool_executor(self, executor):
        """设置工具执行器"""
        self.tool_executor = executor
        
    def set_tool_service(self, tool_service):
        """设置工具服务（供 Qwen-Agent 工具使用）"""
        self._tool_service = tool_service
        logger.info("✅ QwenAgentServiceV2 已设置 tool_service")
        
    async def initialize(self):
        """异步初始化"""
        if self.is_initialized:
            return
            
        logger.info("🚀 初始化 Qwen-Agent 服务...")
        
        if not QWEN_AGENT_AVAILABLE:
            logger.warning("⚠️ qwen-agent 不可用，使用简化模式")
            self.is_initialized = True
            return
            
        try:
            # 配置 LLM（使用 vLLM API）
            # 🔥 模型支持 32k+ tokens，设置较大的限制
            llm_cfg = {
                'model': self.model_name,
                'model_server': self.api_base,
                'api_key': self.api_key,
                'generate_cfg': {
                    'max_input_tokens': 20000,  # 模型支持 32k+
                    'max_retries': 3,
                    # 🔥 Qwen3 官方推荐参数（thinking mode）
                    'temperature': 0.6,
                    'top_p': 0.95,
                    'top_k': 20,
                }
            }
            
            # 创建 LLM 实例
            self.llm = get_chat_model(llm_cfg)
            
            # 系统提示词
            system_prompt = self._build_system_prompt()
            
            # 🔥 创建 Assistant，配置 LCA 工具
            # Qwen-Agent 会自动管理工具调用流程
            tool_names = get_lca_tool_names()
            logger.info(f"📦 配置 Qwen-Agent 工具: {tool_names}")
            
            self.assistant = Assistant(
                llm=llm_cfg,
                system_message=system_prompt,
                name='LCA-Assistant',
                description='LCA data extraction assistant',
                function_list=tool_names  # 注册我们的 LCA 工具
            )
            
            logger.info("✅ Qwen-Agent 初始化成功")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Qwen-Agent 初始化失败: {e}")
            # 降级到简化模式
            self.is_initialized = True
    
    def _sync_initialize(self):
        """同步初始化（用于在事件循环中调用）"""
        if self.is_initialized:
            return
            
        logger.info("🚀 同步初始化 Qwen-Agent 服务...")
        
        if not QWEN_AGENT_AVAILABLE:
            logger.warning("⚠️ qwen-agent 不可用，使用简化模式")
            self.is_initialized = True
            return
            
        try:
            llm_cfg = {
                'model': self.model_name,
                'model_server': self.api_base,
                'api_key': self.api_key,
                'generate_cfg': {
                    'max_input_tokens': 20000,
                    'max_retries': 3,
                    'top_p': 0.8,
                }
            }
            
            self.llm = get_chat_model(llm_cfg)
            system_prompt = self._build_system_prompt()
            tool_names = get_lca_tool_names()
            
            self.assistant = Assistant(
                llm=llm_cfg,
                system_message=system_prompt,
                name='LCA-Assistant',
                description='LCA data extraction assistant',
                function_list=tool_names
            )
            
            logger.info("✅ Qwen-Agent 同步初始化成功")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Qwen-Agent 同步初始化失败: {e}")
            self.is_initialized = True
            
    def _build_system_prompt(self, pdf_session_id: str = None, doc_context: str = None) -> str:
        """构建系统提示词 - 与微调数据格式一致"""
        
        # 🔥 使用与微调数据完全一致的 system prompt
        prompt = """You are an expert LCA assistant for Additive Manufacturing.

## Core Task
Extract quantitative LCI data from documents or answer specific questions.
When asked to extract LCI data, proactively search and record all available data without waiting for clarification.

## Tools
- search_document: Search text segments containing data via keywords
- define_lca_scope: Record Functional Unit
- record_process_flow: Record LCI flows (quantitative values)
- record_parameter: Record intermediate parameters for calculation
- execute_calculation: Calculate derived values
- get_session_summary: Check recorded data

## Strategic Workflow
1. **Anchor (Functional Unit)**: Identify the study basis (e.g., "manufacturing a 316L impeller by SLM"). Record via `define_lca_scope`.
2. **Inputs**: Extract Material, Energy, Gas, Cooling. Use `record_process_flow` or `record_parameter`.
3. **Outputs**: Extract Waste, Emissions. **Crucial**: Record Product quantitative flow here via `record_process_flow`.
4. **Validation** (Optional): Check completeness for full inventory tasks.
*For short QA: Jump directly to the requested info.*

## LCI Categories (11 types)
**Input**: Raw Material, Process Energy, Post-processing Energy, Feedstock Energy, Gas, Cooling Media
**Output**: Product, Recovered Material, Waste, Emission
**Scope**: Functional Unit

## Key Guidelines
1. **Batch Search**: Always combine related keywords (e.g., `["electricity", "power"]`) in one search.
2. **Calc vs Record**: 
   - Explicit data (e.g., "120 kWh") → `record_process_flow`
   - Needs calculation (e.g., "500W, 2h") → `record_parameter` + `execute_calculation` → `record_process_flow`
3. **Energy Classification**:
   - *Process*: Printing/Machine operation
   - *Post-processing*: Heat treatment/Machining
   - *Feedstock*: Powder production
4. **Note Field**: Use for context (e.g., "SLM machine", "Atomization") to distinguish same-name flows or key qualifiers (e.g., "99.9% purity", "recycled") to enrich flows.
5. **Session ID**: Include in tool calls for traceability. The system will handle it automatically.
6. **Summary Check**: Use `get_session_summary` to review recorded data when needed.
7. **Missing Data**: Skip if not found. Do not hallucinate.

**Remember**: Be concise. Extract quantitative data accurately.

**AUTOMATIC SESSION INJECTION**: 
When you call document-related tools, the system will AUTOMATICALLY inject the session_id for you.
You do NOT need to ask the user for session_id. Just call the tools directly with other required parameters."""

        if pdf_session_id:
            prompt += f"\n\n**Current Session**: PDF_SESSION_ID: {pdf_session_id}"
        
        if doc_context:
            prompt += f"\n\n**DOCUMENT CONTEXT**:\n{doc_context}"
            
        return prompt
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        聊天补全接口
        
        使用 Qwen-Agent 的 Assistant 进行对话
        """
        if not self.is_initialized:
            await self.initialize()
            
        try:
            if QWEN_AGENT_AVAILABLE and self.assistant:
                return await self._qwen_agent_chat(messages, tools, max_tokens, temperature)
            else:
                return await self._simple_chat(messages, tools, max_tokens, temperature)
                
        except Exception as e:
            logger.error(f"Chat completion 失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        流式聊天补全接口 - 使用 Qwen-Agent 的 bot.run() Iterator
        
        🔥 关键改进：
        - 使用 Qwen-Agent 的 run() 方法，保留完整的工具调用能力
        - 流式输出 thinking 过程和最终内容
        - 实时推送工具调用状态
        
        Yields:
            Dict with type: "thinking", "content", "tool_call", "tool_result", "done", "error"
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if not QWEN_AGENT_AVAILABLE or not self.assistant:
                # 降级到简单流式
                async for chunk in self._simple_stream(messages, tools, max_tokens, temperature):
                    yield chunk
                return
            
            # 从消息中提取 pdf_session_id
            pdf_session_id = None
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if "PDF_SESSION_ID:" in content:
                        match = re.search(r'PDF_SESSION_ID:\s*(\S+)', content)
                        if match:
                            pdf_session_id = match.group(1)
                            break
            
            # 设置工具服务
            if hasattr(self, '_tool_service') and self._tool_service:
                set_tool_service(self._tool_service, pdf_session_id)
                logger.info(f"🔧 流式模式：已设置工具服务，pdf_session_id: {pdf_session_id}")
            
            # 格式化消息为 Qwen-Agent 格式
            qwen_messages = self._format_messages_for_qwen_agent(messages)
            
            logger.info(f"🌊 开始 Qwen-Agent 流式调用，消息数: {len(qwen_messages)}")
            
            # 🔥 使用 Qwen-Agent 的 run() 方法 - 它返回 Iterator
            # 在异步上下文中运行同步的 generator
            import asyncio
            
            last_content = ""
            last_thinking = ""
            tool_results = []
            
            # Qwen-Agent 的 run() 是同步的 generator，需要在线程中运行
            def run_agent():
                return list(self.assistant.run(messages=qwen_messages))
            
            # 在线程池中运行
            loop = asyncio.get_event_loop()
            all_responses = await loop.run_in_executor(None, run_agent)
            
            # 处理所有响应
            for response in all_responses:
                if not response:
                    continue
                    
                last_msg = response[-1] if response else None
                if not last_msg:
                    continue
                
                # 获取内容
                if isinstance(last_msg, dict):
                    content = last_msg.get('content', '')
                    role = last_msg.get('role', '')
                    func_call = last_msg.get('function_call')
                else:
                    content = getattr(last_msg, 'content', '')
                    role = getattr(last_msg, 'role', '')
                    func_call = getattr(last_msg, 'function_call', None)
                
                # 解析 thinking 和 content
                clean_content = content or ''
                new_thinking = ''
                
                if content:
                    # 提取 thinking（处理完整和不完整的标签）
                    if '<think>' in content:
                        if '</think>' in content:
                            # 完整的 think 标签
                            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                            if think_match:
                                new_thinking = think_match.group(1).strip()
                            clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                        else:
                            # 不完整的 think 标签（被截断）
                            idx = content.find('<think>')
                            new_thinking = content[idx + 7:].strip()
                            clean_content = content[:idx]
                    
                    # 移除 tool_call 标签（完整和不完整）
                    clean_content = re.sub(r'</?tool_call>?', '', clean_content)
                    clean_content = re.sub(r'\{"name":\s*"[^"]*",\s*"arguments":\s*\{[^}]*\}?\}?', '', clean_content)
                    clean_content = clean_content.strip()
                
                # 发送 thinking
                if new_thinking and new_thinking != last_thinking:
                    delta_thinking = new_thinking[len(last_thinking):] if new_thinking.startswith(last_thinking) else new_thinking
                    if delta_thinking:
                        yield {
                            "type": "thinking",
                            "content": delta_thinking
                        }
                    last_thinking = new_thinking
                
                # 发送内容增量
                if clean_content and clean_content != last_content:
                    delta_content = clean_content[len(last_content):] if clean_content.startswith(last_content) else clean_content
                    if delta_content:
                        yield {
                            "type": "content",
                            "content": delta_content
                        }
                    last_content = clean_content
                
                # 处理工具调用
                if func_call:
                    tool_name = func_call.get('name', '') if isinstance(func_call, dict) else getattr(func_call, 'name', '')
                    yield {
                        "type": "tool_call",
                        "tool_call": {"tool_name": tool_name}
                    }
                
                # 处理工具响应
                if role == 'function' or role == 'tool':
                    tool_name = last_msg.get('name', '') if isinstance(last_msg, dict) else getattr(last_msg, 'name', '')
                    tool_results.append({
                        "tool_name": tool_name,
                        "result": content
                    })
                    yield {
                        "type": "tool_result",
                        "tool_name": tool_name,
                        "result": content[:200] + "..." if len(content) > 200 else content
                    }
            
            # 完成
            yield {
                "type": "done",
                "success": True,
                "thinking": last_thinking,
                "content": last_content,
                "tool_results": tool_results if tool_results else None
            }
            
        except Exception as e:
            logger.error(f"Qwen-Agent 流式聊天失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e)
            }
    
    def _format_messages_for_qwen_agent(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """格式化消息为 Qwen-Agent 格式
        
        🔥 关键：Qwen-Agent 要求第一条非 system 消息必须是 user
        """
        qwen_messages = []
        has_user_message = False
        pending_assistant_msgs = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                qwen_messages.append({"role": "system", "content": content})
            elif role == "tool":
                # 工具响应转换为 user 消息
                qwen_messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{content}\n</tool_response>"
                })
                has_user_message = True
            elif role == "assistant":
                if has_user_message:
                    # 已有 user 消息，可以添加 assistant
                    qwen_messages.append({"role": "assistant", "content": content})
                else:
                    # 还没有 user 消息，暂存 assistant 消息
                    pending_assistant_msgs.append(content)
            elif role == "user":
                # 如果有暂存的 assistant 消息，将其作为上下文注入到 user 消息
                if pending_assistant_msgs and not has_user_message:
                    context = "\n".join([f"[Previous Context]: {m}" for m in pending_assistant_msgs])
                    content = f"{context}\n\n[User Query]: {content}"
                    pending_assistant_msgs = []
                
                qwen_messages.append({"role": "user", "content": content})
                has_user_message = True
        
        # 确保至少有一条 user 消息
        if not has_user_message:
            # 如果没有 user 消息，创建一个默认的
            qwen_messages.append({"role": "user", "content": "Please help me with the document."})
        
        return qwen_messages
    
    async def _simple_stream(self, messages, tools, max_tokens, temperature):
        """简单流式模式（降级方案）"""
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        formatted_messages = self._format_messages_for_stream(messages)
        
        stream = await client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            max_tokens=min(max_tokens, 4096),
            temperature=temperature,
            stream=True
        )
        
        full_content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                full_content += delta
                yield {"type": "content", "content": delta}
        
        yield {"type": "done", "success": True, "content": full_content}
    
    def _format_messages_for_stream(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """格式化消息用于流式输出"""
        formatted = []
        system_context = ""
        welcome_context = ""
        first_non_system_handled = False
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_context = content
            elif role == "tool":
                formatted.append({
                    "role": "user",
                    "content": f"[Tool Response]\n{content}"
                })
            elif role == "assistant" and not first_non_system_handled:
                welcome_context = content
                first_non_system_handled = True
            else:
                formatted.append({
                    "role": role,
                    "content": content
                })
                if role != "system":
                    first_non_system_handled = True
        
        # 将上下文注入到第一条 user 消息
        if formatted and (system_context or welcome_context):
            for i, msg in enumerate(formatted):
                if msg["role"] == "user":
                    context_prefix = ""
                    if system_context:
                        context_prefix += f"[System Context]\n{system_context}\n\n"
                    if welcome_context:
                        context_prefix += f"[Document Status]\n{welcome_context}\n\n"
                    
                    formatted[i]["content"] = context_prefix + "[User Query]\n" + msg["content"]
                    break
        
        return formatted
            
    async def _qwen_agent_chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """
        使用 Qwen-Agent 的 Assistant.run() 进行对话
        
        🔥 真正的 Qwen-Agent 集成：
        - Qwen-Agent 自动解析工具调用
        - Qwen-Agent 自动执行工具（通过我们注册的 BaseTool）
        - Qwen-Agent 自动循环直到任务完成
        """
        
        # 从消息中提取 system prompt 和 pdf_session_id
        system_prompt = None
        pdf_session_id = None
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                if "PDF_SESSION_ID:" in system_prompt:
                    import re as re_module
                    match = re_module.search(r'PDF_SESSION_ID:\s*(\S+)', system_prompt)
                    if match:
                        pdf_session_id = match.group(1)
                break
        
        # 🔥 设置工具服务（让 BaseTool 可以访问）
        if hasattr(self, '_tool_service') and self._tool_service:
            set_tool_service(self._tool_service, pdf_session_id)
            logger.info(f"🔧 已设置工具服务，pdf_session_id: {pdf_session_id}")
        
        # 🔥 动态更新 Assistant 的 system_message（确保包含正确的 PDF_SESSION_ID）
        if system_prompt and self.assistant:
            self.assistant.system_message = system_prompt
            logger.info(f"🔧 已更新 Assistant 的 system_message，pdf_session_id: {pdf_session_id}")
        
        # 🔥 格式化消息：只保留 user 和 assistant 消息
        # 注意：Qwen-Agent 要求第一条消息必须是 user
        qwen_messages = []
        first_user_found = False
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # system 消息已经通过 assistant.system_message 设置，跳过
                continue
            elif role == "tool":
                # 工具响应转换为 user 消息
                qwen_messages.append({
                    "role": "user",
                    "content": f"[Tool Response]\n{content}"
                })
                first_user_found = True
            elif role == "assistant" and not first_user_found:
                # 跳过第一条 user 之前的 assistant 消息（欢迎语）
                # 因为 Qwen-Agent 要求第一条消息必须是 user
                continue
            else:
                qwen_messages.append({
                    "role": role,
                    "content": content
                })
                if role == "user":
                    first_user_found = True
        
        if not qwen_messages:
            return {
                "success": False,
                "error": "No user messages to process"
            }
        
        try:
            # 🔥 调用 Qwen-Agent 的 Assistant.run()
            # 这会自动处理工具调用和多轮对话
            logger.info(f"🚀 调用 Qwen-Agent Assistant.run()，消息数: {len(qwen_messages)}")
            
            responses = []
            step_count = 0
            for response in self.assistant.run(messages=qwen_messages):
                step_count += 1
                responses.append(response)
                # 只在关键时刻打印日志（工具调用、工具结果）
                if response:
                    last_msg = response[-1] if response else None
                    if last_msg and isinstance(last_msg, dict):
                        role = last_msg.get('role', '')
                        func_call = last_msg.get('function_call')
                        if func_call:
                            logger.info(f"🔧 工具调用: {func_call.get('name', 'N/A')}")
                        elif role == 'function':
                            logger.info(f"📥 工具结果: {str(last_msg.get('content', ''))[:100]}...")
            
            # 解析最后一个响应
            if responses:
                last_response = responses[-1]
                
                # 提取内容
                content = ""
                thinking = ""
                tool_results = []
                
                for item in last_response:
                    if isinstance(item, dict):
                        item_role = item.get('role', '')
                        item_content = item.get('content', '')
                        
                        if item_role == 'assistant':
                            content = item_content
                        elif item_role == 'function':
                            # 工具调用结果
                            tool_results.append({
                                "tool_name": item.get('name', ''),
                                "result": item_content
                            })
                    elif hasattr(item, 'content'):
                        content = item.content
                    elif hasattr(item, 'role') and item.role == 'assistant':
                        content = getattr(item, 'content', str(item))
                
                # 🔥 解析 thinking（处理完整和不完整的标签）
                if '<think>' in content:
                    if '</think>' in content:
                        # 完整的 think 标签
                        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                        if think_match:
                            thinking = think_match.group(1).strip()
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                    else:
                        # 不完整的 think 标签（被截断）- 提取 <think> 后的所有内容作为 thinking
                        idx = content.find('<think>')
                        thinking = content[idx + 7:].strip()
                        content = content[:idx].strip()
                
                # 🔥 移除 tool_call 标签（完整和不完整）
                content = re.sub(r'</?tool_call>?', '', content)
                content = re.sub(r'<\|/?tool_call\|>?', '', content)
                # 移除残留的 JSON 工具调用
                content = re.sub(r'\{"name":\s*"[^"]*"[^}]*\}?', '', content)
                content = content.strip()
                
                # 🔥 如果内容为空（只有工具调用），根据工具结果生成回复
                if not content and tool_results:
                    # 尝试从工具结果中提取有意义的信息
                    recorded_items = []
                    for tr in tool_results:
                        result_str = str(tr.get('result', ''))
                        if 'success' in result_str.lower() or 'recorded' in result_str.lower():
                            recorded_items.append(tr.get('tool_name', 'data'))
                    
                    if recorded_items:
                        content = f"I've successfully recorded the data using {len(tool_results)} tool calls. The recorded items include: {', '.join(set(recorded_items))}. Let me know if you need more details or want to extract additional data."
                    else:
                        content = "I've processed your request. The data has been recorded successfully."
                elif not content and thinking:
                    # 如果只有 thinking 没有 content，说明模型被截断了
                    content = "I've analyzed your request. Please let me know if you need more details."
                
                result = {
                    "success": True,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "thinking": thinking
                    }
                }
                
                if tool_results:
                    result["tool_results"] = tool_results
                    logger.info(f"🔧 Qwen-Agent 执行了 {len(tool_results)} 个工具")
                
                return result
            else:
                return {
                    "success": False,
                    "error": "No response from Qwen-Agent"
                }
                
        except Exception as e:
            logger.error(f"Qwen-Agent Assistant.run() 失败: {e}", exc_info=True)
            # 降级到简化模式
            logger.info("⚠️ 降级到简化模式")
            return await self._simple_chat(messages, tools, max_tokens, temperature)
            
    async def _simple_chat(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """简化模式：直接调用 vLLM API"""
        
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
        
        # 格式化消息
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "tool":
                formatted_messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{content}\n</tool_response>"
                })
            else:
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
                
        # 🔥 修复：确保消息以 user 开头（vLLM 要求）
        if formatted_messages:
            first_non_system_idx = None
            for i, msg in enumerate(formatted_messages):
                if msg["role"] != "system":
                    first_non_system_idx = i
                    break
                    
            if first_non_system_idx is not None:
                first_msg = formatted_messages[first_non_system_idx]
                if first_msg["role"] == "assistant":
                    logger.info("🔧 将首条 assistant 消息转换为 system 消息")
                    formatted_messages[first_non_system_idx] = {
                        "role": "system",
                        "content": f"[Document Context]\n{first_msg['content']}"
                    }
                
        # 格式化工具
        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                if "function" in tool:
                    openai_tools.append({
                        "type": "function",
                        "function": tool["function"]
                    })
                    
        # 调用 API
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            tools=openai_tools,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # 解析 thinking
        thinking = ""
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
        result = {
            "success": True,
            "message": {
                "role": "assistant",
                "content": content,
                "thinking": thinking
            }
        }
        
        # 解析工具调用
        if choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "tool_name": tc.function.name,
                    "parameters": json.loads(tc.function.arguments)
                })
            result["message"]["tool_calls"] = tool_calls
            
        return result
        
    async def simple_generate(self, prompt: str, max_tokens: int = 512) -> str:
        """简单文本生成"""
        messages = [{"role": "user", "content": prompt}]
        result = await self.chat_completion(messages, max_tokens=max_tokens)
        if result.get("success"):
            return result["message"]["content"]
        return ""
    
    def stream_chat_generator(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ):
        """
        真正的 tokens 级流式生成器
        
        利用 Qwen-Agent 的 bot.run() 生成器特性，每次 yield 时立即推送
        返回格式：{'type': 'thinking'|'content'|'tool_call'|'tool_result'|'done', ...}
        """
        # 确保初始化
        if not self.is_initialized:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果已经在事件循环中，同步初始化
                    self._sync_initialize()
                else:
                    loop.run_until_complete(self.initialize())
            except RuntimeError:
                self._sync_initialize()
        
        if not QWEN_AGENT_AVAILABLE or not self.assistant:
            yield {"type": "error", "error": "Qwen-Agent not available"}
            return
        
        # 提取 pdf_session_id
        pdf_session_id = None
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if "PDF_SESSION_ID:" in content:
                    import re as re_module
                    match = re_module.search(r'PDF_SESSION_ID:\s*(\S+)', content)
                    if match:
                        pdf_session_id = match.group(1)
                        break
        
        # 设置工具服务
        if hasattr(self, '_tool_service') and self._tool_service:
            set_tool_service(self._tool_service, pdf_session_id)
        
        # 提取 system prompt 并更新 Assistant
        system_prompt = None
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break
        
        if system_prompt and self.assistant:
            self.assistant.system_message = system_prompt
        
        # 格式化消息：只保留 user 和 assistant 消息
        qwen_messages = []
        first_user_found = False
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                continue
            elif role == "tool":
                qwen_messages.append({
                    "role": "user",
                    "content": f"[Tool Response]\n{content}"
                })
                first_user_found = True
            elif role == "assistant" and not first_user_found:
                continue
            else:
                qwen_messages.append({"role": role, "content": content})
                if role == "user":
                    first_user_found = True
        
        if not qwen_messages:
            yield {"type": "error", "error": "No user messages to process"}
            return
        
        try:
            logger.info(f"Streaming: Starting bot.run() with {len(qwen_messages)} messages")
            
            # 用于跟踪状态
            last_content = ""
            last_thinking = ""
            seen_tool_calls = set()
            tool_results = []
            
            # 遍历 bot.run() 生成器 - 这是 tokens 级流式的核心
            for response in self.assistant.run(messages=qwen_messages):
                if not response:
                    continue
                
                # response 是消息列表，取最后一条
                latest_msg = response[-1] if response else None
                if not latest_msg:
                    continue
                
                # 处理不同类型的消息
                if isinstance(latest_msg, dict):
                    role = latest_msg.get('role', '')
                    content = latest_msg.get('content', '')
                    func_call = latest_msg.get('function_call')
                    
                    # 工具调用
                    if func_call:
                        tool_name = func_call.get('name', '')
                        tool_id = f"{tool_name}_{hash(str(func_call.get('arguments', '')))}"
                        if tool_id not in seen_tool_calls:
                            seen_tool_calls.add(tool_id)
                            yield {
                                "type": "tool_call",
                                "tool_name": tool_name,
                                "arguments": func_call.get('arguments', '')
                            }
                    
                    # 工具结果
                    elif role == 'function':
                        tool_results.append({
                            "tool_name": latest_msg.get('name', ''),
                            "result": content[:500]  # 截断长结果
                        })
                        yield {
                            "type": "tool_result",
                            "tool_name": latest_msg.get('name', ''),
                            "result": content[:200]
                        }
                    
                    # Assistant 回复（包含 thinking 和 content）
                    elif role == 'assistant' and content:
                        # 解析 thinking
                        current_thinking = ""
                        current_content = content
                        
                        if '<think>' in content:
                            if '</think>' in content:
                                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                                if think_match:
                                    current_thinking = think_match.group(1).strip()
                                current_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                            else:
                                # 不完整的 think 标签
                                idx = content.find('<think>')
                                current_thinking = content[idx + 7:].strip()
                                current_content = content[:idx].strip()
                        
                        # 清理 tool_call 标签
                        current_content = re.sub(r'</?tool_call>?', '', current_content)
                        current_content = re.sub(r'<\|/?tool_call\|>?', '', current_content)
                        current_content = re.sub(r'\{"name":\s*"[^"]*"[^}]*\}?', '', current_content)
                        current_content = current_content.strip()
                        
                        # 推送 thinking（如果有变化）
                        if current_thinking and current_thinking != last_thinking:
                            last_thinking = current_thinking
                            yield {
                                "type": "thinking",
                                "content": current_thinking
                            }
                        
                        # 推送 content（如果有变化）- 全量快照模式
                        if current_content and current_content != last_content:
                            last_content = current_content
                            yield {
                                "type": "content",
                                "content": current_content
                            }
                
                # 处理 Message 对象
                elif hasattr(latest_msg, 'content'):
                    content = getattr(latest_msg, 'content', '')
                    if content and content != last_content:
                        last_content = content
                        yield {
                            "type": "content",
                            "content": content
                        }
            
            # 完成
            final_content = last_content
            if not final_content and tool_results:
                final_content = "I've processed your request. The data has been recorded successfully."
            
            yield {
                "type": "done",
                "success": True,
                "content": final_content,
                "thinking": last_thinking,
                "tool_results": tool_results
            }
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}
