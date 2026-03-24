#!/usr/bin/env python3
"""
LLM对话管理服务
管理用户与LLM的对话会话，支持工具调用和上下文管理
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from .local_qwen_service import LocalQwenService
from .tool_service import LCAToolService

logger = logging.getLogger(__name__)
logger.info("🔥🔥🔥 llm_chat_service.py 模块已加载（带调试代码）🔥🔥🔥")

# 🔥 自定义 JSON encoder 处理 datetime 对象
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@dataclass
class ChatMessage:
    """聊天消息"""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = None
    tool_calls: List[Dict[str, Any]] = None
    tool_call_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ChatSession:
    """聊天会话"""
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_activity: datetime
    context: Dict[str, Any] = None  # 额外上下文信息
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class LLMChatService:
    """LLM对话管理服务"""
    
    def __init__(self, 
                 qwen_service: LocalQwenService,
                 tool_service: LCAToolService,
                 session_timeout: int = 3600):
        """
        初始化对话服务
        
        Args:
            qwen_service: 本地Qwen服务
            tool_service: 工具服务
            session_timeout: 会话超时时间(秒)
        """
        self.qwen_service = qwen_service
        self.tool_service = tool_service
        self.session_timeout = session_timeout
        self.sessions: Dict[str, ChatSession] = {}
        
        # 启动清理任务
        self._cleanup_task = None
        
        logger.info("LLM对话服务初始化完成")
    
    async def initialize(self):
        """初始化服务（延迟加载模式）"""
        # 注意：Qwen服务采用延迟加载，仅在首次聊天请求时初始化
        # 这里不主动加载模型，节省显存
        
        # 启动会话清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info("LLM对话服务启动完成（延迟加载模式）")
    
    def create_chat_session(self, session_id: str = None, pdf_session_id: str = None) -> str:
        """
        创建新的聊天会话
        
        Args:
            session_id: 可选的会话ID，如果不提供则自动生成
            pdf_session_id: 可选的PDF会话ID，用于关联文档上下文（可为空，支持独立模式）
            
        Returns:
            str: 会话ID
        """
        logger.info(f"[DEBUG create_chat_session] 收到参数: session_id={session_id}, pdf_session_id={pdf_session_id}")
        
        if session_id is None:
            session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.sessions)}"
        
        # 创建会话（支持无PDF上下文的独立模式）
        context = {}
        if pdf_session_id:
            context["pdf_session_id"] = pdf_session_id
            context["mode"] = "document_based"  # 基于文档的模式
        else:
            context["mode"] = "standalone"  # 独立模式
        
        logger.info(f"[DEBUG create_chat_session] 构建context: {context}")
        
        session = ChatSession(
            session_id=session_id,
            messages=[],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            context=context
        )
        
        logger.info(f"[DEBUG create_chat_session] ChatSession对象创建完成, context={session.context}")
        
        # 添加完整的系统消息（包含角色定义和文档上下文）
        system_content = """You are an expert LCA assistant for Additive Manufacturing.

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
3. **Outputs**: Extract Waste, Emissions. Record Product quantitative flow here via `record_process_flow`.
4. **Validation** (Optional): Check completeness for full inventory tasks.
*For short QA: Jump directly to the requested info.*

## LCI Categories (11 types)
**Input**: Raw Material, Process Energy, Post-processing Energy, Feedstock Energy, Gas, Cooling Media
**Output**: Product, Recovered Material, Waste, Emission
**Scope**: Functional Unit

## Key Guidelines
1. **Batch Search**: Always combine related keywords (e.g., `["electricity", "power"]`) in one search.
2. **Calc vs Record**: 
   - Explicit data (e.g., "120 kWh") -> `record_process_flow`
   - Needs calculation (e.g., "500W, 2h") -> `record_parameter` + `execute_calculation` -> `record_process_flow`
3. **Energy Classification**:
   - *Process*: Printing/Machine operation
   - *Post-processing*: Heat treatment/Machining
   - *Feedstock*: Powder production
4. **Note Field**: Use for context (e.g., "SLM machine", "Atomization") to distinguish same-name flows.
5. **Session ID**: The system handles it automatically. Just call tools directly.
6. **Missing Data**: Skip if not found. Do not hallucinate.

**AUTOMATIC SESSION INJECTION**: 
When you call document-related tools, the system will AUTOMATICALLY inject the session_id.
You do NOT need to ask the user for session_id. Just call the tools directly."""

        if context.get("mode") == "standalone":
            system_content += "\n\nMODE: standalone"
        else:
            system_content += f"\n\nMODE: document_based\nPDF_SESSION_ID: {pdf_session_id if pdf_session_id else 'none'}"
            
            # 🔥 注入文档前几个 chunk 作为上下文，让 LLM 知道文档内容
            if pdf_session_id and hasattr(self, 'tool_service') and self.tool_service:
                try:
                    # 直接从 session_manager 获取知识库（同步方式）
                    session_data = self.tool_service.session_manager.get_session(pdf_session_id)
                    if session_data and session_data.knowledge_base:
                        # 使用空查询获取前几个 chunk
                        kb = session_data.knowledge_base
                        # ChromaDB 的 query 方法是同步的
                        results = kb.query("document content overview", n_results=2)
                        if results and results.get("documents") and results["documents"][0]:
                            doc_context = "\n\n**DOCUMENT PREVIEW (first chunks)**:\n"
                            for i, doc in enumerate(results["documents"][0][:2]):
                                chunk_text = doc[:500] if doc else ""
                                doc_context += f"\n[Chunk {i}]: {chunk_text}...\n"
                            system_content += doc_context
                            logger.info(f"✅ 已注入文档上下文 ({len(results['documents'][0])} chunks)")
                except Exception as e:
                    logger.warning(f"无法获取文档上下文: {e}")

        system_message = ChatMessage(
            role="system",
            content=system_content
        )
        
        session.messages.append(system_message)
        
        # 如果有文档，添加一条初始助手消息确认文档已就绪
        if pdf_session_id:
            # 获取文档信息
            try:
                doc_info = self.tool_service.kb_manager.get_session_info(pdf_session_id)
                doc_details = f"Document ID: {pdf_session_id[:8]}..., Pages: {doc_info.get('total_pages', 'N/A')}, Chunks: {doc_info.get('total_chunks', 'N/A')}"
            except Exception as e:
                doc_details = f"Document ID: {pdf_session_id[:8]}..."
                logger.warning(f"无法获取文档详细信息: {e}")
            
            initial_assistant_message = ChatMessage(
                role="assistant",
                content=f"Document successfully processed and ready for analysis.\n\n{doc_details}\n\nI have access to the document content and can search through it to answer your questions. What would you like to know about the document?"
            )
            session.messages.append(initial_assistant_message)
            logger.warning(f"✅ 已添加初始助手消息确认文档就绪 (pdf_session_id: {pdf_session_id[:8]}...)")
        
        self.sessions[session_id] = session
        
        logger.info(f"创建聊天会话: {session_id}")
        return session_id
    
    async def chat(self, 
                   session_id: str, 
                   user_message: str,
                   include_tools: bool = True) -> Dict[str, Any]:
        """
        处理用户聊天消息
        
        Args:
            session_id: 会话ID
            user_message: 用户消息
            include_tools: 是否包含工具调用能力
            
        Returns:
            Dict[str, Any]: 聊天响应
        """
        try:
            logger.info(f"🚀 chat() - session_id={session_id[:20]}...")
            
            # 检查会话是否存在
            if session_id not in self.sessions:
                return {
                    "success": False,
                    "error": "Chat session not found. Please create a new session."
                }
            
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
            
            logger.info(f"📦 会话上下文: mode={session.context.get('mode')}")
            
            # 解析PDF会话ID（如果存在）- 支持多种格式
            pdf_session_id = None
            clean_message = user_message
            
            # 方法1：从消息中解析
            if user_message.startswith("[PDF_SESSION_ID:"):
                end_idx = user_message.find("]")
                if end_idx != -1:
                    pdf_session_id = user_message[16:end_idx].strip()
                    clean_message = user_message[end_idx+1:].strip()
                    logger.info(f"Detected PDF session ID from message: {pdf_session_id}")
            
            # 方法2：从会话上下文中获取
            if not pdf_session_id and session.context.get("pdf_session_id"):
                pdf_session_id = session.context["pdf_session_id"]
                logger.info(f"Using PDF session ID from context: {pdf_session_id}")
            
            # 将PDF会话ID存储到聊天会话上下文中
            if pdf_session_id:
                session.context["pdf_session_id"] = pdf_session_id
            
            # 调试日志
            # pdf_session_id 已在上下文中
            
            # 注意：文档上下文已经在create_chat_session时包含在初始system prompt中
            # 不需要在这里再添加额外的消息
            
            # 添加用户消息（使用清理后的消息）
            user_msg = ChatMessage(role="user", content=clean_message)
            session.messages.append(user_msg)
            
            # 准备对话历史（排除系统消息进行API调用）
            messages_for_llm = []
            
            logger.info(f"📝 会话消息数: {len(session.messages)}")
            for msg in session.messages:
                msg_dict = {"role": msg.role, "content": msg.content}
                if msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                messages_for_llm.append(msg_dict)
            
            # 准备工具定义
            tools = None
            if include_tools:
                tools = self._get_available_tools()
            
            # 日志：简化输出
            logger.info(f"📨 发送给LLM消息数: {len(messages_for_llm)}, 工具数: {len(tools) if tools else 0}")
            
            # 调用LLM
            llm_response = await self.qwen_service.chat_completion(
                messages=messages_for_llm,
                tools=tools,
                max_tokens=4096,
                temperature=0.6  # Qwen3 推荐值，适合工具调用/数据提取
            )
            
            if not llm_response.get("success"):
                return {
                    "success": False,
                    "error": f"LLM call failed: {llm_response.get('error')}"  # 🔥 修复：改为英文
                }
            
            assistant_message = llm_response["message"]
            
            # 检查是否有工具调用
            tool_calls = assistant_message.get("tool_calls")
            logger.info(f"🔎 检查工具调用: tool_calls={tool_calls}, type={type(tool_calls)}, bool={bool(tool_calls)}")
            
            if tool_calls:
                logger.info(f"✅ 检测到工具调用，开始处理...")
                return await self._handle_tool_calls(session_id, assistant_message)
            else:
                logger.info(f"ℹ️  没有工具调用，返回普通对话响应")
                # 普通对话响应
                assistant_msg = ChatMessage(
                    role="assistant",
                    content=assistant_message["content"]
                )
                session.messages.append(assistant_msg)
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": assistant_message["content"],
                    "thinking": llm_response.get("thinking", ""),  # 🔥 修复：添加 thinking 字段
                    "message_type": "text",
                    "usage": llm_response.get("usage", {})
                }
        
        except Exception as e:
            logger.error(f"Chat message processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing message: {str(e)}"
            }
    
    async def stream_chat(self, 
                          session_id: str, 
                          user_message: str,
                          include_tools: bool = True):
        """
        流式聊天 - 实时返回 reasoning 和最终响应
        
        这是一个混合模式实现：
        1. 使用 vLLM 流式 API 获取 <think> 内容并实时返回
        2. 检测到工具调用后，执行工具（非流式）
        3. 工具执行完后，继续流式生成最终回复
        
        Yields:
            Dict with type: "thinking", "content", "tool_call", "tool_result", "done", "error"
        """
        import re
        
        try:
            # 检查会话是否存在
            if session_id not in self.sessions:
                yield {"type": "error", "error": "Chat session not found"}
                return
            
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
            
            # 解析PDF会话ID
            pdf_session_id = None
            clean_message = user_message
            
            if user_message.startswith("[PDF_SESSION_ID:"):
                end_idx = user_message.find("]")
                if end_idx != -1:
                    pdf_session_id = user_message[16:end_idx].strip()
                    clean_message = user_message[end_idx+1:].strip()
            
            if not pdf_session_id and session.context.get("pdf_session_id"):
                pdf_session_id = session.context["pdf_session_id"]
            
            if pdf_session_id:
                session.context["pdf_session_id"] = pdf_session_id
            
            # 添加用户消息
            user_msg = ChatMessage(role="user", content=clean_message)
            session.messages.append(user_msg)
            
            # 准备消息
            messages_for_llm = []
            for msg in session.messages:
                msg_dict = {"role": msg.role, "content": msg.content}
                if msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                messages_for_llm.append(msg_dict)
            
            # 准备工具
            tools = self._get_available_tools() if include_tools else None
            
            # 🔥 流式调用 LLM
            logger.info(f"🌊 开始流式聊天，session_id={session_id}")
            
            full_content = ""
            thinking_content = ""
            in_thinking = False
            tool_calls_detected = []
            
            # 检查 qwen_service 是否支持流式
            if hasattr(self.qwen_service, 'stream_chat_completion'):
                # 使用流式 API
                async for chunk in self.qwen_service.stream_chat_completion(
                    messages=messages_for_llm,
                    tools=tools,
                    max_tokens=4096,  # 🔥 限制 max_tokens 避免超过模型上下文限制
                    temperature=0.7
                ):
                    chunk_type = chunk.get("type", "content")
                    
                    if chunk_type == "content":
                        content = chunk.get("content", "")
                        
                        # 🔥 逐字符处理，正确解析 <think> 标签
                        for char in content:
                            full_content += char
                            
                            # 检测 <think> 开始
                            if not in_thinking and full_content.endswith("<think>"):
                                in_thinking = True
                                # 不输出 <think> 标签本身
                                continue
                            
                            # 检测 </think> 结束
                            if in_thinking and full_content.endswith("</think>"):
                                in_thinking = False
                                # 移除最后的 </think> 从 thinking_content
                                if thinking_content.endswith("</think"):
                                    thinking_content = thinking_content[:-7]
                                elif thinking_content.endswith("</thin"):
                                    thinking_content = thinking_content[:-6]
                                elif thinking_content.endswith("</thi"):
                                    thinking_content = thinking_content[:-5]
                                elif thinking_content.endswith("</th"):
                                    thinking_content = thinking_content[:-4]
                                elif thinking_content.endswith("</t"):
                                    thinking_content = thinking_content[:-3]
                                elif thinking_content.endswith("</"):
                                    thinking_content = thinking_content[:-2]
                                elif thinking_content.endswith("<"):
                                    thinking_content = thinking_content[:-1]
                                continue
                            
                            if in_thinking:
                                # 在 thinking 中，但不输出可能是标签的部分
                                if not full_content.endswith(("<", "</", "</t", "</th", "</thi", "</thin", "</think")):
                                    thinking_content += char
                                    yield {"type": "thinking", "content": char}
                            else:
                                # 普通内容，但不输出可能是标签的部分
                                if not full_content.endswith(("<", "<t", "<th", "<thi", "<thin", "<think")):
                                    yield {"type": "content", "content": char}
                    
                    elif chunk_type == "tool_call":
                        tool_call = chunk.get("tool_call", {})
                        tool_calls_detected.append(tool_call)
                        yield {"type": "tool_call", "tool_call": tool_call}
                    
                    elif chunk_type == "final":
                        # 流式完成
                        pass
                
                # 处理工具调用
                if tool_calls_detected:
                    logger.info(f"🔧 检测到 {len(tool_calls_detected)} 个工具调用，开始执行...")
                    
                    # 执行工具调用
                    for tool_call in tool_calls_detected:
                        tool_name = tool_call.get("tool_name")
                        parameters = tool_call.get("parameters", {})
                        
                        yield {"type": "tool_executing", "tool_name": tool_name}
                        
                        # 执行工具
                        tool_result = await self._execute_tool(session_id, tool_name, parameters)
                        
                        yield {
                            "type": "tool_result",
                            "tool_name": tool_name,
                            "result": tool_result
                        }
                    
                    # 工具执行完成后，让 LLM 生成总结
                    # 这里使用非流式，因为工具结果需要完整处理
                    summary_result = await self._generate_tool_summary(session_id, tool_calls_detected)
                    
                    yield {
                        "type": "content",
                        "content": summary_result.get("message", "")
                    }
                
                # 保存助手消息
                # 清理 thinking 标签
                clean_content = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
                
                assistant_msg = ChatMessage(role="assistant", content=clean_content)
                session.messages.append(assistant_msg)
                
                yield {
                    "type": "done",
                    "session_id": session_id,
                    "full_content": clean_content,
                    "thinking": thinking_content
                }
            
            else:
                # 降级：使用非流式 API，但模拟流式输出
                logger.info("⚠️ qwen_service 不支持流式，使用非流式降级")
                
                result = await self.chat(session_id, user_message, include_tools)
                
                if result.get("success"):
                    # 模拟流式输出 thinking
                    if result.get("thinking"):
                        yield {"type": "thinking", "content": result["thinking"]}
                    
                    # 模拟流式输出 content
                    content = result.get("message", "")
                    # 分块输出
                    chunk_size = 50
                    for i in range(0, len(content), chunk_size):
                        yield {"type": "content", "content": content[i:i+chunk_size]}
                    
                    yield {
                        "type": "done",
                        "session_id": session_id,
                        "full_content": content,
                        "thinking": result.get("thinking", "")
                    }
                else:
                    yield {"type": "error", "error": result.get("error", "Unknown error")}
                    
        except Exception as e:
            logger.error(f"流式聊天失败: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}
    
    async def _generate_tool_summary(self, session_id: str, tool_calls: list) -> Dict[str, Any]:
        """让 LLM 根据工具执行结果生成总结"""
        try:
            session = self.sessions[session_id]
            
            # 准备消息
            messages = []
            for msg in session.messages:
                messages.append({"role": msg.role, "content": msg.content})
            
            # 调用 LLM 生成总结（不带工具）
            result = await self.qwen_service.chat_completion(
                messages=messages,
                tools=None,  # 不带工具，让 LLM 生成自然语言总结
                max_tokens=2048,
                temperature=0.7
            )
            
            if result.get("success"):
                return {"message": result["message"].get("content", "")}
            else:
                return {"message": "Tool execution completed."}
                
        except Exception as e:
            logger.error(f"生成工具总结失败: {e}")
            return {"message": "Tool execution completed."}
    
    async def _handle_tool_calls(self, session_id: str, assistant_message: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """处理工具调用
        
        Args:
            depth: 递归深度，用于防止无限循环
        """
        try:
            # 🔥 递归深度限制
            MAX_DEPTH = 5
            if depth >= MAX_DEPTH:
                logger.warning(f"⚠️ [SYSTEM_DEFENSE] 递归深度达到限制 ({depth}/{MAX_DEPTH})，强制停止")
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": (
                        "✅ I have completed the data extraction process.\n\n"
                        "Multiple tool calls have been executed. Please check the session summary for all recorded data."
                    ),
                    "message_type": "tool_response",
                    "tool_results": []
                }
            
            session = self.sessions[session_id]
            tool_calls = assistant_message.get("tool_calls", [])
            
            logger.info(f"🛠️ 开始处理工具调用，工具数量: {len(tool_calls)}，递归深度: {depth}")
            
            # 添加助手的工具调用消息
            # 🔥 代码层防御：如果调用了工具，强制清空 content（避免输出描述性文字到前端）
            content = assistant_message.get("content") or ""
            
            if tool_calls:
                # 有工具调用时，LLM 不应该输出描述性文字
                # 正确的流程是：调用工具 → 收到结果 → 通过 Continue Prompt 再生成响应
                if content.strip() and content != "[Calling tools...]":
                    logger.warning(f"⚠️ [SYSTEM_DEFENSE] LLM mixed tool calls with text output (length: {len(content)})")
                    logger.info(f"📝 Content preview (first 200 chars): {content[:200]}")
                    logger.info(f"🔧 Forcing content to be empty to prevent raw output to frontend")
                content = "[Calling tools...]"  # 统一的占位符
            elif not content.strip():
                # 没有工具调用，但 content 为空（异常情况）
                content = "[Processing...]"
            
            # 🔥🔥🔥 关键防御：限制单次工具调用数量并去重
            if len(tool_calls) > 10:
                logger.warning(f"⚠️ [SYSTEM_DEFENSE] 单次生成了 {len(tool_calls)} 个工具调用，限制为前 10 个")
                tool_calls = tool_calls[:10]
            
            # 去重：相同的工具+参数只保留第一个
            seen_calls = set()
            unique_tool_calls = []
            for tc in tool_calls:
                # 创建唯一标识
                call_key = (tc.get("tool_name"), str(sorted(tc.get("parameters", {}).items())))
                if call_key not in seen_calls:
                    seen_calls.add(call_key)
                    unique_tool_calls.append(tc)
                else:
                    logger.warning(f"⚠️ [SYSTEM_DEFENSE] 跳过重复的工具调用: {tc.get('tool_name')}")
            
            if len(unique_tool_calls) < len(tool_calls):
                logger.warning(f"⚠️ [SYSTEM_DEFENSE] 去重后工具调用数: {len(tool_calls)} -> {len(unique_tool_calls)}")
                tool_calls = unique_tool_calls
            
            assistant_msg = ChatMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls
            )
            session.messages.append(assistant_msg)
            
            tool_results = []
            
            # 🔥 检查 pivot 次数限制（防止无限循环）
            if any(tc.get("tool_name") == "record_pivot_failure" for tc in tool_calls):
                # 统计最近的 pivot 次数
                recent_pivots = sum(
                    1 for msg in session.messages[-10:]
                    if msg.role == "tool" and "pivot" in msg.content.lower()
                )
                if recent_pivots >= 3:
                    logger.warning(f"⚠️ Pivot 次数已达到限制 ({recent_pivots}/3)")
                    return {
                        "success": False,
                        "error": "Maximum pivot attempts (3) reached. The data may not exist in the document. Please try different search terms or inform the user.",
                        "thinking": None,
                        "tool_results": []
                    }
            
            # 🔥 检查 record_parameter 调用次数（防止无限记录循环）
            recent_records = sum(
                1 for msg in session.messages[-20:]
                if msg.role == "tool" and "record_parameter" in str(msg.content).lower()
            )
            if recent_records >= 10:
                logger.warning(f"⚠️ [SYSTEM_DEFENSE] Too many record_parameter calls ({recent_records}/10)")
                logger.info(f"🛑 Stopping to prevent infinite recording loop")
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": (
                        "✅ I have recorded multiple parameters from the document.\n\n"
                        "The data collection is complete. Please check the session summary for all recorded values."
                    ),
                    "message_type": "tool_response",
                    "tool_results": []
                }
            
            # 执行每个工具调用
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool_name")
                parameters = tool_call.get("parameters", {})
                
                logger.info(f"执行工具调用: {tool_name}")
                
                # 自动注入PDF会话ID（仅对需要session_id的工具）
                pdf_session_id = session.context.get("pdf_session_id")
                # 需要session_id的工具列表（包括三工具架构的所有工具）
                tools_need_session_id = [
                    "search_document",
                    "build_lca_system", 
                    "execute_analysis",
                    "define_lca_scope",
                    "record_process_flow",
                    "record_parameter",
                    "record_calculation",
                    "get_session_summary",
                    "record_pivot_failure"
                ]
                if pdf_session_id and tool_name in tools_need_session_id:
                    if "session_id" not in parameters:
                        parameters["session_id"] = pdf_session_id
                        logger.info(f"自动注入PDF会话ID: {pdf_session_id} 到工具 {tool_name}")
                
                # process_document工具不需要session_id，它需要file_content和filename
                
                # 调用对应的工具
                if tool_name == "process_document":
                    result = await self.tool_service.process_document(**parameters)
                elif tool_name == "search_document":
                    result = await self.tool_service.search_document(**parameters)
                elif tool_name == "search_lci_database":
                    result = await self.tool_service.search_lci_database(**parameters)
                elif tool_name == "define_lca_scope":
                    result = await self.tool_service.define_lca_scope(**parameters)
                elif tool_name == "record_process_flow":
                    result = await self.tool_service.record_process_flow(**parameters)
                elif tool_name == "record_parameter":
                    result = self.tool_service.record_parameter(**parameters)
                elif tool_name == "record_calculation":
                    result = self.tool_service.record_calculation(**parameters)
                elif tool_name == "execute_calculation":
                    # execute_calculation是同步函数，需要特殊处理
                    from backend.app import calculator
                    expression = parameters.get("expression", "")
                    variables = parameters.get("variables", {})
                    try:
                        calc_result = calculator.calculate(expression, variables)
                        result = {
                            "success": True,
                            "data": {
                                "expression": expression,
                                "result": calc_result,
                                "variables": variables
                            }
                        }
                    except Exception as e:
                        result = {
                            "success": False,
                            "error": f"计算失败: {str(e)}"
                        }
                elif tool_name == "get_session_summary":
                    result = await self.tool_service.get_session_summary(**parameters)
                elif tool_name == "build_lca_system":
                    result = await self.tool_service.build_lca_system(**parameters)
                elif tool_name == "execute_analysis":
                    result = await self.tool_service.execute_analysis(**parameters)
                elif tool_name == "record_pivot_failure":
                    result = self.tool_service.record_pivot_failure(**parameters)
                else:
                    result = {
                        "success": False,
                        "error": f"未知的工具: {tool_name}"
                    }
                
                # 🔥 修复：规范化 tool_results 结构，添加 success 和 error 字段
                tool_results.append({
                    "tool_name": tool_name,
                    "success": result.get("success", False),
                    "result": result.get("data") if result.get("success") else None,
                    "error": result.get("error") if not result.get("success") else None
                })
                
                # 🔥 为LLM准备精简的工具结果（移除技术元数据，只保留核心内容）
                if tool_name == "search_document" and result.get("success"):
                    # 只提取文档内容，移除相似度分数、chunk_id等技术细节
                    # 🔥 修复：search_document 返回的是 result["results"]，不是 result["data"]["chunks"]
                    chunks_data = result.get("results", [])
                    simplified_content = {
                        "tool": "search_document",
                        "status": "success",
                        "chunks_found": len(chunks_data),
                        "document_content": []
                    }
                    for i, chunk in enumerate(chunks_data[:5], 1):  # 最多取5个chunk
                        simplified_content["document_content"].append({
                            "chunk_number": i,
                            "content": chunk.get("content", ""),
                            "source": f"Page {chunk.get('metadata', {}).get('page', 'N/A')}, Chunk {chunk.get('metadata', {}).get('chunk_id', 'N/A')}"
                        })
                    tool_content_for_llm = json.dumps(simplified_content, ensure_ascii=False)
                elif tool_name == "get_session_summary":
                    # 🔥 特殊处理：保留完整数据，但添加明确指令
                    result["_instruction_for_llm"] = (
                        "This is the complete data summary. "
                        "Use it to create a detailed natural language report for the user. "
                        "DO NOT output this JSON directly. "
                        "Extract key information and present it in a user-friendly format with proper organization."
                    )
                    tool_content_for_llm = json.dumps(result, ensure_ascii=False, cls=DateTimeEncoder)
                elif not result.get("success"):
                    # 错误情况
                    tool_content_for_llm = json.dumps({
                        "tool": tool_name,
                        "status": "error",
                        "error": result.get("error", "Unknown error")
                    }, ensure_ascii=False)
                else:
                    # 其他工具保持原样，使用自定义 encoder 处理 datetime
                    tool_content_for_llm = json.dumps(result, ensure_ascii=False, cls=DateTimeEncoder)
                
                # 添加工具结果消息
                tool_msg = ChatMessage(
                    role="tool",
                    content=tool_content_for_llm,
                    tool_call_id=f"call_{len(session.messages)}"
                )
                session.messages.append(tool_msg)
            
            # 🔥🔥🔥 关键修复：不添加 Continue Prompt！
            # 训练数据格式：<tool_response>...</tool_response> 后面直接是 <|im_start|>assistant
            # 如果我们添加 Continue Prompt（role=user），会破坏这个模式，导致模型幻觉
            
            # 构建消息列表：system + 最近对话（包括刚添加的 tool 结果）
            summary_messages = []
            # 先添加 system message
            for msg in session.messages:
                if msg.role == "system":
                    summary_messages.append({"role": msg.role, "content": msg.content})
                    break
            # 再添加最近的对话消息（不过滤 system，因为上面已经添加了）
            for msg in session.messages[-15:]:  # 增加到15条以包含更多上下文
                if msg.role != "system":
                    summary_messages.append({"role": msg.role, "content": msg.content})

            logger.info(f"🔍 让 LLM 继续处理工具结果（不添加 Continue Prompt），上下文消息数: {len(summary_messages)}")
            summary_response = await self.qwen_service.chat_completion(
                messages=summary_messages,  # 🔥 不添加额外的 user 消息！
                tools=self._get_available_tools(),
                max_tokens=4096,
                temperature=0.6  # Qwen3 推荐值，适合工具调用/数据提取
            )
            
            logger.info(f"📊 LLM 响应: success={summary_response.get('success')}, message_length={len(summary_response.get('message', {}).get('content', ''))}")
            
            if summary_response.get("success"):
                assistant_response = summary_response["message"]
                
                # 🔥 关键改进：检查 LLM 是否要继续调用工具
                if assistant_response.get("tool_calls"):
                    logger.info(f"🔄 LLM 要继续调用 {len(assistant_response['tool_calls'])} 个工具，递归处理 (depth={depth+1})...")
                    # 递归调用 _handle_tool_calls，让 LLM 可以多轮调用工具
                    return await self._handle_tool_calls(session_id, assistant_response, depth=depth+1)
                else:
                    # LLM 决定不再调用工具，返回最终响应
                    summary_content = assistant_response["content"]
                    logger.info(f"✅ LLM 完成任务，内容预览: {summary_content[:100]}...")
                    
                    # 🔥 代码层防御：检查最终响应是否包含原始数据标记或 JSON 结构
                    has_raw_markers = ("|\n|" in summary_content or "|\\n|" in summary_content)
                    has_json_structure = (
                        ('"action_id"' in summary_content and '"record_type"' in summary_content) or
                        ('"session_id"' in summary_content and '"timestamp"' in summary_content) or
                        summary_content.count('"') > 50  # JSON 通常有大量引号
                    )
                    
                    if has_raw_markers or has_json_structure:
                        if has_raw_markers:
                            logger.warning(f"⚠️ [SYSTEM_DEFENSE] Final response contains raw data markers (|\\n|)")
                        if has_json_structure:
                            logger.warning(f"⚠️ [SYSTEM_DEFENSE] Final response contains JSON data structure")
                        logger.info(f"📝 Original content length: {len(summary_content)}")
                        logger.info(f"🔧 Replacing with fallback message")
                        summary_content = (
                            "✅ I have completed searching and recording data from the document.\n\n"
                            "The extracted information has been recorded in the database. "
                            "Please check the session summary for detailed action IDs and values."
                        )
                    
                    summary_msg = ChatMessage(
                        role="assistant",
                        content=summary_content
                    )
                    session.messages.append(summary_msg)
                    
                    return {
                        "success": True,
                        "session_id": session_id,
                        "message": summary_content,  # 🔥 使用清理后的 content
                        "message_type": "tool_response",
                        "tool_results": tool_results,
                        "usage": summary_response.get("usage", {})
                    }
            else:
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": "工具调用完成，但总结生成失败。",
                    "message_type": "tool_response",
                    "tool_results": tool_results
                }
        
        except Exception as e:
            logger.error(f"处理工具调用失败: {str(e)}")
            return {
                "success": False,
                "error": f"处理工具调用时出错: {str(e)}"
            }
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具定义"""
        # get_tools_schema() 现在返回OpenAI格式的工具列表
        tools_schema = self.tool_service.get_tools_schema()
        
        # 🔥 关键修改：从 Schema 中移除 session_id，避免 LLM 产生逻辑冲突
        # LLM 不需要知道 session_id 的存在，因为它会被 _handle_tool_calls 自动注入
        # 这解决了 System Prompt（告诉 LLM 不需要 session_id）与 Schema（标记为 required）之间的矛盾
        import copy
        filtered_tools = []
        removed_count = 0
        
        for tool in tools_schema:
            # 深拷贝工具定义，避免修改原始 schema
            tool_copy = copy.deepcopy(tool)
            
            if "function" in tool_copy and "parameters" in tool_copy["function"]:
                parameters = tool_copy["function"]["parameters"]
                tool_name = tool_copy["function"].get("name", "unknown")
                
                # 从 properties 中移除 session_id
                if "properties" in parameters and "session_id" in parameters["properties"]:
                    del parameters["properties"]["session_id"]
                    removed_count += 1
                    logger.info(f"🔧 已从工具 {tool_name} 的 Schema 中移除 session_id (properties)")
                
                # 从 required 列表中移除 session_id
                if "required" in parameters and "session_id" in parameters["required"]:
                    parameters["required"].remove("session_id")
                    logger.info(f"🔧 已从工具 {tool_name} 的 Schema 中移除 session_id (required)")
            
            filtered_tools.append(tool_copy)
        
        logger.info(f"✅ Schema 过滤完成: 共处理 {len(filtered_tools)} 个工具，移除了 {removed_count} 个 session_id")
        return filtered_tools
    
    def get_chat_history(self, session_id: str, limit: int = 20) -> Dict[str, Any]:
        """
        获取聊天历史
        
        Args:
            session_id: 会话ID
            limit: 返回消息数量限制
            
        Returns:
            Dict[str, Any]: 聊天历史
        """
        if session_id not in self.sessions:
            return {
                "success": False,
                "error": "会话不存在"
            }
        
        session = self.sessions[session_id]
        messages = session.messages[-limit:] if limit > 0 else session.messages
        
        # 转换为可序列化格式
        history = []
        for msg in messages:
            if msg.role != "system":  # 不返回系统消息
                history.append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "tool_calls": msg.tool_calls if msg.tool_calls else None
                })
        
        return {
            "success": True,
            "session_id": session_id,
            "messages": history,
            "total_messages": len(session.messages),
            "session_info": {
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat()
            }
        }
    
    def delete_chat_session(self, session_id: str) -> bool:
        """删除聊天会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"删除聊天会话: {session_id}")
            return True
        return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        active_sessions = 0
        total_messages = 0
        
        for session in self.sessions.values():
            if datetime.now() - session.last_activity < timedelta(seconds=self.session_timeout):
                active_sessions += 1
            total_messages += len(session.messages)
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "qwen_service_info": self.qwen_service.get_model_info()
        }
    
    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        while True:
            try:
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if current_time - session.last_activity > timedelta(seconds=self.session_timeout):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    logger.info(f"清理过期会话: {session_id}")
                
                # 每5分钟清理一次
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"清理过期会话失败: {str(e)}")
                await asyncio.sleep(300)
    
    def cleanup(self):
        """清理资源"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self.qwen_service.cleanup()
        self.sessions.clear()
        logger.info("LLM对话服务清理完成")
