"""
vLLM 服务
使用 vLLM 作为推理引擎，提供高性能的流式推理能力

使用方式：
1. 先启动 vLLM 服务器：
   python -m vllm.entrypoints.openai.api_server \
       --model /path/to/merged_model \
       --served-model-name qwen-lca \
       --max-model-len 16384 \
       --dtype bfloat16

2. 然后在代码中使用 VLLMService 连接
"""

import logging
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class VLLMService:
    """
    基于 vLLM 的 LLM 服务
    
    通过 OpenAI 兼容的 API 连接 vLLM 服务器
    支持流式输出和工具调用
    """
    
    def __init__(
        self,
        api_base: str = "http://localhost:8080/v1",
        api_key: str = "EMPTY",
        model_name: str = "qwen-lca",
        session_manager = None
    ):
        """
        初始化 vLLM 服务
        
        Args:
            api_base: vLLM API 地址
            api_key: API 密钥（vLLM 默认不需要）
            model_name: 模型名称（与 vLLM 启动时的 --served-model-name 一致）
            session_manager: 会话管理器
        """
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.session_manager = session_manager
        
        self.client = None
        self.is_initialized = False
        
        # PDF 处理器引用
        self.pdf_processor = None
        
        logger.info(f"VLLMService 初始化，API: {self.api_base}, 模型: {self.model_name}")
        
    def set_pdf_processor(self, pdf_processor):
        """设置 PDF 处理器"""
        self.pdf_processor = pdf_processor
        
    async def initialize(self):
        """异步初始化"""
        if self.is_initialized:
            return
            
        logger.info("🚀 初始化 vLLM 服务...")
        
        self.client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
        
        # 测试连接
        try:
            models = await self.client.models.list()
            available_models = [m.id for m in models.data]
            logger.info(f"✅ vLLM 连接成功，可用模型: {available_models}")
            
            if self.model_name not in available_models:
                logger.warning(f"⚠️ 指定的模型 {self.model_name} 不在可用列表中")
                if available_models:
                    self.model_name = available_models[0]
                    logger.info(f"   使用第一个可用模型: {self.model_name}")
                    
        except Exception as e:
            logger.error(f"❌ vLLM 连接失败: {e}")
            raise RuntimeError(f"无法连接 vLLM 服务: {e}")
            
        self.is_initialized = True
        
    def _build_system_prompt(self, pdf_session_id: str = None) -> str:
        """构建系统提示词（与训练数据一致）"""
        
        base_prompt = """You are an expert LCA assistant for Additive Manufacturing.

## Core Task
Extract quantitative LCI data from documents or answer specific questions.

## Tools
- search_document: Search text segments containing data via keywords
- define_lca_scope: Record Functional Unit
- record_process_flow: Record LCI flows (quantitative values)
- record_parameter: Record intermediate parameters for calculation
- execute_calculation: Calculate derived values
- get_session_summary: View all recorded data

## Strategic Workflow
1. Understand user intent → 2. Search document → 3. Extract & record data → 4. Summarize findings

## LCI Categories
**Inputs:** Raw Material, Process Energy, Post-processing Energy, Feedstock Energy, Gas, Cooling Media
**Outputs:** Product, Recovered Material, Waste, Emission

## Key Guidelines
- Use exact values from documents (no estimation)
- Always provide selected_chunk for traceability
- One tool call at a time, wait for response
- Summarize in natural language for user"""

        if pdf_session_id:
            base_prompt += f"\n\n## Document Context\nPDF_SESSION_ID: {pdf_session_id}"
            
        return base_prompt
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        聊天补全接口
        
        Args:
            messages: 对话消息列表
            tools: 可用工具列表
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            stream: 是否流式输出
            
        Returns:
            包含响应内容和工具调用的字典
        """
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # 转换消息格式
            formatted_messages = self._format_messages(messages)
            
            # 转换工具格式
            openai_tools = None
            if tools:
                openai_tools = self._format_tools(tools)
                
            if stream:
                return await self._stream_chat_completion(
                    formatted_messages, openai_tools, max_tokens, temperature
                )
            else:
                return await self._sync_chat_completion(
                    formatted_messages, openai_tools, max_tokens, temperature
                )
                
        except Exception as e:
            logger.error(f"Chat completion 失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _sync_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """同步聊天补全"""
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            # vLLM 特定参数
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True  # 启用思考模式
                }
            }
        )
        
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # 解析思考过程
        thinking_content = ""
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
        result = {
            "success": True,
            "message": {
                "role": "assistant",
                "content": content,
                "thinking": thinking_content
            },
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0
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
            logger.info(f"🔧 检测到 {len(tool_calls)} 个工具调用")
            
        return result
        
    async def _stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式聊天补全"""
        
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            }
        )
        
        full_content = ""
        tool_calls = []
        
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                
                if delta.content:
                    full_content += delta.content
                    yield {
                        "type": "content",
                        "content": delta.content
                    }
                    
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.function:
                            tool_calls.append({
                                "tool_name": tc.function.name,
                                "parameters": json.loads(tc.function.arguments) if tc.function.arguments else {}
                            })
                            yield {
                                "type": "tool_call",
                                "tool_call": tool_calls[-1]
                            }
                            
        # 最终结果
        yield {
            "type": "final",
            "success": True,
            "message": {
                "role": "assistant",
                "content": full_content,
                "tool_calls": tool_calls if tool_calls else None
            }
        }
        
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """格式化消息列表"""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "tool":
                # 工具响应转换为 user 消息
                formatted.append({
                    "role": "user",
                    "content": f"<tool_response>\n{content}\n</tool_response>"
                })
            else:
                formatted.append({
                    "role": role,
                    "content": content
                })
        
        # 🔥 修复：确保消息以 user 开头（vLLM 要求）
        # 如果第一条非 system 消息是 assistant，将其转换为 system 消息的一部分
        if formatted:
            # 找到第一条非 system 消息
            first_non_system_idx = None
            for i, msg in enumerate(formatted):
                if msg["role"] != "system":
                    first_non_system_idx = i
                    break
                    
            if first_non_system_idx is not None:
                first_msg = formatted[first_non_system_idx]
                if first_msg["role"] == "assistant":
                    # 将 assistant 消息转换为 system 消息
                    logger.info("🔧 将首条 assistant 消息转换为 system 消息")
                    formatted[first_non_system_idx] = {
                        "role": "system",
                        "content": f"[Document Context]\n{first_msg['content']}"
                    }
                
        return formatted
        
    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化工具定义为 OpenAI 格式"""
        formatted = []
        
        for tool in tools:
            if "function" in tool:
                formatted.append({
                    "type": "function",
                    "function": tool["function"]
                })
            elif "name" in tool:
                # 直接是函数定义
                formatted.append({
                    "type": "function",
                    "function": tool
                })
                
        return formatted
        
    async def simple_generate(self, prompt: str, max_tokens: int = 512) -> str:
        """简单文本生成"""
        messages = [{"role": "user", "content": prompt}]
        result = await self.chat_completion(messages, max_tokens=max_tokens)
        if result.get("success"):
            return result["message"]["content"]
        return ""
