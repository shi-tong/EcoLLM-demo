"""
Qwen-Agent 服务
使用 Qwen-Agent 框架替代原有的 LocalQwenService，提供更稳定的工具调用能力

架构说明：
1. 使用 Qwen-Agent 的 Assistant 类管理对话流程
2. 自定义工具类继承 BaseTool，与现有 tool_service 集成
3. 支持本地模型（通过 Transformers）或 API 模式（通过 vLLM/OpenAI）
"""

import logging
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class QwenAgentService:
    """
    基于 Qwen-Agent 的 LLM 服务
    
    支持两种模式：
    1. 本地模型模式：直接加载本地模型（适合有 GPU 的环境）
    2. API 模式：连接 vLLM/OpenAI 兼容的 API（适合分布式部署）
    """
    
    def __init__(
        self,
        model_path: str = None,
        lora_path: str = None,
        api_base: str = None,
        api_key: str = "EMPTY",
        use_local: bool = True,
        session_manager = None  # 兼容 LocalQwenService 的接口
    ):
        """
        初始化 Qwen-Agent 服务
        
        Args:
            model_path: 本地模型路径
            lora_path: LoRA 适配器路径
            api_base: API 服务地址（如 vLLM）
            api_key: API 密钥
            use_local: 是否使用本地模型
            session_manager: 会话管理器（兼容接口）
        """
        self.model_path = model_path or "/home/Research_work/24_yzlin/LCA-LLM/models/Qwen3-8B"
        self.lora_path = lora_path or "/home/Research_work/24_yzlin/LCA-LLM/models/lca_lora"
        self.api_base = api_base
        self.api_key = api_key
        self.use_local = use_local
        self.session_manager = session_manager
        
        self.model = None
        self.tokenizer = None
        self.llm = None
        self.is_initialized = False
        
        # PDF 处理器引用（用于获取文档上下文）
        self.pdf_processor = None
        
    def set_pdf_processor(self, pdf_processor):
        """设置 PDF 处理器"""
        self.pdf_processor = pdf_processor
        
    async def initialize(self):
        """异步初始化模型"""
        if self.is_initialized:
            return
            
        logger.info("🚀 初始化 Qwen-Agent 服务...")
        
        if self.use_local:
            await self._init_local_model()
        else:
            await self._init_api_model()
            
        self.is_initialized = True
        logger.info("✅ Qwen-Agent 服务初始化完成")
        
    async def _init_local_model(self):
        """初始化本地模型"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        logger.info(f"📦 加载本地模型: {self.model_path}")
        
        # 检查 GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法加载本地模型")
            
        # 4-bit 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载 LoRA 适配器
        if self.lora_path and Path(self.lora_path).exists():
            logger.info(f"🔧 加载 LoRA 适配器: {self.lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            
        self.model.eval()
        logger.info(f"✅ 本地模型加载完成，设备: {next(self.model.parameters()).device}")
        
    async def _init_api_model(self):
        """初始化 API 模式"""
        logger.info(f"🌐 连接 API: {self.api_base}")
        # API 模式下不需要加载模型，只需要配置连接信息
        
    def _build_system_prompt(self, pdf_session_id: str = None, mode: str = "standalone") -> str:
        """构建系统提示词"""
        
        # 基础系统提示词（与训练数据一致）
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

        # 如果有文档上下文，添加文档信息
        if pdf_session_id and self.pdf_processor:
            try:
                doc_info = self.pdf_processor.get_document_info(pdf_session_id)
                if doc_info:
                    doc_context = f"""

## Current Document Context
- Document: {doc_info.get('filename', 'Unknown')}
- Title: {doc_info.get('title', 'N/A')}
- Total Chunks: {doc_info.get('total_chunks', 'N/A')}

You have access to this document via search_document tool."""
                    base_prompt += doc_context
            except Exception as e:
                logger.warning(f"获取文档信息失败: {e}")
                
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
            if self.use_local:
                return await self._local_chat_completion(
                    messages, tools, max_tokens, temperature
                )
            else:
                return await self._api_chat_completion(
                    messages, tools, max_tokens, temperature
                )
        except Exception as e:
            logger.error(f"Chat completion 失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def _local_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """本地模型推理"""
        import torch
        
        # 提取 pdf_session_id
        pdf_session_id = None
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                match = re.search(r"PDF_SESSION_ID:\s*([a-f0-9-]+)", content)
                if match:
                    pdf_session_id = match.group(1)
                break
                
        # 构建 prompt
        prompt = self._build_chat_prompt(messages, tools, pdf_session_id)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = inputs.to(self.model.device)
        
        # 生成配置
        from transformers import GenerationConfig
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                generation_config=gen_config
            )
            
        # 解码
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # 解析响应
        return self._parse_response(response)
        
    async def _api_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """API 模式推理"""
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )
        
        # 转换工具格式为 OpenAI 格式
        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                if "function" in tool:
                    openai_tools.append(tool)
                    
        response = await client.chat.completions.create(
            model="qwen-agent",
            messages=messages,
            tools=openai_tools,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # 解析响应
        choice = response.choices[0]
        result = {
            "success": True,
            "message": {
                "role": "assistant",
                "content": choice.message.content or ""
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
        
    def _build_chat_prompt(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        pdf_session_id: str = None
    ) -> str:
        """构建聊天 prompt（Qwen3 格式）"""
        
        # 系统提示词
        system_prompt = self._build_system_prompt(pdf_session_id)
        
        # 添加工具定义
        if tools:
            tools_desc = "\n\n## Available Tools\n"
            for tool in tools:
                if "function" in tool:
                    func = tool["function"]
                    tools_desc += f"- {func['name']}: {func.get('description', '')}\n"
            system_prompt += tools_desc
            
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # 添加对话消息
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            
            if role == "system":
                continue  # 已经处理过
            elif role == "tool":
                # 工具响应格式
                prompt += f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
            else:
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                
        prompt += "<|im_start|>assistant\n"
        return prompt
        
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析模型响应"""
        
        # 清理 tool_response 幻觉
        response = re.sub(r'<tool_response>.*?</tool_response>', '', response, flags=re.DOTALL).strip()
        response = re.sub(r'</?tool_response>', '', response).strip()
        
        # 解析思考过程
        thinking_content = ""
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            thinking_content = think_match.group(1).strip()
            
        # 移除思考标签
        actual_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        
        # 解析工具调用
        tool_calls = []
        tool_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_pattern, actual_response, re.DOTALL)
        
        for match in matches:
            try:
                tool_data = json.loads(match)
                tool_calls.append({
                    "tool_name": tool_data.get("name"),
                    "parameters": tool_data.get("arguments", {})
                })
            except json.JSONDecodeError:
                continue
                
        # 移除工具调用标签，获取纯文本内容
        content = re.sub(tool_pattern, '', actual_response, flags=re.DOTALL).strip()
        
        result = {
            "success": True,
            "message": {
                "role": "assistant",
                "content": content,
                "thinking": thinking_content
            }
        }
        
        if tool_calls:
            result["message"]["tool_calls"] = tool_calls
            
        # 日志
        logger.info(f"📝 响应长度: {len(response)} 字符")
        if thinking_content:
            logger.info(f"💭 思考过程: {len(thinking_content)} 字符")
        if tool_calls:
            logger.info(f"🔧 工具调用: {len(tool_calls)} 个")
            for tc in tool_calls:
                logger.info(f"   - {tc['tool_name']}")
                
        return result
        
    async def simple_generate(self, prompt: str, max_tokens: int = 512) -> str:
        """简单文本生成"""
        messages = [{"role": "user", "content": prompt}]
        result = await self.chat_completion(messages, max_tokens=max_tokens)
        if result.get("success"):
            return result["message"]["content"]
        return ""
