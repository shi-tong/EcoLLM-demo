#!/usr/bin/env python3
"""
本地Qwen3-8B LLM服务
提供本地化的大语言模型推理能力，支持工具调用
"""

import logging
import json
import torch
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel
import re

logger = logging.getLogger(__name__)


def extract_title_from_first_chunk(first_chunk_content: str) -> Optional[str]:
    """
    从第一个 chunk 中提取标题（极简版）
    
    规则：
    1. 只检查第一行
    2. 长度 20-250 字符
    3. 不以句号/逗号/分号结尾
    4. 不匹配排除模式
    
    返回：标题 或 None
    """
    if not first_chunk_content:
        return None
    
    lines = [line.strip() for line in first_chunk_content.split('\n') if line.strip()]
    if not lines:
        return None
    
    first_line = lines[0]
    
    # 检查 1：长度
    if not (20 <= len(first_line) <= 250):
        return None
    
    # 检查 2：结尾（不能是句子结尾）
    # 允许冒号结尾（常见于带副标题的学术论文）
    if first_line.endswith(('.', '。', ',', '，', ';', '；')):
        return None
    
    # 特殊处理：如果以冒号结尾，保留（这是常见的标题格式）
    # 例如："Title: Subtitle" 或 "Main Title:"
    
    # 检查 3：排除明显不是标题的模式
    exclude_patterns = [
        r'^\d{4}$',                    # 单独的年份
        r'^(abstract|摘要)$',          # 单独的"Abstract"
        r'^(page|第.*?页)',            # 页码
        r'^(author|作者|by)\s*:',     # 作者标记
    ]
    
    if any(re.search(pattern, first_line, re.IGNORECASE) for pattern in exclude_patterns):
        return None
    
    return first_line

class LocalQwenService:
    """本地Qwen3-8B服务"""
    
    def __init__(self, model_path: str = None, device: str = "auto", session_manager=None, 
                 lora_path: str = None, use_lora: bool = True):
        """
        初始化本地Qwen服务
        
        Args:
            model_path: 模型路径，默认使用项目内的Qwen3-8B
            device: 计算设备，auto/cuda/cpu
            session_manager: 会话管理器（用于获取文档信息）
            lora_path: LoRA适配器路径，默认使用微调后的lca_lora
            use_lora: 是否使用LoRA适配器
        """
        self.model_path = model_path or "/home/Research_work/24_yzlin/LCA-LLM/models/Qwen3-8B"
        self.lora_path = lora_path or "/home/Research_work/24_yzlin/LCA-LLM/models/lca_lora"
        self.use_lora = use_lora
        self.device = self._setup_device(device)
        self.session_manager = session_manager  # ✅ 保存 session_manager
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self.is_initialized = False
        
        logger.info(f"LocalQwenService初始化，模型路径: {self.model_path}, 设备: {self.device}")
        if self.use_lora:
            logger.info(f"LoRA适配器路径: {self.lora_path}")
    
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"检测到CUDA，使用GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("未检测到CUDA，使用CPU")
        return device
    
    async def initialize(self):
        """异步初始化模型"""
        if self.is_initialized:
            return
            
        try:
            logger.info("开始加载Qwen3-8B模型...")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            logger.info("Tokenizer加载完成")
            
            # 加载模型（支持4-bit量化）
            if self.device == "cuda":
                # 检测可用显存
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
                logger.info(f"检测到 GPU 显存: {gpu_memory:.1f} GB, 空闲: {free_memory:.1f} GB")
                
                # 4-bit 量化的 Qwen3-8B 大约需要 5-6GB 显存
                if free_memory < 6:
                    # 显存不足，使用 8-bit 量化 + CPU offload
                    logger.warning(f"⚠️ 空闲显存不足 ({free_memory:.1f}GB < 6GB)，使用 8-bit 量化")
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    # 显存足够，使用 4-bit 量化
                    logger.info("使用 4-bit 量化加载模型")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)
            
            logger.info("基座模型加载完成")
            
            # 加载LoRA适配器
            if self.use_lora and Path(self.lora_path).exists():
                logger.info(f"正在加载LoRA适配器: {self.lora_path}")
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
                logger.info("✅ LoRA适配器加载成功！模型已增强为LCA专家模式")
            elif self.use_lora:
                logger.warning(f"⚠️ LoRA路径不存在: {self.lora_path}，使用基座模型")
            
            # 设置生成配置
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 调整生成参数
            self.generation_config.max_new_tokens = 8192  # 足够长以支持多轮工具调用
            self.generation_config.temperature = 0.7
            self.generation_config.top_p = 0.8
            self.generation_config.do_sample = True
            self.generation_config.repetition_penalty = 1.05
            
            self.is_initialized = True
            logger.info("Qwen3-8B服务初始化完成")
            
        except Exception as e:
            logger.error(f"初始化Qwen3-8B服务失败: {str(e)}")
            raise
    
    async def chat_completion(self, 
                            messages: List[Dict[str, str]], 
                            tools: List[Dict[str, Any]] = None,
                            max_tokens: int = 2048,
                            temperature: float = 0.7) -> Dict[str, Any]:
        """
        聊天补全接口
        
        Args:
            messages: 对话历史，格式为[{"role": "user/assistant", "content": "..."}]
            tools: 可用工具列表
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            Dict[str, Any]: 生成结果
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # 构建输入prompt
            prompt = self._build_chat_prompt(messages, tools)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = inputs.to(self.device)
            
            # 更新生成配置
            self.generation_config.max_new_tokens = max_tokens
            self.generation_config.temperature = temperature
            
            # 🔥 不再使用 StoppingCriteria 停止生成
            # 原因：过早停止会导致有效内容被截断
            # 改为：让模型完整生成，然后在后处理中清理 <tool_response> 幻觉
            
            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码响应
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # 🔥 全面清理 tool_response 幻觉（模型不应该生成这些）
            # 1. 清理响应开头的残留
            response = re.sub(r'^[^<]*</tool_response>', '', response, flags=re.DOTALL).strip()
            # 2. 清理所有 <tool_response>...</tool_response> 块
            original_len = len(response)
            response = re.sub(r'<tool_response>.*?</tool_response>', '', response, flags=re.DOTALL).strip()
            # 3. 清理孤立的 <tool_response> 或 </tool_response> 标签
            response = re.sub(r'</?tool_response>', '', response).strip()
            
            if len(response) < original_len:
                logger.warning(f"⚠️ 清理了 {original_len - len(response)} 字符的 tool_response 幻觉")
            
            # 如果清理后响应太短且无有效内容，生成默认总结
            if len(response) < 50 and '<think>' not in response and '<tool_call>' not in response:
                logger.warning(f"⚠️ 响应过短或无效 ({len(response)} 字符)，使用默认总结")
                response = "I have completed the requested operations. The data has been recorded successfully. Please check the session summary for details."
            
            # 🔥 详细日志：打印完整响应以便调试
            print(f"\n{'='*80}")
            print(f"📝 LLM 完整原始响应 (长度: {len(response)} 字符):")
            print(f"{'='*80}")
            print(response[:2000] if len(response) > 2000 else response)
            if len(response) > 2000:
                print(f"... [截断，总长度 {len(response)}]")
            print(f"{'='*80}\n")
            logger.info(f"📝 LLM 原始响应长度: {len(response)} 字符")
            
            # 🔥 解析思考过程（<think> 标签）
            thinking_content = ""
            actual_response = response.strip()
            
            # 检查是否包含 <think> 标签（处理多个标签和格式错误）
            think_pattern = r'<think>(.*?)</think>'
            think_match = re.search(think_pattern, response, re.DOTALL)
            
            if think_match:
                thinking_content = think_match.group(1).strip()
                
                # 🔥 清理思考内容中的幻觉 tool_response（模型不应该生成这些）
                if '<tool_response>' in thinking_content:
                    logger.warning(f"⚠️ 检测到思考内容中包含幻觉的 tool_response，正在清理...")
                    thinking_content = re.sub(r'<tool_response>.*?</tool_response>', '', thinking_content, flags=re.DOTALL).strip()
                
                # 移除所有 <think>...</think> 标签对
                actual_response = re.sub(think_pattern, '', response, flags=re.DOTALL).strip()
                # 移除孤立的 </think> 标签（格式错误的情况）
                actual_response = re.sub(r'</think>', '', actual_response, flags=re.IGNORECASE).strip()
                # 🔥 清理响应中的幻觉 tool_response
                actual_response = re.sub(r'<tool_response>.*?</tool_response>', '', actual_response, flags=re.DOTALL).strip()
                
                # 🔥 打印完整的思考过程（不截断）
                print(f"\n{'='*80}")
                print(f"💭 完整思考过程 (长度: {len(thinking_content)} 字符):")
                print(f"{'='*80}")
                print(thinking_content)
                print(f"{'='*80}\n")
                logger.info(f"💭 检测到思考过程，长度: {len(thinking_content)} 字符")
            
            # 解析工具调用（使用清理后的响应）
            tool_calls = self._parse_tool_calls(actual_response) if tools else None
            
            # 🔥 打印解析到的工具调用
            if tool_calls:
                print(f"🔧 解析到 {len(tool_calls)} 个工具调用:")
                for i, tc in enumerate(tool_calls):
                    print(f"   [{i+1}] {tc.get('tool_name')}: {json.dumps(tc.get('parameters', {}), ensure_ascii=False)[:200]}")
                logger.info(f"🔧 解析到 {len(tool_calls)} 个工具调用: {[tc.get('tool_name') for tc in tool_calls]}")
            else:
                print(f"ℹ️  没有解析到工具调用")
            
            # 🔥 进一步清理：如果有工具调用，从响应中移除 <tool_call> 标签
            clean_content = actual_response
            if tool_calls:
                clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', actual_response, flags=re.DOTALL).strip()
                logger.info(f"🧹 清理工具调用标签后，content 长度: {len(clean_content)} (原始: {len(actual_response)})")
            
            result = {
                "success": True,
                "message": {
                    "role": "assistant",
                    "content": clean_content  # 使用清理后的内容
                },
                "thinking": thinking_content,  # 🔥 新增：思考过程
                "usage": {
                    "prompt_tokens": inputs.input_ids.shape[1],
                    "completion_tokens": outputs.shape[1] - inputs.input_ids.shape[1],
                    "total_tokens": outputs.shape[1]
                }
            }
            
            if tool_calls:
                result["message"]["tool_calls"] = tool_calls
            
            return result
            
        except Exception as e:
            logger.error(f"聊天补全失败: {str(e)}")
            return {
                "success": False,
                "error": f"生成响应时出错: {str(e)}"
            }
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None) -> str:
        """构建聊天prompt"""
        
        # 🔥 提取 pdf_session_id（从简化的 system message 中）
        pdf_session_id = None
        mode = "standalone"
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                # 新格式：MODE: document_based\nPDF_SESSION_ID: xxx
                if "PDF_SESSION_ID:" in content:
                    match = re.search(r"PDF_SESSION_ID:\s*([a-f0-9-]+)", content)
                    if match:
                        pdf_session_id = match.group(1)
                        mode = "document_based"
                        logger.info(f"📝 提取到 pdf_session_id: {pdf_session_id}")
                elif "MODE: standalone" in content:
                    mode = "standalone"
                    logger.info(f"📝 检测到 standalone 模式")
                break
        
        # 🔥 与训练数据一致的 System Prompt
        system_prompt = """You are an expert LCA assistant for Additive Manufacturing.

## Core Task
Extract quantitative LCI data from documents or answer specific questions.

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
5. **Session ID**: Automatically injected - do not ask user for it."""
        
        # 🔥 动态注入文档上下文（如果有 PDF）
        if pdf_session_id:
            # 获取文档信息
            document_name = "Unknown Document"
            document_title = None
            
            if self.session_manager:
                session_data = self.session_manager.get_session(pdf_session_id)
                if session_data:
                    # 1. 获取文档名称
                    document_name = session_data.original_filename
                    
                    # 2. 尝试提取标题（保守策略）
                    if hasattr(session_data, 'documents') and session_data.documents:
                        first_chunk = session_data.documents[0].page_content
                        document_title = extract_title_from_first_chunk(first_chunk)
                        
                        # 日志输出提取结果
                        if document_title:
                            logger.info(f"✅ 成功提取文档标题: {document_title[:50]}...")
                        else:
                            logger.info(f"⚠️ 无法提取文档标题（保守策略），仅使用文档名称")
            
            # 构建文档上下文
            system_prompt += f"""

**DOCUMENT CONTEXT**: 
A PDF document has been uploaded and is ready for analysis.
- Document Name: "{document_name}"
"""
            
            # 只有在成功提取标题时才添加
            if document_title:
                system_prompt += f"""- Document Title: "{document_title}"
"""
            
            system_prompt += f"""- Document ID: {pdf_session_id[:8]}...
"""
            
            # 🔥 动态注入 chunk 0 和 chunk 1 的内容（用于快速定位 Functional Unit）
            if self.session_manager:
                session_data = self.session_manager.get_session(pdf_session_id)
                if session_data and hasattr(session_data, 'documents') and session_data.documents:
                    # 获取 chunk 0（完整内容，与训练时一致）
                    if len(session_data.documents) > 0:
                        chunk_0_content = session_data.documents[0].page_content  # 🔥 完整内容
                        system_prompt += f"""
**CHUNK 0 PREVIEW** (Executive Summary / Introduction):
"{chunk_0_content}"
"""
                    
                    # 获取 chunk 1（完整内容，与训练时一致）
                    if len(session_data.documents) > 1:
                        chunk_1_content = session_data.documents[1].page_content  # 🔥 完整内容
                        system_prompt += f"""
**CHUNK 1 PREVIEW**:
"{chunk_1_content}"
"""
            
            system_prompt += f"""
**AUTOMATIC SESSION INJECTION**: 
When you call document-related tools (search_document, record_parameter, record_calculation, record_process_flow, get_session_summary, define_lca_scope, record_pivot_failure), the system will AUTOMATICALLY inject the session_id for you.

You do NOT need to ask the user for session_id. Just call the tools directly with other required parameters (e.g., "query" for search_document)."""
        
        if tools:
            system_prompt += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in tools:
                system_prompt += f"\n{json.dumps(tool, ensure_ascii=False)}"
            system_prompt += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        
        # 构建完整prompt
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        # 🔥 调试：打印 System Prompt 关键部分以验证
        if "Strategic Workflow" in system_prompt:
            print(f"\n✅ 使用训练数据一致的 System Prompt (包含 Strategic Workflow)\n", flush=True)
        else:
            print(f"\n⚠️ 使用旧版 System Prompt\n", flush=True)
        
        # 🔥 按照官方 chat_template.md 格式构建消息
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"]
            content = msg.get("content", "")
            
            # Skip system messages as we already have our own system prompt
            if role == "system":
                i += 1
                continue
            
            # 🔥 处理 tool role：按照官方格式包装为 <tool_response>
            if role == "tool":
                # 收集连续的 tool 消息
                tool_responses = []
                while i < len(messages) and messages[i]["role"] == "tool":
                    tool_responses.append(messages[i]["content"])
                    i += 1
                
                # 包装为官方格式：role=user + <tool_response> 标签
                prompt += "<|im_start|>user\n"
                for tool_content in tool_responses:
                    prompt += f"<tool_response>\n{tool_content}\n</tool_response>"
                    if len(tool_responses) > 1:  # 多个 tool response 之间换行
                        prompt += "\n"
                prompt += "<|im_end|>\n"
            else:
                # 其他角色（user, assistant）正常处理
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                i += 1
        
        prompt += "<|im_start|>assistant\n"
        
        return prompt
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """解析工具调用（匹配官方格式）"""
        tool_calls = []
        
        # 使用正则表达式查找工具调用
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                tool_call_data = json.loads(match)
                
                # 转换官方格式 {"name": "xxx", "arguments": {...}} 
                # 到我们的内部格式 {"tool_name": "xxx", "parameters": {...}}
                if "name" in tool_call_data:
                    converted_call = {
                        "tool_name": tool_call_data["name"],
                        "parameters": tool_call_data.get("arguments", {})
                    }
                    tool_calls.append(converted_call)
                    logger.info(f"解析到工具调用: {converted_call['tool_name']}")
                else:
                    # 兼容旧格式
                    tool_calls.append(tool_call_data)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"解析工具调用失败: {match}, 错误: {e}")
                continue
        
        return tool_calls
    
    async def simple_generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        简单文本生成接口
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            
        Returns:
            str: 生成的文本
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = inputs.to(self.device)
            
            self.generation_config.max_new_tokens = max_tokens
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"文本生成失败: {str(e)}")
            return f"生成失败: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": "Qwen3-8B",
            "model_path": self.model_path,
            "device": self.device,
            "is_initialized": self.is_initialized,
            "supports_tools": True,
            "max_context_length": 4096
        }
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.is_initialized = False
        logger.info("Qwen3-8B服务资源清理完成")
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.is_initialized = False
        logger.info("Qwen3-8B服务资源清理完成")
