from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import ast
import operator
import math
import re
import tempfile
import os
from typing import Optional, Dict, Any, List, Literal, Union
import uuid
import logging
from datetime import datetime
import requests
import asyncio
import queue
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置全局网络超时和重试策略
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
requests.adapters.DEFAULT_TIMEOUT = 120  # 2分钟超时

# 配置重试策略
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],  # 新版本urllib3使用allowed_methods
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

logger = logging.getLogger(__name__)

from backend.services.pdf_processor import PDFProcessor
from backend.services.knowledge_base import TemporaryKnowledgeBase
from backend.services.vectorized_knowledge_base import VectorizedPermanentLCIDatabase
from backend.services.llm_service import CoderLLMService
from backend.services.pylca_executor import PyLCAExecutor
from backend.services.session_manager import SessionManager
from backend.services.mongodb_manager import initialize_mongodb, mongodb_manager
from backend.services.tool_service import LCAToolService
from backend.services.local_qwen_service import LocalQwenService
from backend.services.llm_chat_service import LLMChatService
from backend.services.qwen_agent_service import QwenAgentService
from backend.services.qwen_agent_service_v2 import QwenAgentServiceV2
from backend.services.vllm_service import VLLMService

# 导入单位API路由
from backend.api.unit_api import router as unit_router
from backend.services.keyword_suggester import keyword_suggester
from backend.services.ecoinvent_matcher import get_ecoinvent_matcher
from backend.services.openlca_client import get_openlca_client
from backend.services.lcia_calculator import get_lcia_calculator

app = FastAPI(title="LCA-LLM Service", version="1.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务实例
# 使用表格感知的PDF处理器
from backend.services.table_aware_chunker import EnhancedTableAwarePDFProcessor
pdf_processor = EnhancedTableAwarePDFProcessor()
permanent_lci_db = VectorizedPermanentLCIDatabase()
llm_service = CoderLLMService()
pylca_executor = PyLCAExecutor()
session_manager = SessionManager(session_timeout=7200, cleanup_interval=300)  # 2小时超时，5分钟检查间隔

# 安全计算引擎
class SafeCalculator:
    """安全的数学表达式计算器"""
    
    # 允许的操作符
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # 允许的数学函数
    SAFE_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sqrt': math.sqrt,
        'pow': pow,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
    }
    
    def calculate(self, expression: str, variables: Dict[str, float] = None) -> Dict[str, Any]:
        """
        安全地计算数学表达式
        
        Args:
            expression: 数学表达式字符串
            variables: 变量字典
            
        Returns:
            Dict包含result, success, error等信息
        """
        try:
            # 输入验证
            if not expression or not isinstance(expression, str):
                return {"success": False, "error": "Invalid expression"}
            
            # 长度限制
            if len(expression) > 200:
                return {"success": False, "error": "Expression too long"}
            
            # 字符白名单检查（允许数学函数需要的字符）
            allowed_chars = set('0123456789+-*/().abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_, ')
            if not set(expression).issubset(allowed_chars):
                return {"success": False, "error": "Invalid characters in expression"}
            
            # 替换变量
            if variables:
                for var_name, var_value in variables.items():
                    if not isinstance(var_value, (int, float)):
                        return {"success": False, "error": f"Invalid variable value: {var_name}"}
                    expression = expression.replace(var_name, str(var_value))
            
            # 解析AST
            try:
                node = ast.parse(expression, mode='eval')
            except SyntaxError as e:
                return {"success": False, "error": f"Syntax error: {str(e)}"}
            
            # 计算结果
            result = self._eval_node(node.body)
            
            # 结果验证
            if not isinstance(result, (int, float)):
                return {"success": False, "error": "Invalid result type"}
            
            if math.isnan(result) or math.isinf(result):
                return {"success": False, "error": "Invalid result (NaN or Inf)"}
            
            return {
                "success": True,
                "result": float(result),
                "expression": expression,
                "error": None
            }
            
        except Exception as e:
            return {"success": False, "error": f"Calculation error: {str(e)}"}
    
    def _eval_node(self, node):
        """递归计算AST节点"""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.SAFE_FUNCTIONS:
                raise ValueError(f"Unsupported function: {func_name}")
            args = [self._eval_node(arg) for arg in node.args]
            return self.SAFE_FUNCTIONS[func_name](*args)
        elif isinstance(node, ast.Name):
            raise ValueError(f"Undefined variable: {node.id}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

# 初始化服务
calculator = SafeCalculator()
tool_service = LCAToolService(
    pdf_processor=pdf_processor,
    permanent_lci_db=permanent_lci_db,
    llm_service=llm_service,
    pylca_executor=pylca_executor,
    session_manager=session_manager
)

# 🔥 选择 LLM 服务：通过环境变量 LLM_SERVICE 控制
# LLM_SERVICE=vllm     使用 vLLM 服务（推荐，需要先启动 vLLM 服务器）
# LLM_SERVICE=qwen_agent 使用 Qwen-Agent 服务
# LLM_SERVICE=local    使用原有的 LocalQwenService（默认）
LLM_SERVICE = os.environ.get("LLM_SERVICE", "local").lower()

if LLM_SERVICE == "vllm":
    logger.info("🚀 使用 vLLM 服务")
    vllm_api_base = os.environ.get("VLLM_API_BASE", "http://localhost:8080/v1")
    qwen_service = VLLMService(
        api_base=vllm_api_base,
        session_manager=session_manager
    )
    qwen_service.set_pdf_processor(pdf_processor)
elif LLM_SERVICE == "qwen_agent":
    logger.info("🚀 使用 Qwen-Agent V2 服务 (vLLM + Qwen-Agent)")
    vllm_api_base = os.environ.get("VLLM_API_BASE", "http://localhost:8080/v1")
    # 🔥 支持选择模型：qwen-lca (基座) 或 lca_lora (LoRA微调)
    vllm_model_name = os.environ.get("VLLM_MODEL_NAME", "qwen-lca")
    logger.info(f"📦 使用模型: {vllm_model_name}")
    qwen_service = QwenAgentServiceV2(
        api_base=vllm_api_base,
        model_name=vllm_model_name,
        session_manager=session_manager
    )
    qwen_service.set_pdf_processor(pdf_processor)
    qwen_service.set_tool_service(tool_service)  # 🔥 设置工具服务供 Qwen-Agent 使用
else:
    logger.info("🔧 使用原有的 LocalQwenService")
    qwen_service = LocalQwenService(session_manager=session_manager)

llm_chat_service = LLMChatService(
    qwen_service=qwen_service,
    tool_service=tool_service,
    session_timeout=3600  # 1小时超时
)

@app.on_event("startup")
async def startup_event():
    """FastAPI启动事件"""
    try:
        # 初始化MongoDB连接池
        initialize_mongodb()
        logger.info("MongoDB连接池初始化成功")
        
        # 启动会话清理任务
        session_manager.start_cleanup_task()
        
        # 注意：LLM聊天服务采用延迟初始化，仅在首次使用时加载模型
        logger.info("LLM对话服务配置完成（延迟加载模式）")
        
        logger.info("LCA-LLM后端服务启动完成")
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise e

class LCARequest(BaseModel):
    session_id: str
    instruction: str
    parameters: Optional[Dict[str, Any]] = None

class LCAResponse(BaseModel):
    success: bool
    message: str
    generated_code: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SearchResponse(BaseModel):
    success: bool
    message: str
    pdf_context: Optional[List[Dict[str, Any]]] = None
    lci_data: Optional[List[Dict[str, Any]]] = None
    search_query: Optional[Union[str, List[str]]] = None
    error: Optional[str] = None

# 工具调用请求模型
class ProcessDocumentRequest(BaseModel):
    file_content: str  # base64编码的PDF内容
    filename: str
    search_focus: Optional[str] = None

class BuildLCASystemRequest(BaseModel):
    session_id: str
    system_description: str
    functional_unit: str
    impact_categories: Optional[List[str]] = None

class ExecuteAnalysisRequest(BaseModel):
    session_id: str
    analysis_type: str = "impact_assessment"

class SearchDocumentRequest(BaseModel):
    session_id: str
    query: Optional[str] = None              # 单查询（与queries二选一）
    queries: Optional[List[str]] = None      # 批量查询（与query二选一）
    max_results: Optional[int] = 5           # 单查询模式的结果数
    max_results_per_query: Optional[int] = 3 # 批量模式每个查询的结果数
    max_total_results: Optional[int] = 10    # 批量模式总结果数上限
    extract_mode: Optional[str] = "chunks"   # "chunks" | "sentences" | "key_points"
    min_similarity: Optional[float] = 0.3
    deduplicate: Optional[bool] = True       # 批量模式是否去重

class ToolResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """上传并处理PDF文件，创建临时知识库"""
    try:
        # 验证文件类型
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # 创建会话
        session_id = session_manager.create_session(file.filename)
        session_data = session_manager.get_session(session_id)
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 解析PDF
            documents = pdf_processor.process_pdf(temp_file_path)
            
            # 提取LCI信息
            lci_info = pdf_processor.extract_lci_information(documents)
            
            # 创建临时知识库
            temp_kb = TemporaryKnowledgeBase(
                collection_name=f"session_{session_id}"
            )
            temp_kb.add_documents(documents)
            
            # 保存到会话数据
            session_data.knowledge_base = temp_kb
            session_data.documents = documents
            session_data.metadata = {
                "lci_info": lci_info,
                "file_size": len(content),
                "pages": len(set(doc.metadata.get('page', 0) for doc in documents))
            }
            
            return {
                "success": True,
                "session_id": session_id,
                "message": f"PDF processed successfully, extracted {len(documents)} document chunks",
                "document_count": len(documents),
                "documents": [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    } for doc in documents
                ],
                "lci_summary": {
                    "products": len(lci_info.get("products", [])),
                    "processes": len(lci_info.get("processes", [])),
                    "materials": len(lci_info.get("materials", [])),
                    "energy_flows": len(lci_info.get("energy_flows", [])),
                    "emissions": len(lci_info.get("emissions", []))
                }
            }
        
        finally:
            # 清理临时PDF文件
            os.unlink(temp_file_path)
    
    except Exception as e:
        # 如果出错，清理已创建的会话
        if 'session_id' in locals():
            session_manager.delete_session(session_id)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/search-lci-data", response_model=SearchResponse)
async def search_lci_data(request: LCARequest):
    """搜索LCI数据并返回检索结果（不生成代码）"""
    try:
        # 检查会话是否存在
        session_data = session_manager.get_session(request.session_id)
        if not session_data or not session_data.knowledge_base:
            raise HTTPException(status_code=404, detail="Session not found or expired, please upload PDF again")
        
        temp_kb = session_data.knowledge_base
        
        # 1. 查询临时知识库获取PDF上下文
        raw_pdf_context = temp_kb.query(request.instruction)
        
        # 2. 智能处理PDF上下文，提取关键信息
        processed_pdf_context = pdf_processor.process_search_results(raw_pdf_context, request.instruction)
        
        # 3. 查询永久LCI知识库获取flow数据（使用更严格的过滤）
        lci_data = permanent_lci_db.search_flows(
            request.instruction, 
            processed_pdf_context,
            k=5,  # 限制返回数量
            similarity_threshold=0.3  # 设置相似度阈值
        )
        
        return SearchResponse(
            success=True,
            message=f"Search completed, found {len(lci_data)} LCI data entries",
            pdf_context=processed_pdf_context,
            lci_data=lci_data,
            search_query=request.instruction
        )
    
    except Exception as e:
        return SearchResponse(
            success=False,
            message="Error occurred during search",
            error=str(e)
        )

@app.post("/generate-lca-code", response_model=LCAResponse)
async def generate_lca_code(request: LCARequest):
    """根据用户指令生成pyLCA代码并执行"""
    try:
        # 检查会话是否存在
        session_data = session_manager.get_session(request.session_id)
        if not session_data or not session_data.knowledge_base:
            raise HTTPException(status_code=404, detail="Session not found or expired, please upload PDF again")
        
        temp_kb = session_data.knowledge_base
        
        # 1. 查询临时知识库获取PDF上下文
        pdf_context = temp_kb.query(request.instruction)
        
        # 2. 查询永久LCI知识库获取flow数据
        lci_data = permanent_lci_db.search_flows(request.instruction, pdf_context)
        
        # 3. 整合信息并生成pyLCA代码
        code_context = {
            "pdf_context": pdf_context,
            "lci_data": lci_data,
            "instruction": request.instruction,
            "parameters": request.parameters or {}
        }
        
        generated_code = llm_service.generate_pylca_code(code_context)
        
        # 4. 执行代码 
        execution_result = pylca_executor.execute_code(generated_code)
        
        return LCAResponse(
            success=True,
            message="Code generation and execution successful",
            generated_code=generated_code,
            execution_result=execution_result
        )
    
    except Exception as e:
        return LCAResponse(
            success=False,
            message="Error occurred while processing request",
            error=str(e)
        )

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """获取会话状态和详细信息"""
    session_data = session_manager.get_session(session_id)
    if session_data:
        return {
            "exists": True,
            "session_id": session_id,
            "original_filename": session_data.original_filename,
            "created_at": session_data.created_at.isoformat(),
            "last_accessed": session_data.last_accessed.isoformat(),
            "age_minutes": int(session_data.get_age().total_seconds() / 60),
            "idle_minutes": int(session_data.get_idle_time().total_seconds() / 60),
            "document_count": len(session_data.documents),
            "lci_summary": session_data.metadata.get("lci_info", {}),
            "file_info": {
                "size_bytes": session_data.metadata.get("file_size", 0),
                "pages": session_data.metadata.get("pages", 0)
            }
        }
    else:
        return {"exists": False}

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话和临时知识库"""
    if session_manager.delete_session(session_id):
        return {"success": True, "message": "Session deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions/stats")
async def get_sessions_stats():
    """获取会话统计信息"""
    return session_manager.get_session_stats()

@app.post("/sessions/cleanup")
async def cleanup_expired_sessions():
    """手动清理过期会话"""
    await session_manager.cleanup_expired_sessions()
    return {"success": True, "message": "Expired sessions cleanup completed"}

@app.get("/database/status")
async def get_database_status():
    """获取数据库连接池详细状态"""
    try:
        health_info = mongodb_manager.health_check()
        return {
            "success": True,
            "health_info": health_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# ==================== 工具调用API端点 ====================

@app.post("/tools/process-document", response_model=ToolResponse)
async def tool_process_document(request: ProcessDocumentRequest):
    """工坷1: 处理文档"""
    try:
        result = await tool_service.process_document(
            file_content=request.file_content,
            filename=request.filename,
            search_focus=request.search_focus
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data={
                    "session_id": result.get("session_id"),
                    **result.get("data", {})
                }
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error processing document: {str(e)}"
        )

@app.post("/tools/build-lca-system", response_model=ToolResponse)
async def tool_build_lca_system(request: BuildLCASystemRequest):
    """工坷2: 构建LCA系统"""
    try:
        result = await tool_service.build_lca_system(
            session_id=request.session_id,
            system_description=request.system_description,
            functional_unit=request.functional_unit,
            impact_categories=request.impact_categories
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data=result
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error building LCA system: {str(e)}"
        )

@app.post("/tools/search-document", response_model=ToolResponse)
async def tool_search_document(request: SearchDocumentRequest):
    """工具: 搜索文档内容（支持单查询和批量查询）"""
    try:
        result = await tool_service.search_document(
            session_id=request.session_id,
            query=request.query,
            queries=request.queries,
            max_results=request.max_results,
            max_results_per_query=request.max_results_per_query,
            max_total_results=request.max_total_results,
            extract_mode=request.extract_mode,
            min_similarity=request.min_similarity,
            deduplicate=request.deduplicate
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data=result
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error searching document: {str(e)}"
        )

class SearchLCIDatabaseRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.3

# ==================== 第一阶段新增工具请求模型 ====================


class ChunkInfo(BaseModel):
    """文档片段信息"""
    chunk_id: str
    content: str
    score: Optional[float] = 0.0  # 相似度分数，默认为0.0

class ExpertDecision(BaseModel):
    rationale: Optional[str] = None  # 专家决策理由

class DefineLCAScopeRequest(BaseModel):
    session_id: str
    parameter_name: str  # 'Function Unit', 'System Boundary', 'Geographical Scope'
    description: str
    value: Optional[float] = None
    unit: Optional[str] = None
    source_content: Optional[str] = None
    note: Optional[str] = None  # 🔥 NEW: 备注信息
    search_query: Optional[Union[str, List[str]]] = None
    search_context: Optional[List[ChunkInfo]] = None
    selected_chunk: Optional[ChunkInfo] = None
    intent: Optional[str] = "select_best"  # 动作意图: "select_best" | "refine_same" | "pivot_query"
    link_to: Optional[str] = None
    

class RecordProcessFlowRequest(BaseModel):
    session_id: str
    flow_type: str  # 'Input', 'Output' - simplified classification
    category: str   # 工作台的11个LCI类别: Input flows ('Raw Material', 'Process Energy', 'Post-processing Energy', 'Feedstock Energy', 'Gas', 'Cooling Media') or Output flows ('Product', 'Recovered Material', 'Waste', 'Emission')
    name: str
    value: float
    unit: str
    location: Optional[str] = None
    cas_number: Optional[str] = None      # New: CAS number for chemical identification
    process_name: Optional[str] = None    # New: Process step context
    note: Optional[str] = None  # 🔥 NEW: 备注信息
    search_query: Optional[Union[str, List[str]]] = None    # 🔥 NEW: 触发搜索的原始查询字符串
    search_context: Optional[List[ChunkInfo]] = None  # 完整搜索结果供LLM学习
    selected_chunk: Optional[ChunkInfo] = None     # 专家选择的最相关片段
    intent: Optional[str] = "select_best"  # 动作意图: "select_best" | "refine_same" | "pivot_query"
    link_to: Optional[str] = None  # 指向上一步动作的action_id

# 记录Pivot失败的请求模型
class RecordPivotFailureRequest(BaseModel):
    session_id: str
    failed_query: Union[str, List[str]]  # 失败的搜索查询（必需，支持字符串或数组）
    link_to: Optional[str] = None  # 连接到上一个action_id（可选）
    failed_context: Optional[List[ChunkInfo]] = None  # 失败查询返回的chunks（可选）

# 🔥 NEW: 专门记录智能跳过的请求模型
class RecordSmartSkipRequest(BaseModel):
    session_id: str
    category: str                  # 跳过的类别（必需）- 工作台的11个LCI类别
    skip_reason: str = "already_recorded"  # 跳过原因（默认 "already_recorded"）
    link_to: Optional[str] = None  # 连接到上一个action_id（可选）
    skipped_chunk: Optional[ChunkInfo] = None  # 被跳过的chunk信息（可选，用于数据收集）
    skip_rationale: Optional[str] = None       # 跳过说明（可选，用于数据收集）
    search_context: Optional[List[ChunkInfo]] = None  # 🔥 NEW: 完整的搜索结果（必需，用于训练数据）
    search_query: Optional[Union[str, List[str]]] = None  # 🔥 FIX: 支持字符串或数组（与 record-process-flow 一致）

class RecordParameterRequest(BaseModel):
    """记录参数请求 - 用于记录待用于计算的原始参数"""
    session_id: str
    parameter_name: str = Field(..., min_length=1, max_length=200, description="参数名称")
    parameter_value: float = Field(..., description="参数值")
    selected_chunk: ChunkInfo = Field(..., description="选择的文档块（作为证据）")
    
    # 可选字段
    parameter_unit: Optional[str] = Field(None, max_length=50, description="参数单位")
    link_to: Optional[str] = None
    intent: Literal["select_best", "refine_same"] = "select_best"
    
    # 数据收集字段（可选）
    note: Optional[str] = None
    search_query: Optional[Union[str, List[str]]] = None
    search_context: Optional[List[ChunkInfo]] = None

class RecordCalculationRequest(BaseModel):
    """记录计算动作请求"""
    session_id: str
    calculation_expression: str = Field(..., min_length=1, max_length=500, description="计算表达式")
    calculation_result: float = Field(..., description="计算结果")
    data_dependencies: List[str] = Field(..., description="依赖的参数action_id列表")
    
    # 可选字段
    calculation_unit: Optional[str] = Field(None, max_length=50, description="结果单位")
    link_to: Optional[str] = None
    intent: Literal["calculate"] = "calculate"

class CalculateRequest(BaseModel):
    """LLM调用的计算请求"""
    expression: str = Field(..., min_length=1, max_length=200, description="数学表达式，如 '10 * 5' 或 '(15 + 25) * 0.8'")
    variables: Optional[Dict[str, float]] = Field(None, description="变量值，如 {'Power': 10, 'Time': 5}")
    
class CalculateResponse(BaseModel):
    """计算响应"""
    result: float
    expression: str
    success: bool
    error: Optional[str] = None

class ExecuteCalculationRequest(BaseModel):
    """专家执行计算请求"""
    session_id: str
    expression: str = Field(..., min_length=1, max_length=200, description="数学表达式")
    variables: Optional[Dict[str, float]] = Field(None, description="变量值")

class ExecuteCalculationResponse(BaseModel):
    """执行计算响应"""
    success: bool
    result: Optional[float] = None
    expression: str
    error: Optional[str] = None

class GetSessionSummaryRequest(BaseModel):
    session_id: str


@app.post("/tools/search-lci-database", response_model=ToolResponse)
async def tool_search_lci_database(request: SearchLCIDatabaseRequest):
    """工具: 搜索LCI数据库"""
    try:
        result = await tool_service.search_lci_database(
            query=request.query,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data=result
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error searching LCI database: {str(e)}"
        )


@app.post("/tools/execute-analysis", response_model=ToolResponse)
async def tool_execute_analysis(request: ExecuteAnalysisRequest):
    """工坷3: 执行分析"""
    try:
        result = await tool_service.execute_analysis(
            session_id=request.session_id,
            analysis_type=request.analysis_type
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data=result
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error executing LCA analysis: {str(e)}"
        )

# ==================== 第一阶段新增工具API端点 ====================

@app.post("/tools/define-lca-scope", response_model=ToolResponse)
async def tool_define_lca_scope(request: DefineLCAScopeRequest):
    """工具: 定义LCA范围"""
    try:
        result = await tool_service.define_lca_scope(
            session_id=request.session_id,
            parameter_name=request.parameter_name,
            description=request.description,
            value=request.value,
            unit=request.unit,
            source_content=request.source_content,
            note=request.note,
            search_query=request.search_query,
            search_context=[chunk.dict() for chunk in request.search_context] if request.search_context else None,
            selected_chunk=request.selected_chunk.dict() if request.selected_chunk else None,
            intent=request.intent,
            link_to=request.link_to
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data=result
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error defining LCA scope: {str(e)}"
        )

@app.post("/tools/record-process-flow", response_model=ToolResponse)
async def tool_record_process_flow(request: RecordProcessFlowRequest):
    """工具: 记录工艺流"""
    try:
        # 转换ChunkInfo对象为字典
        selected_chunk_dict = None
        if request.selected_chunk:
            selected_chunk_dict = {
                "chunk_id": request.selected_chunk.chunk_id,
                "content": request.selected_chunk.content,
                "score": request.selected_chunk.score
            }
        
        search_context_list = None
        if request.search_context:
            search_context_list = [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "score": chunk.score
                }
                for chunk in request.search_context
            ]
        
        result = await tool_service.record_process_flow(
            session_id=request.session_id,
            flow_type=request.flow_type,
            category=request.category,
            name=request.name,
            value=request.value,
            unit=request.unit,
            location=request.location,
            cas_number=request.cas_number,
            process_name=request.process_name,
            note=request.note,
            search_query=request.search_query,
            search_context=search_context_list,
            selected_chunk=selected_chunk_dict,
            intent=request.intent,
            link_to=request.link_to
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data=result.get("data")  # 只返回内层的data，避免嵌套
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error recording process flow: {str(e)}"
        )

@app.post("/tools/record-failure", response_model=ToolResponse)
async def tool_record_failure(request: RecordPivotFailureRequest):
    """工具 - 记录Pivot失败动作"""
    try:
        # 转换failed_context为字典列表
        failed_context_list = []
        if request.failed_context:
            failed_context_list = [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "score": chunk.score
                }
                for chunk in request.failed_context
            ]
        
        # 调用服务层记录失败动作
        result = tool_service.record_pivot_failure(
            session_id=request.session_id,
            link_to=request.link_to,
            failed_query=request.failed_query,
            failed_context=failed_context_list
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message="Pivot failure recorded successfully",
                data={
                    "new_action_id": result.get("new_action_id"),
                    "new_intent": "pivot_query"
                }
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error recording pivot failure: {str(e)}"
        )

@app.post("/tools/record-smart-skip", response_model=ToolResponse)
async def tool_record_smart_skip(request: RecordSmartSkipRequest):
    """🔥 NEW: 工具 - 记录智能跳过动作"""
    try:
        # 转换skipped_chunk为字典
        skipped_chunk_dict = None
        if request.skipped_chunk:
            skipped_chunk_dict = {
                "chunk_id": request.skipped_chunk.chunk_id,
                "content": request.skipped_chunk.content,
                "score": request.skipped_chunk.score
            }
        
        # 🔥 NEW: 转换search_context为字典列表
        search_context_list = None
        if request.search_context:
            search_context_list = [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "score": chunk.score
                }
                for chunk in request.search_context
            ]
        
        # 调用服务层记录智能跳过动作
        result = tool_service.record_smart_skip(
            session_id=request.session_id,
            category=request.category,
            skip_reason=request.skip_reason,
            link_to=request.link_to,
            skipped_chunk=skipped_chunk_dict,
            skip_rationale=request.skip_rationale,
            search_context=search_context_list,
            search_query=request.search_query
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data={
                    "new_action_id": result.get("new_action_id"),
                    "new_intent": "smart_skip",
                    "category": result.get("category"),
                    "skip_reason": result.get("skip_reason")
                }
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error recording pivot failure: {str(e)}"
        )

@app.post("/tools/calculate")
async def tool_calculate(request: CalculateRequest):
    """工具: 执行数学计算 (供LLM调用)"""
    try:
        # 使用安全计算引擎执行计算
        result = calculator.calculate(request.expression, request.variables)
        
        if result["success"]:
            return CalculateResponse(
                result=result["result"],
                expression=result["expression"],
                success=True,
                error=None
            )
        else:
            return CalculateResponse(
                result=0.0,
                expression=request.expression,
                success=False,
                error=result["error"]
            )
            
    except Exception as e:
        return CalculateResponse(
            result=0.0,
            expression=request.expression,
            success=False,
            error=f"Calculation failed: {str(e)}"
        )

@app.post("/tools/execute-calculation", response_model=ExecuteCalculationResponse)
async def tool_execute_calculation(request: ExecuteCalculationRequest):
    """工具: 执行计算但不记录 (供专家验证使用)"""
    try:
        result = calculator.calculate(request.expression, request.variables)
        return ExecuteCalculationResponse(
            success=result["success"],
            result=result.get("result") if result["success"] else None,
            expression=result.get("expression", request.expression),
            error=result.get("error") if not result["success"] else None
        )
    except Exception as e:
        return ExecuteCalculationResponse(
            success=False,
            result=None,
            expression=request.expression,
            error=f"Calculation failed: {str(e)}"
        )

@app.post("/tools/record-parameter", response_model=ToolResponse)
async def tool_record_parameter(request: RecordParameterRequest):
    """工具: 记录参数 - 用于记录待用于计算的原始参数"""
    try:
        # 转换search_context为字典列表
        search_context_list = []
        if request.search_context:
            for chunk in request.search_context:
                chunk_dict = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "score": chunk.score
                }
                search_context_list.append(chunk_dict)
        
        # 转换selected_chunk
        selected_chunk_dict = {
            "chunk_id": request.selected_chunk.chunk_id,
            "content": request.selected_chunk.content,
            "score": request.selected_chunk.score
        }
        
        # 调用服务层
        result = tool_service.record_parameter(
            session_id=request.session_id,
            parameter_name=request.parameter_name,
            parameter_value=request.parameter_value,
            parameter_unit=request.parameter_unit,
            note=request.note,
            search_query=request.search_query,
            search_context=search_context_list,
            selected_chunk=selected_chunk_dict,
            intent=request.intent,
            link_to=request.link_to
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data={"new_action_id": result.get("new_action_id")}
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
    except Exception as e:
        logger.error(f"Error in record_parameter: {str(e)}")
        return ToolResponse(
            success=False,
            error=str(e)
        )

@app.post("/tools/record-calculation", response_model=ToolResponse)
async def tool_record_calculation(request: RecordCalculationRequest):
    """工具: 记录计算动作"""
    try:
        # 调用服务层
        result = tool_service.record_calculation(
            session_id=request.session_id,
            link_to=request.link_to,
            calculation_expression=request.calculation_expression,
            calculation_result=request.calculation_result,
            calculation_unit=request.calculation_unit,
            data_dependencies=request.data_dependencies,
            intent=request.intent
        )
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data={"new_action_id": result.get("new_action_id")}
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error recording calculation: {str(e)}"
        )

@app.get("/tools/session-summary/{session_id}", response_model=ToolResponse)
async def tool_get_session_summary(
    session_id: str,
    format: str = Query("text", regex="^(text|json)$", description="Return format: 'text' for LLM (default, 80% token savings), 'json' for expert workbench"),
    view: str = Query("llm", regex="^(llm|workbench)$", description="View type: 'llm' for training data (default), 'workbench' for UI")
):
    """
    工具: 获取会话总结
    
    🔥 重要：view="llm"时只包含flow和parameter，保证训练-推理一致性
    
    Args:
        session_id: 会话ID
        format: 返回格式
            - "text": 人类可读文本格式（LLM 默认使用，节省 80% tokens）
            - "json": 完整 JSON 结构（专家工作台使用）
        view: 视图类型
            - "llm": LLM视图，只包含flow和parameter（默认，用于训练数据）
            - "workbench": 工作台视图，包含所有统计（用于UI显示）
    """
    try:
        result = await tool_service.get_session_summary(session_id=session_id, format=format, view=view)
        
        if result.get("success"):
            return ToolResponse(
                success=True,
                message=result.get("message"),
                data=result
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error getting session summary: {str(e)}"
        )

@app.post("/actions/record-summary-check", response_model=ToolResponse)
async def record_summary_check(request: dict):
    """
    记录专家的 get_session_summary 调用动作
    
    这是为 SFT 数据导出准备的，记录专家在何时、为何调用 summary 检查。
    这对于教会 LLM "P → S → C" (Parameter → Summary → Calculation) 链条至关重要。
    
    Args:
        request: {
            "session_id": str,
            "rationale": str,       # 专家填写的理由（为什么此时需要查看 summary）
            "timestamp": str
        }
    """
    try:
        session_id = request.get("session_id")
        rationale = request.get("rationale", "")
        timestamp = request.get("timestamp", datetime.now().isoformat())
        
        db = mongodb_manager.get_database()
        
        # 🔥 获取当前会话统计（与前端展示完全一致）
        # ⚠️ 使用workbench视图，包含完整统计
        summary_result = await tool_service.get_session_summary(session_id=session_id, format="json", view="workbench")
        
        # 🔥 提取与前端展示一致的统计信息 + 具体记录内容
        summary_snapshot = {
            "statistics": {
                "scope_count": 0,
                "flow_count": 0,
                "calc_count": 0,
                "pivot_count": 0,
                "total_actions": 0
            },
            "recorded_flows": [],
            "recorded_parameters": [],
            "recorded_calculations": []
        }
        
        if summary_result.get("success"):
            # 修正：summary_result 的 data 字段直接就是 summary 数据
            summary_data = summary_result.get("data", {})
            statistics = summary_data.get("statistics", {})
            calculation_analysis = summary_data.get("calculation_analysis", {})
            pivot_analysis = summary_data.get("pivot_analysis", {})
            process_flows = summary_data.get("process_flows", {})
            parameter_analysis = summary_data.get("parameter_analysis", {})
            
            scope_count = statistics.get("total_scopes_defined", 0)
            flow_count = statistics.get("total_flows_recorded", 0)
            calc_count = calculation_analysis.get("total_calculations", 0)
            pivot_count = pivot_analysis.get("total_pivots", 0)
            
            # 🔥 提取具体的 flow 记录（像前端 Extraction Log 一样）
            recorded_flows = []
            inputs = process_flows.get("inputs", {})
            outputs = process_flows.get("outputs", {})
            
            for category_key, flows in inputs.items():
                for flow in flows:
                    recorded_flows.append({
                        "type": "Input",
                        "category": category_key,
                        "name": flow.get("name"),
                        "value": flow.get("value"),
                        "unit": flow.get("unit"),
                        "action_id": flow.get("action_id")
                    })
            
            for category_key, flows in outputs.items():
                for flow in flows:
                    recorded_flows.append({
                        "type": "Output",
                        "category": category_key,
                        "name": flow.get("name"),
                        "value": flow.get("value"),
                        "unit": flow.get("unit"),
                        "action_id": flow.get("action_id")
                    })
            
            # 🔥 提取 parameters 和 calculations
            parameters = parameter_analysis.get("parameters", [])
            calculations = calculation_analysis.get("calculations", [])
            
            summary_snapshot = {
                "statistics": {
                    "scope_count": scope_count,
                    "flow_count": flow_count,
                    "calc_count": calc_count,
                    "pivot_count": pivot_count,
                    "total_actions": scope_count + flow_count + calc_count
                },
                "recorded_flows": recorded_flows,  # 具体的 flow 记录
                "recorded_parameters": parameters,  # 具体的 parameter 记录
                "recorded_calculations": calculations  # 具体的 calculation 记录
            }
        
        # 🔥 生成action_id（summary_check也是LLM正常流程的一部分）
        # 排除特殊action_id如ACT_PREVIEW
        last_action = db.lca_actions.find_one(
            {"session_id": session_id, "action_id": {"$regex": "^ACT_\\d{4}$"}},
            sort=[("action_id", -1)]
        )
        
        if last_action and "action_id" in last_action:
            # 从 "ACT_0001" 提取数字部分并递增
            try:
                last_num = int(last_action["action_id"].split("_")[1])
                new_action_id = f"ACT_{last_num + 1:04d}"
            except (ValueError, IndexError):
                new_action_id = "ACT_0001"
        else:
            new_action_id = "ACT_0001"
        
        # 🔥 构建 summary_check 记录（现在有action_id和created_at）
        # 这是LLM正常流程的一部分，用于训练LLM学会何时回顾session状态
        action_record = {
            "session_id": session_id,
            "action_id": new_action_id,
            "record_type": "summary_check",
            "tool_name": "get_session_summary",
            "summary_snapshot": summary_snapshot,
            "created_at": datetime.now().isoformat(),
            "timestamp": timestamp
        }
        
        # 🔥 仅在有 rationale 时添加该字段
        if rationale and rationale.strip():
            action_record["expert_rationale"] = rationale.strip()
        
        # 存储到 MongoDB
        result = db.lca_actions.insert_one(action_record)
        
        logger.info(f"✅ Recorded summary check for session {session_id}")
        
        return ToolResponse(
            success=True,
            message=f"Summary check recorded successfully",
            data={"session_id": session_id, "timestamp": timestamp}
        )
        
    except Exception as e:
        logger.error(f"Failed to record summary check: {str(e)}")
        return ToolResponse(
            success=False,
            error=f"Failed to record summary check: {str(e)}"
        )

@app.delete("/tools/clear-session/{session_id}", response_model=ToolResponse)
async def tool_clear_session_data(session_id: str):
    """清空会话数据"""
    try:
        db = mongodb_manager.get_database()
        
        # 删除LCA范围定义
        result1 = db.lca_scopes.delete_many({"session_id": session_id})
        
        # 删除LCA动作记录
        result2 = db.lca_actions.delete_many({"session_id": session_id})
        
        return ToolResponse(
            success=True,
            message=f"Session data cleared: {result1.deleted_count} scopes, {result2.deleted_count} flows",
            data={
                "session_id": session_id,
                "deleted_scopes": result1.deleted_count,
                "deleted_flows": result2.deleted_count
            }
        )
    except Exception as e:
        logger.error(f"Clear session error: {str(e)}")
        return ToolResponse(
            success=False,
            message="Failed to clear session data",
            error=str(e)
        )

class RecordDocumentPreviewRequest(BaseModel):
    session_id: str

@app.post("/tools/record-document-preview", response_model=ToolResponse)
async def tool_record_document_preview(request: RecordDocumentPreviewRequest):
    """🔥 NEW: 记录文档预览（chunk 0 和 chunk 1）用于上下文感知检索训练"""
    try:
        result = await tool_service.record_document_preview(
            session_id=request.session_id
        )
        
        if result["success"]:
            return ToolResponse(
                success=True,
                message=result["message"],
                data=result.get("data")
            )
        else:
            return ToolResponse(
                success=False,
                message="Failed to record document preview",
                error=result.get("error")
            )
    except Exception as e:
        logger.error(f"Record document preview error: {str(e)}")
        return ToolResponse(
            success=False,
            message="Failed to record document preview",
            error=str(e)
        )

@app.get("/tools/schema")
async def get_tools_schema():
    """获取工具Schema（供LLM使用）"""
    return {
        "success": True,
        "tools": tool_service.get_tools_schema(),
        "description": "LCA Tool Suite - Supports complete LCA analysis workflow"
    }

# ==================== LLM对话API端点 ====================

class CreateChatSessionRequest(BaseModel):
    session_id: Optional[str] = None
    pdf_session_id: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    include_tools: bool = True

class ChatResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    message: Optional[str] = None
    thinking: Optional[str] = None  # 🔥 新增：思考过程
    message_type: Optional[str] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ChatHistoryResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    total_messages: Optional[int] = None
    session_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.post("/chat/create-session")
async def create_chat_session(request: CreateChatSessionRequest):
    """创建新的聊天会话（支持独立模式）"""
    try:
        logger.info(f"[创建聊天会话] session_id={request.session_id}, pdf_session_id={request.pdf_session_id}")
        new_session_id = llm_chat_service.create_chat_session(request.session_id, request.pdf_session_id)
        
        # 确定模式
        mode = "document_based" if request.pdf_session_id else "standalone"
        logger.info(f"[聊天会话创建完成] new_session_id={new_session_id}, mode={mode}")
        
        return {
            "success": True,
            "session_id": new_session_id,
            "message": f"Chat session created successfully in {mode} mode",
            "pdf_session_id": request.pdf_session_id,
            "mode": mode
        }
    except Exception as e:
        logger.error(f"Chat session creation failed: {str(e)}")
        return {
            "success": False,
            "error": f"Failed to create session: {str(e)}"
        }

@app.post("/chat/message", response_model=ChatResponse)
async def send_chat_message(request: ChatRequest):
    """发送聊天消息"""
    try:
        logger.warning(f"🎯🎯🎯 API /chat/message 被调用 - session_id={request.session_id}, message前20字={request.message[:20] if request.message else 'None'}")
        print(f"🎯🎯🎯 API /chat/message 被调用 - session_id={request.session_id}")
        
        # 如果没有提供session_id，自动创建一个
        if not request.session_id:
            session_id = llm_chat_service.create_chat_session()
        else:
            session_id = request.session_id
        
        logger.warning(f"🔵 准备调用 llm_chat_service.chat() - session_id={session_id}")
        print(f"🔵 准备调用 llm_chat_service.chat() - session_id={session_id}")
        
        result = await llm_chat_service.chat(
            session_id=session_id,
            user_message=request.message,
            include_tools=request.include_tools
        )
        
        logger.warning(f"🟢 llm_chat_service.chat() 返回 - success={result.get('success')}")
        print(f"🟢 llm_chat_service.chat() 返回 - success={result.get('success')}")
        
        if result.get("success"):
            return ChatResponse(
                success=True,
                session_id=result["session_id"],
                message=result["message"],
                thinking=result.get("thinking"),  # 🔥 新增：思考过程
                message_type=result.get("message_type", "text"),
                tool_results=result.get("tool_results"),
                usage=result.get("usage")
            )
        else:
            return ChatResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        logger.error(f"处理聊天消息失败: {str(e)}")
        return ChatResponse(
            success=False,
            error=f"处理消息失败: {str(e)}"
        )

@app.post("/chat/stream")
async def stream_chat_message(request: ChatRequest):
    """
    流式聊天端点 - 使用 llm_chat_service 确保正确的 system prompt
    
    在后台执行完整的 chat 调用，完成后推送结果
    """
    import json as json_module
    
    async def generate_stream():
        try:
            session_id = request.session_id
            
            # 发送开始状态
            yield f"event: status\ndata: {json_module.dumps({'type': 'status', 'content': 'Processing...'})}\n\n"
            
            # 使用 llm_chat_service.chat() - 这会正确处理 system prompt 和工具调用
            result = await llm_chat_service.chat(
                session_id=session_id,
                user_message=request.message,
                include_tools=request.include_tools
            )
            
            if result.get("success"):
                content = result.get("message", "")
                thinking = result.get("thinking", "")
                tool_results = result.get("tool_results", [])
                
                # 发送 thinking
                if thinking:
                    yield f"event: thinking\ndata: {json_module.dumps({'type': 'thinking', 'content': thinking}, ensure_ascii=False)}\n\n"
                
                # 发送工具结果
                if tool_results:
                    for tr in tool_results:
                        yield f"event: tool_result\ndata: {json_module.dumps({'type': 'tool_result', 'tool_name': tr.get('tool_name', ''), 'result': str(tr.get('result', ''))[:200]}, ensure_ascii=False)}\n\n"
                
                # 发送内容
                if content:
                    yield f"event: content\ndata: {json_module.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
                
                # 发送完成
                yield f"event: done\ndata: {json_module.dumps({'type': 'done', 'success': True, 'content': content, 'thinking': thinking}, ensure_ascii=False)}\n\n"
            else:
                error = result.get("error", "Unknown error")
                yield f"event: error\ndata: {json_module.dumps({'type': 'error', 'error': error}, ensure_ascii=False)}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"event: error\ndata: {json_module.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str, limit: int = 20):
    """获取聊天历史"""
    try:
        result = llm_chat_service.get_chat_history(session_id, limit)
        
        if result.get("success"):
            return ChatHistoryResponse(
                success=True,
                session_id=result["session_id"],
                messages=result["messages"],
                total_messages=result["total_messages"],
                session_info=result["session_info"]
            )
        else:
            return ChatHistoryResponse(
                success=False,
                error=result.get("error")
            )
            
    except Exception as e:
        logger.error(f"获取聊天历史失败: {str(e)}")
        return ChatHistoryResponse(
            success=False,
            error=f"获取历史失败: {str(e)}"
        )

@app.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """删除聊天会话"""
    try:
        success = llm_chat_service.delete_chat_session(session_id)
        if success:
            return {
                "success": True,
                "message": "会话删除成功"
            }
        else:
            return {
                "success": False,
                "error": "会话不存在"
            }
    except Exception as e:
        logger.error(f"删除聊天会话失败: {str(e)}")
        return {
            "success": False,
            "error": f"删除会话失败: {str(e)}"
        }

@app.get("/chat/info")
async def get_chat_service_info():
    """获取聊天服务信息"""
    try:
        info = llm_chat_service.get_session_info()
        return {
            "success": True,
            "info": info
        }
    except Exception as e:
        logger.error(f"获取聊天服务信息失败: {str(e)}")
        return {
            "success": False,
            "error": f"获取信息失败: {str(e)}"
        }

# ==================== Think生成API端点 ====================

def auto_detect_v30_scenario(request: "ThinkSuggestionRequest") -> str:
    """v3.1场景自动检测 - 支持推理场景的手动触发"""
    
    # Intent优先级检测
    if request.intent == "refine_same":
        return "refine_same_chunk"
    
    if request.intent == "pivot_query":
        if request.ambiguous:
            return "record_parameter_ambiguous"
        return "pivot_query_no_results"
    
    # 工具特定场景
    if request.tool == "record_parameter":
        if request.uncertain:
            return "record_parameter_uncertain"
        elif request.ambiguous:
            return "record_parameter_ambiguous"
        elif request.inference_required:  # 🆕 v3.1: 推理场景优先级
            return "record_parameter_with_inference"
        else:
            return "record_parameter_direct"
    
    elif request.tool == "record_process_flow":
        if request.scope_unclear:
            return "record_flow_uncertain"
        elif request.from_calculation or request.last_intent == "calculate":
            return "record_flow_from_calculation"
        elif request.inference_required:  # 🆕 v3.1: 推理场景优先级
            if request.flow_type == "Input":
                return "record_flow_input_with_inference"
            elif request.flow_type == "Output":
                return "record_flow_output_with_inference"
            else:
                return "record_parameter_with_inference"  # 如果flow_type不明确，当作参数处理
        else:
            # v3.0: 根据flow_type智能选择Input/Output场景
            # 如果前端没有提供flow_type，默认为uncertain
            if request.flow_type == "Input":
                return "record_flow_input"
            elif request.flow_type == "Output":
                return "record_flow_output"
            else:
                return "record_flow_uncertain"
    
    # v3.0: 删除元工具场景 - record_calculation是专家工作台专用
    # LLM运行时使用execute_calculation_verify
    
    return None

def build_data_dict(request: "ThinkSuggestionRequest", scenario: str) -> dict:
    """构建数据字典 - 极简但完整"""
    
    if scenario in ["refine_same_chunk"]:
        chunk_id = request.selected_chunk.get('chunk_id', 'Unknown') if request.selected_chunk else 'Unknown'
        return {'chunk_id': chunk_id}
    
    elif scenario in ["pivot_query_no_results", "record_parameter_ambiguous"]:
        return {'query': request.search_query or 'search query'}
    
    elif "parameter" in scenario:
        if request.parameter_name and request.parameter_value is not None:
            param_name = request.parameter_name
            param_value = request.parameter_value
            param_unit = request.parameter_unit or ""
            
            # v3.0/v3.1: 不同参数场景需要不同数据格式
            if scenario in ["record_parameter_direct"]:
                # direct场景期望{name}{value}{unit}格式
                return {
                    'name': param_name,
                    'value': param_value,
                    'unit': param_unit
                }
            elif scenario in ["record_parameter_with_inference"]:
                # v3.1-fixed: inference场景需要raw_evidence和specific_material格式
                return {
                    'raw_evidence': request.raw_evidence or f"Raw_{param_name}",  # 使用专家填写的原始证据
                    'value': param_value,
                    'unit': param_unit,
                    'specific_material': param_name  # 推理结果
                }
            else:
                # uncertain/ambiguous场景期望{data}格式
                data_str = f"{param_value} {param_unit} {param_name}".strip()
                return {'data': data_str}
        else:
            data_str = "the extracted parameter value"
            return {'data': data_str}
    
    elif "flow" in scenario:
        # v3.0: 适配新的flow场景命名 (record_flow_input, record_flow_output, etc.)
        if request.parameter_name and request.parameter_value is not None:
            param_name = request.parameter_name
            param_value = request.parameter_value
            param_unit = request.parameter_unit or ""
            
            # v3.0/v3.1: 不同场景需要不同数据格式
            if scenario in ["record_flow_uncertain"]:
                # uncertain场景期望{data}格式
                data_str = f"{param_value} {param_unit} {param_name}".strip()
                return {'data': data_str}
            elif scenario in ["record_flow_input_with_inference", "record_flow_output_with_inference"]:
                # v3.1-fixed: inference场景需要raw_evidence和specific_material格式
                return {
                    'raw_evidence': request.raw_evidence or f"Raw_{param_name}",  # 使用专家填写的原始证据
                    'value': param_value,
                    'unit': param_unit,
                    'specific_material': param_name,  # 推理结果
                    'note': 'for Reuse' if 'reuse' in param_name.lower() else ''  # 自动检测标注
                }
            else:
                # input/output场景期望{name}{value}{unit}格式
                return {
                    'name': param_name,
                    'value': param_value, 
                    'unit': param_unit
                }
        else:
            return {'data': 'the extracted flow data'}
    
    elif "calculation" in scenario:
        return {
            'param1': 'parameter_1',
            'param2': 'parameter_2', 
            'id1': '0001',
            'id2': '0002'
        }
    
    return {'data': 'default data'}

class ThinkSuggestionRequest(BaseModel):
    """Think生成请求 - v3.0支持清晰LCI术语和自然决策流程"""
    selected_chunk: Optional[Dict[str, Any]] = None
    intent: str  # select_best, refine_same, pivot_query
    tool: str  # record_parameter, record_process_flow, record_calculation
    search_query: Optional[str] = None
    parameter_name: Optional[str] = None
    parameter_value: Optional[float] = None
    parameter_unit: Optional[str] = None
    # v2.1字段（保持兼容）
    uncertain: Optional[bool] = False  # 不确定性标记
    ambiguous: Optional[bool] = False  # 模糊性标记
    scope_unclear: Optional[bool] = False  # 边界不清
    from_calculation: Optional[bool] = False  # 来自计算
    last_intent: Optional[str] = None  # 上一个操作的intent
    # v3.0新增字段
    flow_type: Optional[str] = None  # Input/Output - 智能场景选择
    # v3.1新增字段
    inference_required: Optional[bool] = False  # 需要推理 - 手动触发推理场景
    # v3.1-fixed新增字段
    raw_evidence: Optional[str] = None  # 文档原始证据 - 用于推理场景的Assertion

@app.post("/api/generate-think-suggestions")
async def generate_think_suggestions(request: ThinkSuggestionRequest):
    """生成<think>块候选"""
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
        from think_generator import ThinkGenerator
        
        generator = ThinkGenerator()
        
        # v3.0自动场景检测 - 清晰LCI术语，自然决策流程
        scenario = auto_detect_v30_scenario(request)
        logger.info(f"场景检测结果: tool={request.tool}, intent={request.intent} -> scenario={scenario}")
        data_dict = build_data_dict(request, scenario)
        
        # 场景检测失败
        if not scenario:
            return {
                "success": False,
                "error": f"Could not determine scenario for tool: {request.tool}, intent: {request.intent}"
            }
        
        # 生成候选
        candidates = generator.generate_candidates(
            scenario=scenario,
            data_dict=data_dict,
            num_candidates=5,
            include_connectors=False
        )
        
        # 提取think文本
        suggestions = [c['think'] for c in candidates]
        
        return {
            "success": True,
            "suggestions": suggestions,
            "scenario": scenario
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"生成think建议失败: {str(e)}\n{error_details}")
        return {
            "success": False,
            "error": f"{str(e)} (Check server logs for details)"
        }

# ==================== 关键词建议API端点 ====================

class KeywordSuggestRequest(BaseModel):
    """关键词建议请求"""
    category: str = Field(..., description="LCI类别，如 'Process Energy'")
    min_keywords: int = Field(5, ge=3, le=10, description="最少关键词数量")
    max_keywords: int = Field(8, ge=5, le=15, description="最多关键词数量")
    extended_count: Optional[int] = Field(None, description="扩展词数量（None=随机1-2个）")

class KeywordSuggestResponse(BaseModel):
    """关键词建议响应"""
    success: bool
    category: str
    keywords: List[str]
    breakdown: Dict[str, List[str]]  # {"core": [...], "extended": [...]}
    error: Optional[str] = None

@app.post("/keywords/suggest", response_model=KeywordSuggestResponse)
async def suggest_keywords(request: KeywordSuggestRequest):
    """
    生成关键词建议
    
    基于两层关键词体系：
    - 核心词（Core）：必选的高频基础词汇
    - 扩展词（Extended）：随机抽样的专业/长尾词汇
    """
    try:
        # 生成关键词建议
        keywords = keyword_suggester.suggest_keywords(
            category=request.category,
            min_keywords=request.min_keywords,
            max_keywords=request.max_keywords,
            extended_count=request.extended_count
        )
        
        # 获取分类信息
        category_keywords = keyword_suggester.get_category_keywords(request.category)
        
        # 区分哪些是核心词，哪些是扩展词
        core_used = [kw for kw in keywords if kw in category_keywords["core"]]
        extended_used = [kw for kw in keywords if kw in category_keywords["extended"]]
        
        return KeywordSuggestResponse(
            success=True,
            category=request.category,
            keywords=keywords,
            breakdown={
                "core": core_used,
                "extended": extended_used
            }
        )
    except Exception as e:
        logger.error(f"关键词建议失败: {str(e)}")
        return KeywordSuggestResponse(
            success=False,
            category=request.category,
            keywords=[],
            breakdown={"core": [], "extended": []},
            error=str(e)
        )

@app.get("/keywords/categories")
async def get_keyword_categories():
    """获取所有可用的LCI类别"""
    try:
        categories = keyword_suggester.get_all_categories()
        return {
            "success": True,
            "categories": categories
        }
    except Exception as e:
        logger.error(f"获取类别失败: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/keywords/category/{category}")
async def get_category_keywords(category: str):
    """获取指定类别的完整关键词库（核心+扩展）"""
    try:
        keywords = keyword_suggester.get_category_keywords(category)
        return {
            "success": True,
            "category": category,
            "keywords": keywords
        }
    except Exception as e:
        logger.error(f"获取类别关键词失败: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# ==================== 单位相关API端点 ====================

# 注册单位API路由
app.include_router(unit_router)

# ==================== Ecoinvent 匹配 API ====================

class MatchFlowRequest(BaseModel):
    """匹配流请求"""
    flow_name: str = Field(..., description="流名称")
    category: Optional[str] = Field(None, description="LCI 类别")
    flow_type: Optional[str] = Field(None, description="流类型 (Input/Output)")
    top_k: int = Field(5, description="返回结果数量")

class ConfirmMatchRequest(BaseModel):
    """确认匹配请求"""
    session_id: str
    action_id: str
    ecoinvent_uuid: str

@app.post("/ecoinvent/match-flow")
async def match_ecoinvent_flow(request: MatchFlowRequest):
    """将用户提取的流与 ecoinvent 数据库匹配"""
    try:
        matcher = get_ecoinvent_matcher()
        results = matcher.match_flow(
            flow_name=request.flow_name,
            category=request.category,
            flow_type=request.flow_type,
            top_k=request.top_k
        )
        return {
            "success": True,
            "query": request.flow_name,
            "matches": results
        }
    except Exception as e:
        logger.error(f"匹配流失败: {e}")
        return {"success": False, "error": str(e)}

@app.get("/ecoinvent/match-session/{session_id}")
async def match_session_flows(session_id: str):
    """批量匹配会话中的所有 LCI 数据"""
    try:
        matcher = get_ecoinvent_matcher()
        results = matcher.batch_match_session(session_id)
        return results
    except Exception as e:
        logger.error(f"批量匹配失败: {e}")
        return {"success": False, "error": str(e)}

@app.post("/ecoinvent/confirm-match")
async def confirm_ecoinvent_match(request: ConfirmMatchRequest):
    """确认一个匹配结果"""
    try:
        matcher = get_ecoinvent_matcher()
        result = matcher.confirm_match(
            session_id=request.session_id,
            action_id=request.action_id,
            ecoinvent_uuid=request.ecoinvent_uuid
        )
        return result
    except Exception as e:
        logger.error(f"确认匹配失败: {e}")
        return {"success": False, "error": str(e)}

@app.get("/ecoinvent/search-flows")
async def search_ecoinvent_flows(
    query: str = Query(..., description="搜索关键词"),
    limit: int = Query(10, description="返回数量")
):
    """直接搜索 ecoinvent flows"""
    try:
        db = mongodb_manager.get_database()
        results = list(db.flows.find(
            {"name": {"$regex": query, "$options": "i"}},
            {"uuid": 1, "name": 1, "category": 1, "flowType": 1, "_id": 0}
        ).limit(limit))
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return {"success": False, "error": str(e)}

# ==================== openLCA IPC API ====================

class OpenLCAConfigRequest(BaseModel):
    """openLCA 配置请求"""
    host: str = Field("localhost", description="openLCA IPC 主机地址")
    port: int = Field(8080, description="openLCA IPC 端口")

@app.get("/openlca/test")
async def test_openlca_connection():
    """测试 openLCA 连接"""
    try:
        client = get_openlca_client()
        result = client.test_connection()
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/openlca/configure")
async def configure_openlca(request: OpenLCAConfigRequest):
    """配置 openLCA IPC 地址"""
    try:
        from backend.services.openlca_client import OpenLCAClient
        global _client_instance
        _client_instance = OpenLCAClient(host=request.host, port=request.port)
        result = _client_instance.test_connection()
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/openlca/flows")
async def get_openlca_flows(limit: int = Query(50, description="返回数量")):
    """获取 openLCA 中的流"""
    try:
        client = get_openlca_client()
        return client.get_flows(limit=limit)
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/openlca/impact-methods")
async def get_openlca_impact_methods():
    """获取 openLCA 中的影响评价方法"""
    try:
        client = get_openlca_client()
        return client.get_impact_methods()
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================== LCIA 计算 API ====================

@app.get("/lcia/sessions")
async def get_all_lci_sessions():
    """获取所有有 LCI 数据的 session 列表"""
    try:
        calculator = get_lcia_calculator()
        return calculator.get_all_sessions()
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/lcia/session/{session_id}/data")
async def get_session_lci_data(session_id: str):
    """获取会话的 LCI 数据"""
    try:
        calculator = get_lcia_calculator()
        return calculator.get_session_lci_data(session_id)
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/lcia/methods")
async def get_lcia_methods(limit: int = Query(50, description="返回数量")):
    """获取可用的 LCIA 方法列表"""
    try:
        calculator = get_lcia_calculator()
        return calculator.get_lcia_methods(limit=limit)
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/lcia/methods/{method_uuid}")
async def get_lcia_method_details(method_uuid: str):
    """获取 LCIA 方法详情"""
    try:
        calculator = get_lcia_calculator()
        return calculator.get_lcia_method_details(method_uuid)
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/lcia/session/{session_id}/match")
async def match_session_flows(session_id: str, use_llm: bool = False):
    """批量匹配会话中所有流
    
    Args:
        session_id: 会话 ID
        use_llm: 是否使用 LLM 辅助重写流名称以提高匹配精度（默认 False）
    """
    try:
        calculator = get_lcia_calculator()
        return calculator.match_all_flows(session_id, use_llm_rewrite=use_llm)
    except Exception as e:
        return {"success": False, "error": str(e)}

class LCIACalculationRequest(BaseModel):
    """LCIA 计算请求"""
    lcia_method_uuid: str = Field(..., description="LCIA 方法 UUID")
    flow_mappings: List[Dict] = Field(default=[], description="流映射列表")

@app.post("/lcia/session/{session_id}/calculate")
async def calculate_lcia(session_id: str, request: LCIACalculationRequest):
    """执行 LCIA 计算"""
    try:
        calculator = get_lcia_calculator()
        return calculator.calculate_lcia(
            session_id=session_id,
            lcia_method_uuid=request.lcia_method_uuid,
            flow_mappings=request.flow_mappings
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==================== 系统管理API端点 ====================

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查MongoDB连接池状态
        db_health = mongodb_manager.health_check()
        
        return {
            "status": "healthy", 
            "service": "LCA-LLM Backend",
            "database": db_health,
            "tools_available": len(tool_service.get_tools_schema()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "LCA-LLM Backend", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
