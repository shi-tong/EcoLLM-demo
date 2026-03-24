#!/usr/bin/env python3
"""
LCA工具服务封装层
将现有API功能封装为三个核心工具，供LLM调用
"""

import logging
import tempfile
import os
import base64
import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

from .pdf_processor import PDFProcessor
from .knowledge_base import TemporaryKnowledgeBase
from .vectorized_knowledge_base import VectorizedPermanentLCIDatabase
from .llm_service import CoderLLMService
from .pylca_executor import PyLCAExecutor
from .session_manager import SessionManager

logger = logging.getLogger(__name__)

class LCAToolService:
    """LCA工具服务 - 封装三个核心工具"""
    
    def __init__(self, 
                 pdf_processor: PDFProcessor,
                 permanent_lci_db: VectorizedPermanentLCIDatabase,
                 llm_service: CoderLLMService,
                 pylca_executor: PyLCAExecutor,
                 session_manager: SessionManager):
        """
        初始化工具服务
        
        Args:
            pdf_processor: PDF处理器
            permanent_lci_db: 永久LCI知识库
            llm_service: LLM服务
            pylca_executor: pyLCA执行器
            session_manager: 会话管理器
        """
        self.pdf_processor = pdf_processor
        self.permanent_lci_db = permanent_lci_db
        self.llm_service = llm_service
        self.pylca_executor = pylca_executor
        self.session_manager = session_manager
        
        # 初始化语义匹配用的 embedding 模型（复用与知识库相同的模型）
        self._init_semantic_matcher()
        
        logger.info("LCA工具服务初始化完成")
    
    def _init_semantic_matcher(self):
        """
        初始化语义匹配器，用于 coverage boost 的语义计算
        复用 all-MiniLM-L6-v2 模型
        """
        local_model_path = "/home/Research_work/24_yzlin/LCA-LLM/models/all-MiniLM-L6-v2"
        
        try:
            if os.path.exists(local_model_path):
                self.semantic_model = SentenceTransformer(local_model_path)
                logger.info(f"✅ 语义匹配器加载本地模型: {local_model_path}")
            else:
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("✅ 语义匹配器加载在线模型: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"⚠️ 语义匹配器初始化失败，回退到精确匹配: {e}")
            self.semantic_model = None
    
    def _save_temp_file(self, file_content: str, filename: str) -> str:
        """保存临时文件并返回路径"""
        try:
            # 解码base64内容
            file_data = base64.b64decode(file_content)
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=Path(filename).suffix or '.pdf'
            )
            temp_file.write(file_data)
            temp_file.close()
            
            logger.info(f"临时文件已保存: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"保存临时文件失败: {str(e)}")
            raise Exception(f"Failed to save temporary file: {str(e)}")
    
    async def process_document(self, 
                             file_content: str, 
                             filename: str, 
                             search_focus: str = None) -> Dict[str, Any]:
        """
        工具1: 处理文档 (修正版 - 纯文档预处理器)
        
        职责说明:
        - 仅负责PDF文档的基础预处理工作
        - 生成唯一的session_id用于后续工具调用
        - 提取文档全文并创建向量化索引
        - 不进行LCI数据的自动提取和分析
        
        Args:
            file_content: PDF文件的base64编码内容
            filename: 文件名
            search_focus: 保留参数，但不在此阶段使用
            
        Returns:
            Dict[str, Any]: 处理结果（简化版）
        """
        try:
            logger.info(f"开始处理文档: {filename}")
            
            # 1. 创建会话
            session_id = self.session_manager.create_session(filename)
            session_data = self.session_manager.get_session(session_id)
            
            # 2. 保存临时文件
            temp_file_path = self._save_temp_file(file_content, filename)
            
            try:
                # 3. 解析PDF，提取完整文本并创建文档片段
                full_text, documents = self.pdf_processor.process_pdf_and_get_full_text(temp_file_path, session_id)
                logger.info(f"PDF解析完成，共提取{len(documents)}个文档片段，文本长度{len(full_text)}")
                
                # 5. 创建临时知识库（仅用于后续search_document工具）
                temp_kb = TemporaryKnowledgeBase(
                    collection_name=f"session_{session_id}"
                )
                temp_kb.add_documents(documents)
                
                # 6. 保存到会话数据（简化的元数据）
                session_data.knowledge_base = temp_kb
                session_data.documents = documents
                session_data.metadata = {
                    "filename": filename,
                    "file_size": len(base64.b64decode(file_content)),
                    "total_pages": len(set(doc.metadata.get('page', 0) for doc in documents)),
                    "total_chunks": len(documents),
                    "processing_timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"文档预处理完成: {filename}")
                
                # 返回简化的结果，符合第一阶段目标
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": f"文档处理成功，共创建{len(documents)}个文本分块",
                    "data": {
                        "filename": filename,
                        "total_pages": len(set(doc.metadata.get('page', 0) for doc in documents)),
                        "total_chunks": len(documents),
                        "full_text": full_text
                    }
                }
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            
        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            # 清理会话（如果已创建）
            if 'session_id' in locals():
                self.session_manager.delete_session(session_id)
            return {
                "success": False,
                "error": f"处理文档时出错: {str(e)}"
            }
    
    async def build_lca_system(self, 
                             session_id: str,
                             system_description: str,
                             functional_unit: str,
                             impact_categories: List[str] = None) -> Dict[str, Any]:
        """
        工具2: 构建LCA系统
        基于文档信息和LCI数据构建完整的LCA系统并生成pyLCA代码
        
        Args:
            session_id: 会话ID
            system_description: LCA系统描述，如'光伏板生产系统'
            functional_unit: 功能单位，如'1 kWp'、'1 kg产品'
            impact_categories: 关注的影响类别，如['GWP', 'AP', 'EP']
            
        Returns:
            Dict[str, Any]: 构建结果
        """
        try:
            logger.info(f"开始构建LCA系统: {system_description}")
            
            # 1. 检查会话是否存在
            session_data = self.session_manager.get_session(session_id)
            if not session_data or not session_data.knowledge_base:
                return {
                    "success": False,
                    "error": "会话不存在或已过期，请重新上传PDF"
                }
            
            temp_kb = session_data.knowledge_base
            
            # 2. 构建系统建模指令
            impact_categories = impact_categories or ["GWP"]
            instruction = f"""
            构建{system_description}的LCA模型:
            - 功能单位: {functional_unit}
            - 影响类别: {', '.join(impact_categories)}
            - 基于已上传的文档信息
            - 包含完整的系统边界和流程
            """
            
            # 3. 查询临时知识库获取PDF上下文
            pdf_context = temp_kb.query(instruction)
            
            # 4. 查询永久LCI知识库获取flow数据
            lci_data = self.permanent_lci_db.search_flows(instruction, pdf_context)
            
            # 5. 整合信息并生成pyLCA代码
            code_context = {
                "pdf_context": pdf_context,
                "lci_data": lci_data,
                "instruction": instruction,
                "parameters": {
                    "system_description": system_description,
                    "functional_unit": functional_unit,
                    "impact_categories": impact_categories
                }
            }
            
            # 6. 生成代码
            generated_code = self.llm_service.generate_pylca_code(code_context)
            
            # 7. 保存生成的代码到会话
            session_data.generated_code = generated_code
            session_data.system_info = {
                "system_description": system_description,
                "functional_unit": functional_unit,
                "impact_categories": impact_categories
            }
            
            logger.info("LCA系统构建完成")
            
            return {
                "success": True,
                "message": "LCA系统构建和代码生成成功",
                "system_info": {
                    "system_description": system_description,
                    "functional_unit": functional_unit,
                    "impact_categories": impact_categories
                },
                "generated_code": generated_code,
                "code_length": len(generated_code),
                "context_summary": {
                    "pdf_context_count": len(pdf_context),
                    "lci_data_count": len(lci_data)
                }
            }
            
        except Exception as e:
            logger.error(f"构建LCA系统失败: {str(e)}")
            return {
                "success": False,
                "error": f"构建LCA系统时出错: {str(e)}"
            }
    
    async def search_document(self,
                            session_id: str,
                            query: str = None,
                            queries: List[str] = None,
                            max_results: int = 5,
                            max_results_per_query: int = 3,
                            max_total_results: int = 10,
                            extract_mode: str = "chunks",
                            min_similarity: float = 0.3,
                            deduplicate: bool = True) -> Dict[str, Any]:
        """
        工具3: 增强版文档搜索（支持单查询和批量查询）
        
        Args:
            session_id: 会话ID
            query: 单个搜索查询（与queries二选一）
            queries: 多个搜索查询列表（与query二选一）
            max_results: 单查询模式的最大返回结果数
            max_results_per_query: 批量模式每个查询的结果数
            max_total_results: 批量模式总结果数上限
            extract_mode: 提取模式 ('chunks'|'sentences'|'key_points')
            min_similarity: 最小相似度阈值
            deduplicate: 批量模式是否去重相同chunk
            
        Returns:
            Dict: 搜索结果，包含智能提取的内容
        """
        try:
            logger.info(f"搜索文档内容，会话ID: {session_id}")
            
            # 获取会话数据
            session_data = self.session_manager.get_session(session_id)
            if not session_data:
                return {
                    "success": False,
                    "error": "会话不存在，请先上传文档"
                }
            
            if not session_data.knowledge_base:
                return {
                    "success": False,
                    "error": "文档尚未处理，请先上传并处理文档"
                }
            
            # 🔥 防御性检查：如果 query 是 list，转换为批量模式
            if isinstance(query, list):
                queries = query
                query = None
            
            # 判断模式：批量 or 单查询
            if queries:
                # 批量模式：改进版三阶段处理
                # 阶段 1: 收集所有结果（不去重，不截断）
                # 阶段 2: 去重（保留最高相似度）
                # 阶段 3: 计算 boost，排序，取 Top N
                logger.info(f"批量查询模式: {queries}")
                
                # 阶段 1: 收集所有候选结果
                all_candidates = {}  # chunk_id -> best_result
                
                for q in queries:
                    search_results = session_data.knowledge_base.search(q, top_k=max_results_per_query * 2)
                    
                    # 过滤低相似度结果（不截断数量）
                    filtered = [
                        result for result in search_results 
                        if result.get("similarity_score", 0) >= min_similarity
                    ]
                    
                    for result in filtered:
                        metadata = result.get("metadata", {})
                        chunk_id = metadata.get("chunk_id")
                        
                        if not chunk_id:
                            continue
                        
                        # 阶段 2: 去重时保留最高相似度
                        if deduplicate:
                            if chunk_id not in all_candidates:
                                all_candidates[chunk_id] = result
                            elif result.get("similarity_score", 0) > all_candidates[chunk_id].get("similarity_score", 0):
                                # 更新为更高相似度的结果
                                all_candidates[chunk_id] = result
                        else:
                            # 不去重模式：直接添加（用唯一键）
                            unique_key = f"{chunk_id}_{len(all_candidates)}"
                            all_candidates[unique_key] = result
                
                # 转换为列表（阶段 3 的排序在 _process_search_results 后进行）
                search_results = list(all_candidates.values())
                query_for_processing = " | ".join(queries)  # 用于 boost 计算
            else:
                # 单查询模式（原有逻辑）
                logger.info(f"单查询模式: {query}")
                search_results = session_data.knowledge_base.search(query, top_k=max_results * 2)
                
                # 过滤低相似度结果
                search_results = [
                result for result in search_results 
                if result.get("similarity_score", 0) >= min_similarity
            ][:max_results]
                
                query_for_processing = query
            
            # 根据提取模式处理结果
            processed_results = self._process_search_results(
                search_results, query_for_processing, extract_mode
            )
            
            # 阶段 3: 按提升后的分数重新排序
            processed_results.sort(key=lambda x: x.get("_boosted_score", 0), reverse=True)
            
            # 阶段 3: 排序后截断（批量模式使用 max_total_results，单查询使用 max_results）
            if queries:
                processed_results = processed_results[:max_total_results]
            else:
                processed_results = processed_results[:max_results]
            
            # 格式化最终结果（移除内部字段）
            formatted_results = []
            for i, result in enumerate(processed_results, 1):
                metadata = result.get("metadata", {})
                chunk_id = metadata.get("chunk_id", f"temp_chunk_{i}")
                display_content = result.get("extracted_content", result.get("content", ""))
                
                # 只返回必要字段，不包含_boosted_score
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "content": display_content
                })
            
            # 返回结果（格式根据模式不同）
            if queries:
                return {
                    "success": True,
                    "message": f"找到 {len(formatted_results)} 个相关结果",
                    "queries": queries,  # 返回数组
                    "results": formatted_results
                }
            else:
                return {
                    "success": True,
                    "message": f"找到 {len(formatted_results)} 个相关结果",
                    "query": query,  # 返回字符串
                    "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"搜索文档失败: {str(e)}")
            return {
                "success": False,
                "error": f"搜索文档失败: {str(e)}"
            }
    
    async def record_document_preview(self, session_id: str) -> Dict[str, Any]:
        """
        记录文档预览（chunk 0 和 chunk 1）到 MongoDB
        
        用于训练数据导出时的上下文感知检索 (Context-Aware Retrieval)
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 记录结果
        """
        try:
            from .mongodb_manager import mongodb_manager
            
            logger.info(f"记录文档预览，会话ID: {session_id}")
            
            # 获取会话数据
            session_data = self.session_manager.get_session(session_id)
            if not session_data:
                return {
                    "success": False,
                    "error": "会话不存在，请先上传文档"
                }
            
            if not session_data.documents or len(session_data.documents) == 0:
                return {
                    "success": False,
                    "error": "文档尚未处理，请先上传并处理文档"
                }
            
            # 提取 chunk 0 和 chunk 1
            chunk_0 = None
            chunk_1 = None
            
            if len(session_data.documents) > 0:
                chunk_0 = {
                    "chunk_id": "0",
                    "content": session_data.documents[0].page_content,
                    "metadata": session_data.documents[0].metadata if hasattr(session_data.documents[0], 'metadata') else {}
                }
            
            if len(session_data.documents) > 1:
                chunk_1 = {
                    "chunk_id": "1",
                    "content": session_data.documents[1].page_content,
                    "metadata": session_data.documents[1].metadata if hasattr(session_data.documents[1], 'metadata') else {}
                }
            
            # 存储到 MongoDB（lca_actions 集合）
            db = mongodb_manager.get_database()
            
            preview_record = {
                "session_id": session_id,
                "action_id": "ACT_PREVIEW",  # 特殊 action_id
                "tool_name": "record_document_preview",
                "record_type": "document_preview",  # 特殊类型
                "timestamp": datetime.now(),
                "document_preview": {
                    "chunk_0": chunk_0,
                    "chunk_1": chunk_1
                },
                "metadata": {
                    "exclude_from_export": True,  # 🔥 标记为不导出到训练数据
                    "document_name": session_data.original_filename
                }
            }
            
            # 检查是否已存在（避免重复记录）
            existing = db.lca_actions.find_one({
                "session_id": session_id,
                "record_type": "document_preview"
            })
            
            if existing:
                # 更新现有记录
                db.lca_actions.update_one(
                    {"_id": existing["_id"]},
                    {"$set": preview_record}
                )
                logger.info(f"更新文档预览记录: {session_id}")
            else:
                # 插入新记录
                db.lca_actions.insert_one(preview_record)
                logger.info(f"创建文档预览记录: {session_id}")
            
            return {
                "success": True,
                "message": "文档预览已记录",
                "data": {
                    "session_id": session_id,
                    "chunk_0_length": len(chunk_0["content"]) if chunk_0 else 0,
                    "chunk_1_length": len(chunk_1["content"]) if chunk_1 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"记录文档预览失败: {str(e)}")
            return {
                "success": False,
                "error": f"记录文档预览失败: {str(e)}"
            }
    
    def _calculate_coverage_boost(self, content: str, query: str) -> float:
        """
        计算关键词覆盖率提升分数（语义增强版）
        
        对每个查询关键词：
        1. 先尝试精确匹配（快速路径）
        2. 若不匹配，使用语义相似度判断是否"覆盖到了"
        
        Returns: 0-0.25之间的提升分数
        """
        if not query:
            return 0.0
        
        # 提取查询关键词（支持批量查询，用 | 分隔）
        queries = query.split('|') if '|' in query else [query]
        all_keywords = []
        for q in queries:
            keywords = q.strip().lower().split()
            all_keywords.extend(keywords)
        
        # 去重但保持列表形式（用于 embedding）
        all_keywords = list(set(all_keywords))
        
        if not all_keywords:
            return 0.0
        
        content_lower = content.lower()
        matched_count = 0
        
        # 如果语义模型可用，使用语义匹配
        if self.semantic_model is not None:
            try:
                # 计算 chunk 的 embedding（一次性）
                chunk_embedding = self.semantic_model.encode(content_lower, normalize_embeddings=True)
                
                for kw in all_keywords:
                    # 快速路径：精确匹配
                    if kw in content_lower:
                        matched_count += 1
                    else:
                        # 语义匹配：计算关键词与 chunk 的相似度
                        kw_embedding = self.semantic_model.encode(kw, normalize_embeddings=True)
                        similarity = np.dot(kw_embedding, chunk_embedding)
                        
                        # 阈值 0.35：平衡精确性和召回率
                        # 能匹配: "titanium"/"Ti-6Al-4V", "energy"/"electricity", "waste"/"scrap"
                        # 注意: 部分领域缩写如 "aluminum"/"AlSi10Mg" 可能无法匹配
                        if similarity > 0.35:
                            matched_count += 1
                            
            except Exception as e:
                logger.warning(f"语义匹配计算失败，回退到精确匹配: {e}")
                matched_count = sum(1 for kw in all_keywords if kw in content_lower)
        else:
            # 回退：精确匹配
            matched_count = sum(1 for kw in all_keywords if kw in content_lower)
        
        # 覆盖率：匹配的关键词数 / 总关键词数
        coverage = matched_count / len(all_keywords)
        
        # 权重 0.15（语义匹配后覆盖率普遍较高，降低影响）
        return coverage * 0.15
    
    def _calculate_data_density_boost(self, content: str) -> float:
        """
        计算数据密度提升分数（简化版）
        
        阈值：5个数字为满分，仅过滤年份
        Returns: 0-0.12之间的提升分数
        """
        import re
        
        # 提取所有数字
        numbers = re.findall(r'\d+\.?\d*', content)
        
        # 仅过滤年份（1900-2100），不过滤页码
        # 理由：chunking 时页码已被清除，且 LCI 数据常在 1-999 范围内
        relevant_numbers = []
        for num in numbers:
            try:
                val = float(num)
                # 只过滤年份
                if not (1900 <= val <= 2100):
                    relevant_numbers.append(num)
            except ValueError:
                continue
        
        # 阈值：5个数字为满分
        density = min(len(relevant_numbers) / 5.0, 1.0)
        
        # 权重 0.12
        return density * 0.12
    
    def _calculate_table_boost(self, content: str) -> float:
        """
        计算表格标记提升分数
        
        表格是 LCI 数据的核心载体，给予较高权重
        - pipe_count >= 3: 有表格，+0.18
        Returns: 0 或 0.18
        """
        pipe_count = content.count('|')
        
        if pipe_count >= 3:
            return 0.18  # 有表格
        else:
            return 0.0   # 无表格
    
    def _calculate_enhanced_boost(self, content: str, query: str) -> float:
        """
        计算增强提升分数
        
        组合：覆盖率(0.15) + 数据密度(0.12) + 表格标记(0.18)
        Returns: 0-0.45之间的提升分数
        """
        coverage_boost = self._calculate_coverage_boost(content, query)
        density_boost = self._calculate_data_density_boost(content)
        table_boost = self._calculate_table_boost(content)
        
        return coverage_boost + density_boost + table_boost
    
    def _process_search_results(self, results, query, extract_mode):
        """处理搜索结果，根据模式进行智能提取"""
        processed = []
        
        for result in results:
            content = result.get("content", "")
            similarity = result.get("similarity_score", 0)
            
            # 计算增强提升分数（不保存，仅用于排序）
            enhanced_boost = self._calculate_enhanced_boost(content, query)
            boosted_score = similarity + enhanced_boost
            
            # 临时存储用于排序（不会返回给前端）
            result["_boosted_score"] = boosted_score
            
            if extract_mode == "sentences":
                # 提取最相关的句子
                extracted = self._extract_relevant_sentences(content, query)
                processed.append({
                    **result,
                    "extracted_content": extracted["content"],
                    "confidence": extracted["confidence"],
                    "extract_type": "sentences"
                })
            
            elif extract_mode == "key_points":
                # 提取关键要点
                extracted = self._extract_key_points(content, query)
                processed.append({
                    **result,
                    "extracted_content": extracted["content"],
                    "confidence": extracted["confidence"],
                    "extract_type": "key_points"
                })
            
            else:  # chunks (默认)
                processed.append({
                    **result,
                    "extracted_content": content,
                    "confidence": similarity,
                    "extract_type": "chunk"
                })
        
        return processed
    
    def _extract_relevant_sentences(self, content, query):
        """从内容中提取最相关的句子"""
        import re
        
        # 分割句子（支持中英文）
        sentences = re.split(r'[.!?。！？]\s*', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return {"content": content[:200] + "...", "confidence": 0.3}
        
        # 简单的关键词匹配评分
        query_words = set(query.lower().split())
        best_sentences = []
        
        for sentence in sentences:
            score = 0
            sentence_words = set(sentence.lower().split())
            
            # 计算词汇重叠度
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                score = overlap / len(query_words)
                best_sentences.append((sentence, score))
        
        if best_sentences:
            # 选择得分最高的句子
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            selected = best_sentences[:2]  # 最多选择2个句子
            content = " ".join([s[0] for s in selected])
            confidence = max([s[1] for s in selected])
        else:
            # 如果没有匹配，返回前几个句子
            content = ". ".join(sentences[:2])
            confidence = 0.2
        
        return {"content": content, "confidence": min(confidence, 0.9)}
    
    def _extract_key_points(self, content, query):
        """从内容中提取关键要点"""
        import re
        
        # 寻找可能的要点标记
        point_patterns = [
            r'^\s*[-•]\s*(.+)',  # 列表项
            r'^\s*\d+[.)]\s*(.+)',  # 编号项
            r'([A-Z][^.!?]*(?:method|approach|process|result|conclusion)[^.!?]*[.!?])',  # 包含关键词的句子
        ]
        
        points = []
        for pattern in point_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            points.extend(matches)
        
        if not points:
            # 如果没有找到明显的要点，提取包含查询词的句子
            sentences = re.split(r'[.!?。！？]\s*', content)
            query_words = set(query.lower().split())
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if query_words.intersection(sentence_words):
                    points.append(sentence.strip())
        
        if points:
            # 选择最相关的要点
            selected_points = points[:3]  # 最多3个要点
            content = "• " + "\n• ".join(selected_points)
            confidence = 0.7
        else:
            # 降级到句子提取
            return self._extract_relevant_sentences(content, query)
        
        return {"content": content, "confidence": confidence}

    async def search_lci_database(self, query: str, max_results: int = 5, similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        搜索LCI数据库
        直接搜索生命周期清单数据库中的标准数据
        
        Args:
            query: 搜索查询
            max_results: 最大返回结果数
            similarity_threshold: 相似度阈值
        """
        try:
            logger.info(f"搜索LCI数据库: {query}")
            
            # 直接搜索LCI数据库
            lci_data = self.permanent_lci_db.search_flows(
                query=query,
                context=[],  # 无PDF上下文
                k=max_results,
                similarity_threshold=similarity_threshold
            )
            
            return {
                "success": True,
                "message": f"Found {len(lci_data)} LCI database entries for '{query}'",
                "query": query,
                "results": lci_data,
                "total_results": len(lci_data),
                "search_parameters": {
                    "max_results": max_results,
                    "similarity_threshold": similarity_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"LCI数据库搜索失败: {str(e)}")
            return {
                "success": False,
                "error": f"LCI database search failed: {str(e)}",
                "query": query
            }
    
    async def execute_analysis(self, 
                             session_id: str,
                             analysis_type: str = "impact_assessment") -> Dict[str, Any]:
        """
        工具3: 执行分析
        执行LCA计算并分析环境影响结果
        
        Args:
            session_id: 会话ID
            analysis_type: 分析类型，如'impact_assessment', 'contribution_analysis'
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            logger.info(f"开始执行LCA分析: {analysis_type}")
            
            # 1. 检查会话和代码是否存在
            session_data = self.session_manager.get_session(session_id)
            if not session_data:
                return {
                    "success": False,
                    "error": "会话不存在或已过期"
                }
            
            if not hasattr(session_data, 'generated_code') or not session_data.generated_code:
                return {
                    "success": False,
                    "error": "未找到生成的代码，请先构建LCA系统"
                }
            
            # 2. 执行pyLCA代码
            execution_result = self.pylca_executor.execute_code(session_data.generated_code)
            
            # 3. 格式化分析结果
            formatted_result = self._format_analysis_result(
                execution_result, 
                analysis_type,
                session_data.system_info if hasattr(session_data, 'system_info') else {}
            )
            
            # 4. 保存结果到会话
            session_data.analysis_results = {
                "execution_result": execution_result,
                "formatted_result": formatted_result,
                "analysis_type": analysis_type
            }
            
            logger.info("LCA分析执行完成")
            
            return {
                "success": True,
                "message": "LCA计算和分析完成",
                "analysis_type": analysis_type,
                "execution_result": execution_result,
                "formatted_result": formatted_result,
                "system_info": getattr(session_data, 'system_info', {})
            }
            
        except Exception as e:
            logger.error(f"执行LCA分析失败: {str(e)}")
            return {
                "success": False,
                "error": f"执行LCA分析时出错: {str(e)}"
            }
    
    def _format_analysis_result(self, execution_result: Dict[str, Any], 
                              analysis_type: str, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """格式化分析结果"""
        try:
            if not execution_result.get("success"):
                return {
                    "analysis_type": analysis_type,
                    "status": "failed",
                    "error": execution_result.get("error", "执行失败")
                }
            
            # 基础格式化结果
            formatted = {
                "analysis_type": analysis_type,
                "status": "success",
                "system_description": system_info.get("system_description", "未知系统"),
                "functional_unit": system_info.get("functional_unit", "未知单位"),
                "execution_time": execution_result.get("execution_time", 0),
                "output": execution_result.get("output", "")
            }
            
            # 根据分析类型添加特定格式化
            if analysis_type == "impact_assessment":
                formatted["impact_results"] = self._extract_impact_results(execution_result)
            elif analysis_type == "contribution_analysis":
                formatted["contribution_data"] = self._extract_contribution_data(execution_result)
            elif analysis_type == "sensitivity_analysis":
                formatted["sensitivity_data"] = self._extract_sensitivity_data(execution_result)
            
            return formatted
            
        except Exception as e:
            logger.error(f"格式化分析结果失败: {str(e)}")
            return {
                "analysis_type": analysis_type,
                "status": "error",
                "error": f"结果格式化失败: {str(e)}"
            }
    
    def _extract_impact_results(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取影响评估结果"""
        # 简单的结果提取逻辑，可根据实际pyLCA输出格式调整
        output = execution_result.get("output", "")
        
        # 这里可以添加更复杂的结果解析逻辑
        return {
            "raw_output": output,
            "summary": "影响评估计算完成，详细结果请查看输出"
        }
    
    def _extract_contribution_data(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取贡献分析数据"""
        return {
            "raw_output": execution_result.get("output", ""),
            "summary": "贡献分析计算完成"
        }
    
    def _extract_sensitivity_data(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """提取敏感性分析数据"""
        return {
            "raw_output": execution_result.get("output", ""),
            "summary": "敏感性分析计算完成"
        }
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get tools schema definition (OpenAI Function Calling format, for LLM use)
        
        Tool selection principles:
        - All tools are equal, choose appropriate tools based on user needs
        - Each tool has clear usage scenarios and purposes
        - LLM needs to understand each tool's functionality and make correct choices
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "process_document",
                    "description": "Process newly uploaded PDF documents. Used only for document upload, not in chat conversations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_content": {
                                "type": "string",
                                "description": "Base64 encoded content of the PDF file"
                            },
                            "filename": {
                                "type": "string",
                                "description": "Filename of the uploaded document"
                            },
                            "search_focus": {
                                "type": "string",
                                "description": "Search focus, such as 'material consumption', 'energy usage', 'emission factors', etc."
                            }
                        },
                        "required": ["file_content", "filename"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "search_document",
                    "description": "Search uploaded PDF document content. Supports single query or batch search for multiple related keywords.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Document processing session ID"
                            },
                            "query": {
                                "type": "string",
                                "description": "Single search query (use this OR queries, not both)"
                            },
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Batch search: multiple related keywords (use this OR query, not both). Example: ['electricity', 'energy', 'power']"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum results for single query mode",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "max_results_per_query": {
                                "type": "integer",
                                "description": "Maximum results per query in batch mode",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "max_total_results": {
                                "type": "integer",
                                "description": "Maximum total results in batch mode",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 20
                            },
                            "deduplicate": {
                                "type": "boolean",
                                "description": "Remove duplicate results based on chunk_id in batch mode",
                                "default": True
                            },
                            "extract_mode": {
                                "type": "string",
                                "description": "Content extraction mode",
                                "enum": ["chunks", "sentences", "key_points"],
                                "default": "chunks"
                            },
                            "min_similarity": {
                                "type": "number",
                                "description": "Minimum similarity threshold (0-1)",
                                "default": 0.3,
                                "minimum": 0.0,
                                "maximum": 1.0
                            }
                        },
                        "required": ["session_id"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "record_document_preview",
                    "description": "Record document preview (chunk 0 and chunk 1) to MongoDB for training data export. This tool is used ONLY for data annotation, not for normal chat. Click once after uploading a document to save the preview for Context-Aware Retrieval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Document processing session ID"
                            }
                        },
                        "required": ["session_id"]
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "build_lca_system",
                    "description": "Build complete LCA system based on document information and LCI data, then generate pyLCA code",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Document processing session ID"
                            },
                            "system_description": {
                                "type": "string",
                                "description": "LCA system description, such as 'solar panel production system', 'steel production system'"
                            },
                            "functional_unit": {
                                "type": "string",
                                "description": "Functional unit, such as '1 kWp', '1 kg product', '1 ton steel'"
                            },
                            "impact_categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Impact categories of interest, such as ['GWP', 'AP', 'EP', 'CED']"
                            }
                        },
                        "required": ["session_id", "system_description", "functional_unit"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "search_lci_database",
                    "description": "Search LCI (Life Cycle Inventory) database. Used to obtain standard environmental data, emission factors, material properties, etc. Suitable for users who need standard LCA data, comparative analysis, or to supplement missing environmental data from documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query, such as 'steel production', 'electricity consumption', 'plastic manufacturing', 'transportation emissions', etc."
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 10
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Similarity threshold (0-1)",
                                "default": 0.3,
                                "minimum": 0.1,
                                "maximum": 0.9
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "define_lca_scope",
                    "description": "Define core LCA scope parameters extractable from production documents: Functional Unit, System Boundary, Geographical Scope",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Document processing session ID"
                            },
                            "parameter_name": {
                                "type": "string",
                                "enum": [
                                    "Functional Unit", 
                                    "System Boundary", 
                                    "Geographical Scope"
                                ],
                                "description": "Core LCA scope parameter extractable from production process documents - Golden Three Elements"
                            },
                            "description": {
                                "type": "string",
                                "description": "Complete, original description text extracted from document"
                            },
                            "value": {
                                "type": "number",
                                "description": "Quantitative value (mainly for functional unit)"
                            },
                            "unit": {
                                "type": "string",
                                "description": "Unit (mainly for functional unit)"
                            },
                            "source_content": {
                                "type": "string",
                                "description": "More complete original text context containing the description"
                            },
                            "note": {
                                "type": "string",
                                "description": "Additional context or notes about this scope definition"
                            },
                            "selected_chunk": {
                                "type": "string",
                                "description": "The exact text snippet from the document that contains this scope information - IMPORTANT for traceability"
                            }
                        },
                        "required": ["session_id", "parameter_name", "description"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "record_process_flow",
                    "description": "Record process flow data - the main tool for LCI data extraction with comprehensive classification system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Document processing session ID"
                            },
                            "flow_type": {
                                "type": "string",
                                "enum": ["Input", "Output"],
                                "description": "Type of flow - simplified to Input/Output, detailed classification by category"
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "Raw Material", "Process Energy", "Post-processing Energy", "Feedstock Energy", "Gas", "Cooling Media",
                                    "Product", "Recovered Material", "Waste", "Emission"
                                ],
                                "description": "工作台的11个LCI类别: Input flows (Raw Material, Process Energy, Post-processing Energy, Feedstock Energy, Gas, Cooling Media) or Output flows (Product, Recovered Material, Waste, Emission)"
                            },
                            "name": {
                                "type": "string",
                                "description": "Specific name of material or energy"
                            },
                            "value": {
                                "type": "number",
                                "description": "Quantity value"
                            },
                            "unit": {
                                "type": "string",
                                "description": "Unit"
                            },
                            "location": {
                                "type": "string",
                                "description": "Geographic location where process occurs"
                            },
                            "cas_number": {
                                "type": "string",
                                "description": "CAS number for chemical substances - key for precise identification and future database linking"
                            },
                            "process_name": {
                                "type": "string",
                                "description": "Name of the process step this flow belongs to - provides structural context for LCI data"
                            },
                            "note": {
                                "type": "string",
                                "description": "Additional context or notes about the flow (e.g., 'SLM machine', 'Atomization process', 'from Table 2')"
                            },
                            "selected_chunk": {
                                "type": "string",
                                "description": "The exact text snippet from the document that contains this data - IMPORTANT for traceability and matching"
                            }
                        },
                        "required": ["session_id", "flow_type", "category", "name", "value", "unit"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "record_parameter",
                    "description": "Record raw parameters extracted from documents - the first step of the three-tool architecture (Parameter → Calculation → Process Flow). Use this to extract quantitative parameters (like power, time, material amount) that will be used in calculations later.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Document processing session ID"
                            },
                            "parameter_name": {
                                "type": "string",
                                "description": "Descriptive parameter name, such as 'motor_power', 'printing_time', 'material_input', 'extruder_temperature'"
                            },
                            "parameter_value": {
                                "type": "number",
                                "description": "Numerical parameter value"
                            },
                            "parameter_unit": {
                                "type": "string",
                                "description": "Parameter unit, such as 'kW', 'h', 'kg', '°C', 'MPa'"
                            },
                            "selected_chunk": {
                                "type": "string",
                                "description": "The exact text snippet from the document containing this parameter - REQUIRED for traceability"
                            },
                            "note": {
                                "type": "string",
                                "description": "Additional context or notes about this parameter (e.g., 'from Table 2', 'calculated from figure')"
                            }
                        },
                        "required": ["session_id", "parameter_name", "parameter_value", "selected_chunk"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "record_calculation",
                    "description": "Record mathematical calculations - the second step of the three-tool architecture. Use this to record calculations performed on extracted parameters. IMPORTANT: Always call get_session_summary first to retrieve parameter action_ids for data_dependencies field.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Session ID"
                            },
                            "calculation_expression": {
                                "type": "string",
                                "description": "Calculation expression, such as 'power * time' or '(2.5 + 1.5) * 8' or 'motor_power * operation_time'"
                            },
                            "calculation_result": {
                                "type": "number",
                                "description": "Calculated result value"
                            },
                            "calculation_unit": {
                                "type": "string",
                                "description": "Result unit, such as 'kWh', 'MJ', 'kg', 'kg CO2-eq'"
                            },
                            "data_dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of parameter action_ids that this calculation depends on. Example: ['ACT_0001', 'ACT_0002']. Get these from get_session_summary before calling this tool."
                            },
                            "expert_decision": {
                                "type": "object",
                                "properties": {
                                    "rationale": {
                                        "type": "string",
                                        "description": "Natural language explanation of why this calculation is performed and what it represents"
                                    }
                                },
                                "required": ["rationale"]
                            }
                        },
                        "required": ["session_id", "calculation_expression", "calculation_result", "data_dependencies", "expert_decision"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "execute_calculation",
                    "description": "Execute mathematical calculation without recording - helper tool for verifying calculation results before recording them with record_calculation. Supports basic arithmetic operations and functions like sqrt, log, exp.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate, such as '10.5 * 5', 'sqrt(16)', '(2.5 + 1.5) * 8'"
                            },
                            "variables": {
                                "type": "object",
                                "description": "Variable values if expression contains variable names, such as {'power': 10.5, 'time': 5}. Optional if expression contains only numbers."
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            
            {
                "type": "function",
                "function": {
                    "name": "get_session_summary",
                    "description": "Get session summary - provides LLM with 'working memory' and 'self-reflection' capability. Returns all recorded scopes, parameters (with action_ids), calculations, and flows. MUST call this before record_calculation to get parameter action_ids.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "session_id": {
                                "type": "string",
                                "description": "Document processing session ID to query"
                            }
                        },
                        "required": ["session_id"]
                    }
                }
            }
        ]

    # ==================== 第一阶段新增工具 ====================
    
    async def define_lca_scope(self, 
                             session_id: str,
                             parameter_name: str,
                             description: str,
                             value: float = None,
                             unit: str = None,
                             source_content: str = None,
                             note: str = None,  # 🔥 NEW: 备注信息
                             search_query: str = None,
                             search_context: List[Dict[str, Any]] = None,
                             selected_chunk: Dict[str, Any] = None,
                             intent: str = "select_best",
                             link_to: str = None) -> Dict[str, Any]:
        """
        工具: 定义LCA范围 - 定义分析边界
        从文档中提取LCA核心范围信息（功能单位、系统边界、地理范围）
        
        Args:
            session_id: 会话ID
            parameter_name: 范围参数名称（必需）- 'Function Unit', 'System Boundary', 或 'Geographical Scope'
            description: 参数的完整描述（必需）
            value: 参数的量化数值（可选，主要用于功能单位）
            unit: 参数的单位（可选，主要用于功能单位）
            source_content: 更完整的原文上下文片段（可选）
            note: 备注信息（可选，用于区分细节）
            search_query: 搜索查询（可选，用于数据收集）
            search_context: 搜索结果列表（可选，用于数据收集）
            selected_chunk: 选择的文档片段（可选）
            intent: 动作意图（可选，默认 "select_best"）
            link_to: 指向上一步动作的action_id（可选）
            
        Returns:
            Dict[str, Any]: 定义结果 (包含action_id)
        """
        try:
            logger.info(f"定义LCA范围: {parameter_name} for session {session_id}")
            
            # 验证参数 - 黄金三要素：从生产工艺文档中可提取的核心LCA范围信息
            valid_parameter_names = [
                'Function Unit',        # 功能单位：生产什么产品，多少数量
                'System Boundary',      # 系统边界：包含哪些工艺步骤
                'Geographical Scope',   # 地理范围：生产地点、区域信息
                'Functional Unit'       # 向后兼容旧名称
            ]
            if parameter_name not in valid_parameter_names:
                return {
                    "success": False,
                    "error": f"Invalid parameter_name. Must be one of: {valid_parameter_names}"
                }
            
            # 获取MongoDB连接并生成递增的action_id（按照Expert_Workbench Decision Logic Schema规范）
            # 🔥 NEW: 统一使用lca_actions集合，通过record_type区分scope和flow
            from .mongodb_manager import mongodb_manager
            db = mongodb_manager.get_database()
            collection = db.lca_actions
            
            # 查找当前session的最大action_id（排除特殊action_id如ACT_PREVIEW）
            existing_actions = list(collection.find(
                {"session_id": session_id, "action_id": {"$regex": "^ACT_\\d{4}$"}},  # 🔥 只匹配 ACT_0001 格式
                {"action_id": 1}
            ).sort("action_id", -1).limit(1))
            
            if existing_actions:
                # 提取最后一个action_id的数字部分并递增
                last_action_id = existing_actions[0]["action_id"]
                try:
                    last_num = int(last_action_id.split("_")[1])
                    next_num = last_num + 1
                except (ValueError, IndexError):
                    next_num = 1
            else:
                # 第一个action_id
                next_num = 1
            
            action_id = f"ACT_{next_num:04d}"  # 格式：ACT_0001, ACT_0002, etc.
            
            # 构建数据记录 (按照Expert_Workbench Decision Logic Schema v1.3)
            # 🔥 NEW: 统一的数据结构，通过record_type区分scope和flow
            scope_record = {
                # 通用字段
                "session_id": session_id,
                "action_id": action_id,
                "link_to": link_to,
                "type": "record",  # 统一为record类型
                "record_type": "scope",  # 🔥 NEW: 区分scope和flow
                "intent": intent,
                "created_at": datetime.now().isoformat(),
                
                # LCA范围数据
                "parameter_name": parameter_name,
                "description": description,
                
                # 🔥 NEW: 搜索查询
                "search_query": search_query,
                
                # 决策数据
                "search_context": search_context or [],
                "selected_chunk": selected_chunk or {}
            }
            
            # 添加可选字段
            if value is not None:
                scope_record["value"] = value
            if unit is not None:
                scope_record["unit"] = unit
            if source_content is not None:
                scope_record["source_content"] = source_content
            if note is not None:
                scope_record["note"] = note
            
            # 使用插入逻辑：为每条记录生成唯一的record_id，不断累加数据
            insert_result = collection.insert_one(scope_record)
            record_id = str(insert_result.inserted_id)
            
            logger.info(f"LCA范围定义成功: {record_id}")
            
            return {
                "success": True,
                "message": f"Successfully defined LCA scope: {parameter_name}",
                "data": {
                    "record_id": record_id,
                    "action_id": action_id,
                    "session_id": session_id,
                    "parameter_name": parameter_name,
                    "description": description,
                    "value": value,
                    "unit": unit,
                    "intent": intent,
                    "link_to": link_to
                }
            }
            
        except Exception as e:
            logger.error(f"定义LCA范围失败: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to define LCA scope: {str(e)}"
            }
    
    async def record_process_flow(self,
                                session_id: str,
                                flow_type: str,
                                category: str,
                                name: str,
                                value: float,
                                unit: str,
                                location: str = None,
                                cas_number: str = None,
                                process_name: str = None,
                                note: str = None,
                                search_query: str = None,
                                search_context: List[Dict[str, Any]] = None,
                                selected_chunk: Dict[str, Any] = None,
                                intent: str = "select_best",
                                link_to: str = None) -> Dict[str, Any]:
        """
        工具: 记录工艺流 - 记录最终的LCI数据
        记录生命周期清单中的输入/输出流数据
        
        Args:
            session_id: 会话ID
            flow_type: 流的类型（必需）- 'Input' 或 'Output'
            category: 流的分类（必需）- 工作台的11个LCI类别:
                     输入流: 'Raw Material', 'Process Energy', 'Post-processing Energy', 'Feedstock Energy', 'Gas', 'Cooling Media'
                     输出流: 'Product', 'Recovered Material', 'Waste', 'Emission'
            name: 物质或能量的具体名称（必需）
            value: 数量值（必需）
            unit: 单位（必需）
            location: 工艺发生的地理位置（可选）
            cas_number: 化学品CAS号（可选）
            process_name: 工艺步骤名称（可选）
            note: 备注信息（可选，用于区分细节，如"SLM machine", "Atomization"等）
            search_query: 搜索查询（可选，用于数据收集）
            search_context: 搜索结果列表（可选，用于数据收集）
            selected_chunk: 选择的文档片段（可选，作为证据）
            intent: 动作意图（可选，默认 "select_best"）
            link_to: 指向上一步动作的action_id（可选，用于链接到计算）
            
        Returns:
            Dict[str, Any]: 记录结果 (包含action_id)
        """
        try:
            logger.info(f"记录工艺流: {flow_type}/{category}/{name} for session {session_id}")
            
            # 🔥 兼容 selected_chunk 为字符串或字典
            # LLM 可能传递字符串（文档片段文本）或字典（包含 chunk_id）
            if isinstance(selected_chunk, str):
                # 将字符串转换为字典格式
                selected_chunk = {"text": selected_chunk, "chunk_id": None}
            
            # 🔥 验证数据来源（软性警告，不阻止）
            has_document_source = selected_chunk and isinstance(selected_chunk, dict) and selected_chunk.get("chunk_id")
            has_calculation_source = link_to and link_to.startswith("ACT_")
            
            if not has_document_source and not has_calculation_source:
                logger.warning(f"⚠️ record_process_flow called without provenance (no selected_chunk or link_to) - allowing for flexibility")
                # 🔥 不返回错误，允许 LLM 灵活使用工具
                # 相信 LLM 的原生智能判断，避免过度限制
            
            # 验证参数 - 使用工作台的11个LCI分类体系
            valid_flow_types = ['Input', 'Output']
            
            # 工作台的11个LCI类别（与keyword_suggester.py和工作台UI一致）
            valid_input_categories = [
                'Raw Material',              # 原材料
                'Process Energy',            # 机器能耗（printing, laser等）
                'Post-processing Energy',    # 后处理能耗（heat treatment, machining等）
                'Feedstock Energy',          # 粉末制备能耗（atomization等）
                'Gas',                       # 气体（argon, nitrogen等）
                'Cooling Media'              # 冷却/加工液体（water, coolant, cutting fluid等）
            ]
            
            valid_output_categories = [
                'Product',                   # 主要产品
                'Recovered Material',        # 回收材料（recovered powder等）
                'Waste',                     # 废料（support structures, scrap等）
                'Emission'                   # 排放（VOC, particulate, wastewater等）
            ]
            
            valid_categories = valid_input_categories + valid_output_categories
            
            if flow_type not in valid_flow_types:
                return {
                    "success": False,
                    "error": f"Invalid flow_type. Must be one of: {valid_flow_types}"
                }
            
            if category not in valid_categories:
                return {
                    "success": False,
                    "error": f"Invalid category. Must be one of: {valid_categories}"
                }
            
            # 验证flow_type和category的逻辑一致性
            if flow_type == 'Input' and category not in valid_input_categories:
                return {
                    "success": False,
                    "error": f"Invalid category '{category}' for Input flow. Valid Input categories: {valid_input_categories}"
                }
            
            if flow_type == 'Output' and category not in valid_output_categories:
                return {
                    "success": False,
                    "error": f"Invalid category '{category}' for Output flow. Valid Output categories: {valid_output_categories}"
                }
            
            # 获取MongoDB连接并生成递增的action_id（按照Expert_Workbench Decision Logic Schema规范）
            from .mongodb_manager import mongodb_manager
            db = mongodb_manager.get_database()
            collection = db.lca_actions
            
            # 查找当前session的最大action_id（排除特殊action_id如ACT_PREVIEW）
            existing_actions = list(collection.find(
                {"session_id": session_id, "action_id": {"$regex": "^ACT_\\d{4}$"}},  # 🔥 只匹配 ACT_0001 格式
                {"action_id": 1}
            ).sort("action_id", -1).limit(1))
            
            if existing_actions:
                # 提取最后一个action_id的数字部分并递增
                last_action_id = existing_actions[0]["action_id"]
                try:
                    last_num = int(last_action_id.split("_")[1])
                    next_num = last_num + 1
                except (ValueError, IndexError):
                    next_num = 1
            else:
                # 第一个action_id
                next_num = 1
            
            action_id = f"ACT_{next_num:04d}"  # 格式：ACT_0001, ACT_0002, etc.
            
            # 构建数据记录 (按照Expert_Workbench Decision Logic Schema v1.3)
            flow_record = {
                # 通用字段
                "session_id": session_id,
                "action_id": action_id,
                "link_to": link_to,
                "type": "record",
                "record_type": "flow",  # 🔥 NEW: 区分scope和flow
                "intent": intent,
                "created_at": datetime.now().isoformat(),
                
                # 工艺流数据
                "flow_type": flow_type,
                "category": category,
                "name": name,
                "value": value,
                "unit": unit,
                
                # 🔥 NEW: 搜索查询
                "search_query": search_query,
                
                # 决策数据
                "search_context": search_context or [],
                "selected_chunk": selected_chunk or {}
            }
            
            # 添加可选字段
            if location is not None:
                flow_record["location"] = location
            if cas_number is not None:
                flow_record["cas_number"] = cas_number
            if process_name is not None:
                flow_record["process_name"] = process_name
            if note is not None:
                flow_record["note"] = note
            
            # 使用插入逻辑：为每条记录生成唯一的record_id，不断累加数据
            insert_result = collection.insert_one(flow_record)
            record_id = str(insert_result.inserted_id)
            
            logger.info(f"工艺流记录成功: {record_id}")
            
            return {
                "success": True,
                "message": f"Successfully recorded process flow: {name}",
                "data": {
                    "record_id": record_id,
                    "action_id": action_id,
                    "session_id": session_id,
                    "flow_type": flow_type,
                    "category": category,
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "location": location,
                    "flow_id": record_id,  # 保持向后兼容
                    "intent": intent,
                    "link_to": link_to
                }
            }
            
        except Exception as e:
            logger.error(f"记录工艺流失败: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to record process flow: {str(e)}"
            }
    
    def record_pivot_failure(self,
                           session_id: str,
                           failed_query: str,
                           link_to: str = None,
                           failed_context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        记录Pivot失败动作
        立即记录一个独立的失败动作，intent为"pivot_query"
        
        Args:
            session_id: 会话ID
            failed_query: 失败的搜索查询（必需，字符串或数组）
            link_to: 连接到上一个action_id（可选）
            failed_context: 失败查询返回的chunks（可选，用于数据收集）
            
        Returns:
            Dict[str, Any]: 记录结果 (包含new_action_id)
        """
        try:
            logger.info(f"记录Pivot失败动作: session {session_id}, failed_query: {failed_query}")
            
            # 获取MongoDB连接并生成递增的action_id
            from .mongodb_manager import mongodb_manager
            db = mongodb_manager.get_database()
            collection = db.lca_actions
            
            # 生成新的action_id (格式: ACT_0001, ACT_0002, ...)
            # 排除特殊action_id如ACT_PREVIEW
            last_action = collection.find_one(
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
            
            # 构建失败动作记录
            now = datetime.now().isoformat()
            failure_record = {
                "action_id": new_action_id,
                "session_id": session_id,
                "record_type": "pivot",
                "intent": "pivot_query",
                "link_to": link_to,
                "created_at": now,
                "timestamp": now,
                
                # 失败动作的核心字段
                "failed_query": failed_query,
                "failed_context": failed_context or []
            }
            
            # 插入到MongoDB
            result = collection.insert_one(failure_record)
            
            logger.info(f"Pivot失败动作已记录: {new_action_id} (ObjectId: {result.inserted_id})")
            
            return {
                "success": True,
                "message": f"Pivot failure recorded with action_id: {new_action_id}",
                "new_action_id": new_action_id,
                "new_intent": "pivot_query"
            }
            
        except Exception as e:
            logger.error(f"记录Pivot失败动作失败: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to record pivot failure: {str(e)}"
            }
    
    def record_smart_skip(self,
                         session_id: str,
                         category: str,
                         skip_reason: str = "already_recorded",
                         link_to: str = None,
                         skipped_chunk: Dict[str, Any] = None,
                         skip_rationale: str = None,
                         search_context: List[Dict[str, Any]] = None,
                         search_query: str = None) -> Dict[str, Any]:
        """
        🔥 NEW: 记录智能跳过动作
        当搜索成功但数据已在之前的优质chunk中提取过，智能跳过该类别
        
        Args:
            session_id: 会话ID
            category: 跳过的类别（必需）- 工作台的11个LCI类别:
                     输入流: 'Raw Material', 'Process Energy', 'Post-processing Energy', 'Feedstock Energy', 'Gas', 'Cooling Media'
                     输出流: 'Product', 'Recovered Material', 'Waste', 'Emission'
            skip_reason: 跳过原因（默认 "already_recorded"）
            link_to: 连接到上一个action_id（可选）
            skipped_chunk: 被跳过的chunk信息（可选，用于数据收集）
            skip_rationale: 专家填写的跳过说明（可选，用于数据收集）
            search_context: 完整的搜索结果（必需，用于训练数据生成）
            search_query: 搜索查询（可选，用于训练数据生成）
            
        Returns:
            Dict[str, Any]: 记录结果 (包含new_action_id)
        """
        try:
            logger.info(f"记录智能跳过动作: session {session_id}, category: {category}, reason: {skip_reason}")
            
            # 获取MongoDB连接并生成递增的action_id
            from .mongodb_manager import mongodb_manager
            db = mongodb_manager.get_database()
            collection = db.lca_actions
            
            # 生成新的action_id (格式: ACT_0001, ACT_0002, ...)
            # 排除特殊action_id如ACT_PREVIEW
            last_action = collection.find_one(
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
            
            # 🔥 构建智能跳过动作记录
            now = datetime.now().isoformat()
            skip_record = {
                "action_id": new_action_id,
                "session_id": session_id,
                "record_type": "smart_skip",  # 🔥 添加 record_type
                "intent": "smart_skip",  # 🔥 关键: 标记为智能跳过动作
                "link_to": link_to,
                "created_at": now,  # 🔥 添加 created_at 用于排序
                "timestamp": now,
                
                # 🔥 智能跳过的核心字段
                "category": category,
                "skip_reason": skip_reason,
                "skipped_chunk": skipped_chunk or {},
                "skip_rationale": skip_rationale,
                
                # 🔥 NEW: 保存完整的搜索上下文（用于训练数据生成）
                "search_context": search_context or [],
                "search_query": search_query
            }
            
            # 插入到MongoDB
            result = collection.insert_one(skip_record)
            
            logger.info(f"智能跳过动作已记录: {new_action_id} (ObjectId: {result.inserted_id})")
            
            return {
                "success": True,
                "message": f"Smart skip recorded for category: {category}",
                "new_action_id": new_action_id,
                "new_intent": "smart_skip",
                "category": category,
                "skip_reason": skip_reason
            }
            
        except Exception as e:
            logger.error(f"记录智能跳过动作失败: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to record smart skip: {str(e)}"
            }
    
    def record_parameter(self,
                        session_id: str,
                        parameter_name: str,
                        parameter_value: float,
                        selected_chunk: Union[str, Dict[str, Any]],
                        parameter_unit: str = None,
                        note: str = None,
                        search_query: str = None,
                        search_context: List[Dict[str, Any]] = None,
                        intent: str = "select_best",
                        link_to: str = None) -> Dict[str, Any]:
        """
        工具: 记录参数 - 用于记录待用于计算的原始参数
        
        这是三工具架构的第一步：从文档中提取原始参数值
        
        Args:
            session_id: 会话ID
            parameter_name: 参数名称，如 "power", "printing_time"（必需）
            parameter_value: 参数值（必需）
            selected_chunk: 选择的文档块（必需，作为证据）- 可以是字符串或字典
            parameter_unit: 参数单位（可选）
            note: 备注信息（可选，用于区分细节）
            search_query: 搜索查询（可选，用于数据收集）
            search_context: 搜索上下文（可选，用于数据收集）
            
        Returns:
            Dict[str, Any]: 记录结果
        """
        try:
            logger.info(f"记录参数: {parameter_name} = {parameter_value} {parameter_unit or ''}")
            
            # 兼容 selected_chunk 为字符串或字典
            if isinstance(selected_chunk, str):
                selected_chunk = {"text": selected_chunk, "chunk_id": None}
            
            # 获取MongoDB连接
            from .mongodb_manager import mongodb_manager
            db = mongodb_manager.get_database()
            collection = db.lca_actions
            
            # 生成新的action_id
            # 排除特殊action_id如ACT_PREVIEW
            last_action = collection.find_one(
                {"session_id": session_id, "action_id": {"$regex": "^ACT_\\d{4}$"}},
                sort=[("action_id", -1)]
            )
            
            if last_action and "action_id" in last_action:
                try:
                    last_num = int(last_action["action_id"].split("_")[1])
                    new_action_id = f"ACT_{last_num + 1:04d}"
                except (ValueError, IndexError):
                    new_action_id = "ACT_0001"
            else:
                new_action_id = "ACT_0001"
            
            # 构建参数记录（与Scope/Flow保持一致的结构）
            parameter_record = {
                "action_id": new_action_id,
                "session_id": session_id,
                "record_type": "parameter",
                "intent": intent,  # 使用传入的 intent 参数
                "link_to": link_to,  # 使用传入的 link_to 参数
                "timestamp": datetime.utcnow(),
                
                # 参数数据
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "parameter_unit": parameter_unit,
                
                # 搜索上下文（必须，因为是提取动作）
                "search_query": search_query,
                "search_context": search_context if search_context else [],
                "selected_chunk": selected_chunk
            }
            
            # 添加可选字段
            if note is not None:
                parameter_record["note"] = note
            
            # 插入到MongoDB
            result = collection.insert_one(parameter_record)
            
            if result.inserted_id:
                logger.info(f"参数记录成功: {new_action_id}")
                return {
                    "success": True,
                    "message": f"Parameter recorded with action_id: {new_action_id}",
                    "new_action_id": new_action_id
                }
            else:
                logger.error("参数记录失败: MongoDB插入失败")
                return {
                    "success": False,
                    "error": "Failed to insert into MongoDB"
                }
                
        except Exception as e:
            logger.error(f"记录参数时出错: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def record_calculation(self,
                          session_id: str,
                          calculation_expression: str,
                          calculation_result: float,
                          data_dependencies: List[str],
                          link_to: str = None,
                          calculation_unit: str = None,
                          intent: str = "calculate") -> Dict[str, Any]:
        """
        工具: 记录计算动作
        显式记录计算过程，避免LLM幻觉计算
        
        Args:
            session_id: 会话ID
            calculation_expression: 计算表达式，如 "10 * 5" 或 "Power * Time"（必需）
            calculation_result: 计算结果（必需）
            data_dependencies: 依赖的参数action_id列表（必需）
            link_to: 链接到的上一个动作ID（可选）
            calculation_unit: 计算结果单位（可选）
            intent: 动作意图（可选，默认 "calculate"）
            
        Returns:
            Dict[str, Any]: 记录结果
        """
        try:
            logger.info(f"记录计算动作: {calculation_expression} = {calculation_result}")
            
            # 获取MongoDB连接
            from .mongodb_manager import mongodb_manager
            db = mongodb_manager.get_database()
            collection = db.lca_actions
            
            # 生成新的action_id (格式: ACT_0001, ACT_0002, ...)
            # 排除特殊action_id如ACT_PREVIEW
            last_action = collection.find_one(
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
            
            # 构建计算记录
            calculation_record = {
                "action_id": new_action_id,
                "session_id": session_id,
                "record_type": "calculation",
                "intent": intent,
                "link_to": link_to,
                "timestamp": datetime.now().isoformat(),
                
                # 计算特定数据
                "calculation_expression": calculation_expression,
                "calculation_result": calculation_result,
                "calculation_unit": calculation_unit,
                "data_dependencies": data_dependencies if data_dependencies else []
            }
            
            # 插入MongoDB
            result = collection.insert_one(calculation_record)
            
            logger.info(f"计算动作已记录: {new_action_id} (ObjectId: {result.inserted_id})")
            
            return {
                "success": True,
                "message": f"Calculation recorded with action_id: {new_action_id}",
                "new_action_id": new_action_id,
                "new_intent": "calculate"
            }
            
        except Exception as e:
            logger.error(f"记录计算动作失败: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to record calculation: {str(e)}"
            }
    
    async def get_session_summary(self, session_id: str, format: str = "text", view: str = "llm") -> Union[Dict[str, Any], str]:
        """
        工具: 获取会话总结
        赋予LLM"工作记忆"和"状态自省"的能力
        
        🔥 重要：为保证训练-推理一致性，LLM视图只包含LLM会调用的工具结果
        
        Args:
            session_id: 需要查询的文档处理会话ID
            format: 返回格式
                - "text": 人类可读文本格式（LLM 使用，默认，节省 80% tokens）
                - "json": 完整 JSON 结构（专家工作台使用）
            view: 视图类型
                - "llm": LLM视图，只包含flow和parameter（训练数据使用）
                - "workbench": 工作台视图，包含所有统计（工作台UI使用）
            
        Returns:
            Union[Dict[str, Any], str]: 
                - format="json": 完整会话总结字典
                - format="text": 简洁的文本格式字符串
        """
        try:
            logger.info(f"获取会话总结: {session_id}")
            
            # 获取MongoDB连接
            from .mongodb_manager import mongodb_manager
            db = mongodb_manager.get_database()
            
            # 查询所有动作记录（统一集合）
            all_actions = list(db.lca_actions.find(
                {"session_id": session_id}
            ).sort([("created_at", 1)]))  # 按时间排序
            
            # 🔥 根据view类型过滤actions
            if view == "llm":
                # LLM视图：只保留LLM会调用的工具（scope, flow, parameter）
                # 排除：calculation（LLM不调用）, smart_skip（辅助标记）, pivot_query（辅助标记）
                visible_actions = [
                    a for a in all_actions 
                    if a.get("record_type") in ["scope", "flow", "parameter"]
                ]
                # 但保留smart_skip用于summary分析（LLM需要知道哪些类别被跳过）
                smart_skip_for_summary = [
                    a for a in all_actions
                    if a.get("intent") == "smart_skip"
                ]
            else:
                # 工作台视图：包含所有actions
                visible_actions = all_actions
                smart_skip_for_summary = []
            
            # 分类动作记录（基于过滤后的actions）
            scope_actions = [a for a in visible_actions if a.get("record_type") == "scope"]
            flow_actions = [a for a in visible_actions if a.get("record_type") == "flow"]
            parameter_actions = [a for a in visible_actions if a.get("record_type") == "parameter"]
            calculation_actions = [a for a in visible_actions if a.get("record_type") == "calculation"]
            pivot_actions = [a for a in visible_actions if a.get("intent") == "pivot_query"]
            # smart_skip从专门的列表获取（LLM视图时从all_actions，工作台视图时从visible_actions）
            smart_skip_actions = smart_skip_for_summary if view == "llm" else [a for a in visible_actions if a.get("intent") == "smart_skip"]
            
            # 构建结构化总结 - 基于lca_actions统一架构
            summary = {
                "session_id": session_id,
                "lca_scope": {},
                "process_flows": {
                    "inputs": {
                        "raw_material": [],              # 原材料
                        "process_energy": [],            # 机器能耗
                        "post_processing_energy": [],    # 后处理能耗
                        "feedstock_energy": [],          # 粉末制备能耗
                        "gas": [],                       # 气体
                        "cooling_media": []              # 冷却/加工液体
                    },
                    "outputs": {
                        "product": [],                   # 主要产品
                        "recovered_material": [],        # 回收材料
                        "waste": [],                     # 废料
                        "emission": []                   # 排放
                    }
                },
                "decision_chain": {
                    "total_actions": len(all_actions),
                    "actions_by_intent": {
                        "select_best": 0,
                        "refine_same": 0, 
                        "pivot_query": 0,
                        "calculate": len(calculation_actions)
                    },
                    "actions_by_type": {
                        "scope": len(scope_actions),
                        "flow": len(flow_actions),
                        "parameter": len(parameter_actions),
                        "calculation": len(calculation_actions),
                        "pivot": len(pivot_actions)
                    },
                    "link_relationships": 0,  # 有link_to的动作数量
                    "action_sequence": []     # 动作序列（最近10个）
                },
                "parameter_analysis": {
                    "total_parameters": len(parameter_actions),
                    "parameter_names": [],       # 参数名称列表
                    "parameter_values": [],      # 参数值列表
                    "parameter_units": [],       # 参数单位列表
                    "linked_parameters": 0       # 有链接关系的参数数量
                },
                "calculation_analysis": {
                    "total_calculations": len(calculation_actions),
                    "calculation_expressions": [],  # 计算表达式列表
                    "calculation_results": [],      # 计算结果列表
                    "calculation_units": [],        # 单位统计
                    "linked_calculations": 0        # 有链接关系的计算数量
                },
                "pivot_analysis": {
                    "total_pivots": len(pivot_actions),
                    "pivot_reasons": [],      # 失败原因统计
                    "success_after_pivot": 0, # pivot后成功的次数
                    "continuous_pivots": 0    # 连续pivot的次数
                },
                "smart_skip_analysis": {
                    "total_smart_skips": len(smart_skip_actions),
                    "skipped_categories": [],  # 跳过的类别列表
                    "skip_reasons": []         # 跳过原因统计
                },
                "statistics": {
                    "total_scopes_defined": len(scope_actions),
                    "total_flows_recorded": len(flow_actions),
                    "total_parameters": len(parameter_actions),
                    "total_calculations": len(calculation_actions),
                    "flows_by_type": {"Input": 0, "Output": 0},
                    "flows_by_category": {
                        # 输入流分类（工作台的11个LCI类别）
                        "Raw Material": 0, "Process Energy": 0, "Post-processing Energy": 0,
                        "Feedstock Energy": 0, "Gas": 0, "Cooling Media": 0,
                        # 输出流分类  
                        "Product": 0, "Recovered Material": 0, "Waste": 0, "Emission": 0
                    }
                },
                "completeness_assessment": {
                    "has_functional_unit": False,
                    "has_system_boundary": False, 
                    "has_impact_categories": False,
                    "has_geographical_scope": False,
                    "has_inputs": False,
                    "has_outputs": False,
                    "has_process_context": False,  # 是否包含工艺步骤信息
                    "data_quality_score": 0.0      # 基于决策特征的质量评分
                }
            }
            
            # 处理LCA范围数据（基于lca_actions）
            for scope_action in scope_actions:
                param_name = scope_action.get("parameter_name")
                if param_name:
                    summary["lca_scope"][param_name] = {
                        "parameter_value": scope_action.get("parameter_value"),
                        "action_id": scope_action.get("action_id"),
                        "intent": scope_action.get("intent"),
                        "timestamp": scope_action.get("timestamp"),
                        "selected_chunk_id": scope_action.get("selected_chunk", {}).get("chunk_id"),
                        "note": scope_action.get("note")
                    }
                
                # 更新完整性评估
                if param_name == "Function Unit":
                    summary["completeness_assessment"]["has_functional_unit"] = True
                elif param_name == "System Boundary":
                    summary["completeness_assessment"]["has_system_boundary"] = True
                elif param_name == "Impact Categories":
                    summary["completeness_assessment"]["has_impact_categories"] = True
                elif param_name == "Geographical Scope":
                    summary["completeness_assessment"]["has_geographical_scope"] = True
            
            # 处理工艺流数据（基于lca_actions）
            quality_scores = []
            
            for flow_action in flow_actions:
                flow_type = flow_action.get("flow_type")
                category = flow_action.get("category")
                
                if flow_type and category:
                    # 统计计数
                    summary["statistics"]["flows_by_type"][flow_type] += 1
                    summary["statistics"]["flows_by_category"][category] += 1
                    
                    # 构建流数据，包含决策追踪信息
                    flow_data = {
                        "name": flow_action.get("name"),
                        "value": flow_action.get("value"),
                        "unit": flow_action.get("unit"),
                        "location": flow_action.get("location"),
                        "cas_number": flow_action.get("cas_number"),
                        "process_name": flow_action.get("process_name"),
                        "action_id": flow_action.get("action_id"),
                        "intent": flow_action.get("intent"),
                        "link_to": flow_action.get("link_to"),
                        "note": flow_action.get("note"),
                        "selected_chunk_id": flow_action.get("selected_chunk", {}).get("chunk_id")
                }
                
                # 检查是否有工艺上下文信息
                if flow_action.get("process_name"):
                    summary["completeness_assessment"]["has_process_context"] = True
                    
                    # 数据质量评分（简化版：基于是否有 selected_chunk）
                    selected_chunk = flow_action.get("selected_chunk", {})
                    if selected_chunk and selected_chunk.get("chunk_id"):
                        quality_scores.append(1.0)  # 有数据来源即认为质量较高
                    else:
                        quality_scores.append(0.5)  # 无数据来源降低评分
                
                # 工作台的11个LCI类别映射（与keyword_suggester.py和工作台UI一致）
                category_mapping = {
                    # 输入流映射
                    "Raw Material": "raw_material",
                    "Process Energy": "process_energy",
                    "Post-processing Energy": "post_processing_energy",
                    "Feedstock Energy": "feedstock_energy",
                    "Gas": "gas",
                    "Cooling Media": "cooling_media",
                    # 输出流映射
                    "Product": "product",
                    "Recovered Material": "recovered_material",
                    "Waste": "waste",
                    "Emission": "emission"
                }
                
                category_key = category_mapping.get(category)
                if not category_key:
                    logger.warning(f"Unknown category: {category}")
                    continue
                
                if flow_type == "Input":
                    summary["process_flows"]["inputs"][category_key].append(flow_data)
                    summary["completeness_assessment"]["has_inputs"] = True
                elif flow_type == "Output":
                    summary["process_flows"]["outputs"][category_key].append(flow_data)
                    summary["completeness_assessment"]["has_outputs"] = True
            
            # 计算平均数据质量评分
            if quality_scores:
                summary["completeness_assessment"]["data_quality_score"] = round(
                    sum(quality_scores) / len(quality_scores), 3
                )
            
            # 分析决策链
            linked_actions = 0
            for action in all_actions:
                intent = action.get("intent")
                if intent:
                    # 🔥 使用 get() 避免 KeyError，支持动态 intent 类型
                    current_count = summary["decision_chain"]["actions_by_intent"].get(intent, 0)
                    summary["decision_chain"]["actions_by_intent"][intent] = current_count + 1
                
                if action.get("link_to"):
                    linked_actions += 1
            
            summary["decision_chain"]["link_relationships"] = linked_actions
            
            # 构建动作序列（最近10个）
            recent_actions = all_actions[-10:] if len(all_actions) > 10 else all_actions
            for action in recent_actions:
                summary["decision_chain"]["action_sequence"].append({
                    "action_id": action.get("action_id"),
                    "intent": action.get("intent"),
                    "record_type": action.get("record_type"),
                    "link_to": action.get("link_to"),
                    "timestamp": action.get("timestamp")
                })
            
            # 分析Pivot Query
            pivot_reasons = []
            success_after_pivot = 0
            continuous_pivots = 0
            
            for i, action in enumerate(all_actions):
                if action.get("intent") == "pivot_query":
                    # 收集失败原因
                    pivot_rationale = action.get("pivot_rationale")
                    if pivot_rationale:
                        pivot_reasons.append(pivot_rationale)
                    
                    # 检查pivot后是否有成功动作
                    if i < len(all_actions) - 1:
                        next_action = all_actions[i + 1]
                        if (next_action.get("intent") == "select_best" and 
                            next_action.get("link_to") == action.get("action_id")):
                            success_after_pivot += 1
                    
                    # 检查连续pivot
                    if i > 0:
                        prev_action = all_actions[i - 1]
                        if prev_action.get("intent") == "pivot_query":
                            continuous_pivots += 1
            
            summary["pivot_analysis"]["pivot_reasons"] = pivot_reasons
            summary["pivot_analysis"]["success_after_pivot"] = success_after_pivot
            summary["pivot_analysis"]["continuous_pivots"] = continuous_pivots
            
            # 处理smart_skip统计
            skipped_categories = []
            skip_reasons = []
            
            for skip_action in smart_skip_actions:
                category = skip_action.get("category")
                skip_reason = skip_action.get("skip_reason", "already_recorded")
                
                if category:
                    skipped_categories.append(category)
                if skip_reason:
                    skip_reasons.append(skip_reason)
            
            summary["smart_skip_analysis"]["skipped_categories"] = skipped_categories
            summary["smart_skip_analysis"]["skip_reasons"] = skip_reasons
            
            # 处理parameter统计 - 改为结构化列表
            parameters_list = []
            linked_parameters = 0
            
            for param_action in parameter_actions:
                # 构建结构化的参数对象（包含action_id）
                param_obj = {
                    "action_id": param_action.get("action_id"),
                    "parameter_name": param_action.get("parameter_name"),
                    "parameter_value": param_action.get("parameter_value"),
                    "parameter_unit": param_action.get("parameter_unit"),
                    "intent": param_action.get("intent"),
                    "link_to": param_action.get("link_to"),
                    "timestamp": param_action.get("timestamp")
                }
                parameters_list.append(param_obj)
                
                # 统计有链接关系的参数
                if param_action.get("link_to"):
                    linked_parameters += 1
            
            summary["parameter_analysis"] = {
                "total_parameters": len(parameter_actions),
                "parameters": parameters_list,  # 结构化列表（包含action_id）
                "linked_parameters": linked_parameters
            }
            
            # 处理calculation统计 - 改为结构化列表
            calculations_list = []
            linked_calculations = 0
            
            for calc_action in calculation_actions:
                # 构建结构化的计算对象（包含action_id）
                calc_obj = {
                    "action_id": calc_action.get("action_id"),
                    "calculation_expression": calc_action.get("calculation_expression"),
                    "calculation_result": calc_action.get("calculation_result"),
                    "calculation_unit": calc_action.get("calculation_unit"),
                    "data_dependencies": calc_action.get("data_dependencies", []),
                    "intent": calc_action.get("intent"),
                    "link_to": calc_action.get("link_to"),
                    "timestamp": calc_action.get("timestamp"),
                }
                calculations_list.append(calc_obj)
                
                # 统计有链接关系的计算
                if calc_action.get("link_to"):
                    linked_calculations += 1
            
            summary["calculation_analysis"]["calculations"] = calculations_list  # 结构化列表（包含action_id）
            summary["calculation_analysis"]["linked_calculations"] = linked_calculations
            
            logger.info(f"会话总结生成成功，包含{len(scope_actions)}个范围定义、{len(flow_actions)}条工艺流、{len(calculation_actions)}个计算动作、{len(pivot_actions)}个pivot动作")
            
            # 根据 format 参数返回不同格式
            if format == "json":
                # 返回完整 JSON 结构（专家工作台使用）
                return {
                "success": True,
                    "message": f"Session summary retrieved: {len(scope_actions)} scope definitions, {len(flow_actions)} process flows, {len(calculation_actions)} calculations, {len(pivot_actions)} pivot actions",
                "data": summary
            }
            else:  # format == "text" (默认)
                # 返回人类可读文本格式（LLM 使用，节省 80% tokens）
                text_summary = self._format_summary_as_text(
                    flow_actions, 
                    parameter_actions, 
                    calculation_actions,
                    view=view  # 传递view参数
                )
                return {
                    "success": True,
                    "message": "Session summary (text format)",
                    "data": {"text": text_summary}
                }
            
        except Exception as e:
            logger.error(f"获取会话总结失败: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to get session summary: {str(e)}"
            }
    
    def _format_summary_as_text(self, flow_actions: List[dict], parameter_actions: List[dict], calculation_actions: List[dict], view: str = "llm") -> str:
        """
        将 session summary 格式化为人类可读的文本
        
        🔥 重要：LLM视图不包含calculation统计，保证训练-推理一致性
        
        Args:
            flow_actions: 流动作列表
            parameter_actions: 参数动作列表
            calculation_actions: 计算动作列表
            view: 视图类型（"llm" 或 "workbench"）
        
        Returns:
            简洁的文本格式
        """
        lines = []
        
        # 统计信息
        if view == "llm":
            # LLM视图：只显示flow和parameter
            lines.append(f"Total Actions: {len(flow_actions) + len(parameter_actions)}")
            lines.append(f"Flows: {len(flow_actions)}, Parameters: {len(parameter_actions)}")
        else:
            # 工作台视图：包含calculation
            lines.append(f"Total Actions: {len(flow_actions) + len(parameter_actions) + len(calculation_actions)}")
            lines.append(f"Flows: {len(flow_actions)}, Parameters: {len(parameter_actions)}, Calculations: {len(calculation_actions)}")
        lines.append("")
        
        # 流记录部分
        if flow_actions:
            lines.append("Recorded Flows:")
            for f in flow_actions:
                name = f.get("name", "unknown")
                value = f.get("value", "?")
                unit = f.get("unit", "")
                flow_type = f.get("flow_type", "?")
                category = f.get("category", "?")
                
                # 格式：  - Ti6Al4V: 20.83 kg (Input/raw_material)
                line = f"  - {name}: {value} {unit} ({flow_type}/{category})".strip()
                lines.append(line)
            lines.append("")
        
        # 参数部分
        if parameter_actions:
            lines.append("Recorded Parameters:")
            for p in parameter_actions:
                name = p.get("parameter_name", "unknown")
                value = p.get("parameter_value", "?")
                unit = p.get("parameter_unit", "")
                
                # 格式：  - power: 950 W
                line = f"  - {name}: {value} {unit}".strip()
                lines.append(line)
            lines.append("")
        
        # 计算部分（仅工作台视图）
        if view == "workbench" and calculation_actions:
            lines.append("Recorded Calculations:")
            for c in calculation_actions:
                name = c.get("calculation_name", "unknown")
                result = c.get("result", "?")
                unit = c.get("unit", "")
                
                # 格式：  - energy: 3420000 J
                line = f"  - {name}: {result} {unit}".strip()
                lines.append(line)
            lines.append("")
        
        # 如果没有任何数据
        if not flow_actions and not parameter_actions:
            lines.append("No data recorded yet.")
        
        return "\n".join(lines)