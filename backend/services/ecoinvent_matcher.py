"""
Ecoinvent 流匹配服务

将用户提取的 LCI 数据与 ecoinvent 数据库中的 flows 和 processes 进行语义匹配
"""

import logging
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class EcoinventMatcher:
    """Ecoinvent 流匹配器"""
    
    # LLM 重写 Prompt 模板
    LLM_REWRITE_PROMPT = """Task: Translate a user-extracted flow name into a precise search term for the Ecoinvent database.

Flow Name: "{flow_name}"
Category: {category}
Material Context: {material_context}
Process Context: {process_context}

Instructions:
1. Identify the flow type from Category (Raw Material, Gas, Energy, Waste, etc.)
2. For Raw Material → use material name (e.g., "316L steel", "titanium alloy", "argon")
3. For Gas → use gas name (e.g., "argon", "nitrogen", "CO2")
4. For Energy → use "electricity" or specific energy type
5. For Recovered Material/Reuse → add "scrap" or "recycled" (e.g., "steel scrap")
6. For Waste → add "waste" (e.g., "steel waste", "metal waste")
7. Output ONLY the search term (2-4 words), no explanation

Examples:
- "X2CrNiMo1712 Stainless Steel" + Raw Material → "316L stainless steel"
- "Argon Gas" + Gas → "argon"
- "SLM Powder Reuse" + Recovered Material + "316L" → "steel scrap"
- "Non-recyclable Powder" + Waste + "steel" → "steel waste"
- "Electricity - SLM" + Process Energy → "electricity"

Search term:"""
    
    def __init__(self, db=None, enable_llm_rewrite: bool = False):
        """
        初始化匹配器
        
        Args:
            db: MongoDB 数据库连接
            enable_llm_rewrite: 是否启用 LLM 辅助重写（默认关闭，需要时开启）
        """
        self.db = db
        self.model = None
        self._initialized = False
        self._use_precomputed_vectors = True  # 使用预计算的 Qwen3-Embedding 向量
        self._enable_llm_rewrite = enable_llm_rewrite
        self._llm_client = None
        
    def _ensure_initialized(self):
        """确保初始化完成"""
        if self._initialized:
            return
            
        if self.db is None:
            from .mongodb_manager import mongodb_manager
            self.db = mongodb_manager.get_database()
        
        # 检查是否有预计算向量
        sample = self.db.flows.find_one({"embedding_vector": {"$exists": True}})
        if sample and sample.get("embedding_vector"):
            self._use_precomputed_vectors = True
            logger.info("✅ 使用预计算的 Qwen3-Embedding 向量 (1024维)")
            # 加载 Qwen3-Embedding 模型用于查询向量化（强制使用 CPU 避免 GPU 内存冲突）
            try:
                import torch
                device = "cpu"  # 强制使用 CPU，避免与 vLLM 争抢 GPU 内存
                self.model = SentenceTransformer(
                    '/home/Research_work/24_yzlin/LCA-LLM/models/Qwen3-embedding-0.6B',
                    device=device
                )
                logger.info(f"✅ Qwen3-Embedding 模型加载成功 (device={device})")
            except Exception as e:
                logger.warning(f"Qwen3-Embedding 加载失败，使用 all-MiniLM-L6-v2: {e}")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
                self._use_precomputed_vectors = False
        else:
            self._use_precomputed_vectors = False
            logger.info("加载 all-MiniLM-L6-v2 模型 (CPU)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        
        self._initialized = True
        logger.info("✅ EcoinventMatcher 初始化完成")
    
    def match_flow(self, 
                   flow_name: str, 
                   category: str = None,
                   flow_type: str = None,
                   top_k: int = 5,
                   context: Dict[str, Any] = None,
                   use_llm: bool = False,
                   required_unit: str = None) -> List[Dict[str, Any]]:
        """
        匹配单个流到 ecoinvent flows
        
        使用两阶段匹配：
        1. 文本搜索（MongoDB regex）获取候选
        2. 语义相似度排序（支持上下文增强和 LLM 重写）
        
        Args:
            flow_name: 用户提取的流名称（如 "Electricity", "316L stainless steel"）
            category: LCI 类别（如 "Process Energy", "Raw Material"）
            flow_type: 流类型（"Input" 或 "Output"）
            top_k: 返回前 k 个匹配结果
            context: 上下文信息，用于增强匹配（可选）
                     包含: note, selected_chunk, functional_unit 等
            use_llm: 是否使用 LLM 辅助重写（默认 False）
            required_unit: 要求的单位（可选），用于过滤单位不兼容的流量
            
        Returns:
            匹配结果列表，每个包含 ecoinvent flow 信息和相似度分数
        """
        self._ensure_initialized()
        
        try:
            # 🆕 阶段 0: 如果启用 LLM，先获取重写后的名称
            enhanced_query = self._build_enhanced_query(flow_name, category, context, use_llm=use_llm)
            search_term = enhanced_query if use_llm else flow_name
            
            # 🆕 阶段 1: 精确匹配（最高优先级）
            exact_match = self.db.flows.find_one({"name": {"$regex": f"^{search_term}$", "$options": "i"}})
            if exact_match:
                logger.info(f"✅ 精确匹配成功: {search_term} → {exact_match.get('name')}")
                return [{
                    "uuid": exact_match.get("uuid"),
                    "name": exact_match.get("name"),
                    "category": exact_match.get("category"),
                    "categories": exact_match.get("categories", []),
                    "flowType": exact_match.get("flowType"),
                    "unit": self._get_flow_unit(exact_match),
                    "similarity": 1.0,  # 精确匹配
                    "cas": exact_match.get("cas"),
                }]
            
            # 🆕 阶段 2: 关键词匹配（次优先级）
            # 提取关键词进行搜索
            keywords = search_term.lower().split()
            
            # 构建 MongoDB 查询
            text_filter = {"$or": [
                {"name": {"$regex": search_term, "$options": "i"}},
            ]}
            
            # 添加关键词匹配
            for kw in keywords[:3]:  # 最多用前3个关键词
                if len(kw) > 2:  # 忽略太短的词
                    text_filter["$or"].append({"name": {"$regex": kw, "$options": "i"}})
            
            # 根据类别添加过滤
            category_filter = self._build_search_filter(category, flow_type)
            if category_filter:
                combined_filter = {"$and": [text_filter, category_filter]}
            else:
                combined_filter = text_filter
            
            # 获取候选
            candidates = list(self.db.flows.find(combined_filter).limit(100))
            
            # 如果文本搜索结果太少，扩大搜索范围
            if len(candidates) < 10:
                candidates = list(self.db.flows.find(text_filter).limit(100))
            
            if not candidates:
                logger.warning(f"未找到候选 flows: {search_term}")
                return []
            
            # 阶段 3: 语义相似度排序（最后的手段）
            query = enhanced_query
            logger.debug(f"Enhanced query: {query}")
            query_embedding = self.model.encode(query)
            
            results = []
            for flow in candidates:
                # 优先使用预计算向量
                if self._use_precomputed_vectors and flow.get("embedding_vector"):
                    flow_embedding = np.array(flow["embedding_vector"])
                else:
                    flow_text = self._build_flow_text(flow)
                    flow_embedding = self.model.encode(flow_text)
                
                similarity = np.dot(query_embedding, flow_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(flow_embedding) + 1e-8
                )
                
                results.append({
                    "uuid": flow.get("uuid"),
                    "name": flow.get("name"),
                    "category": flow.get("category"),
                    "categories": flow.get("categories", []),
                    "flowType": flow.get("flowType"),
                    "unit": self._get_flow_unit(flow),
                    "similarity": float(similarity),
                    "cas": flow.get("cas"),
                })
            
            # 按相似度排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 如果指定了 required_unit，过滤单位不兼容的结果
            if required_unit:
                from .unit_compatibility import are_units_compatible
                filtered_results = []
                for result in results:
                    result_unit = result.get("unit", "")
                    if are_units_compatible(required_unit, result_unit):
                        filtered_results.append(result)
                    else:
                        logger.debug(f"过滤单位不兼容: {result['name']} ({result_unit}) vs required ({required_unit})")
                
                if filtered_results:
                    logger.info(f"单位过滤: {len(results)} → {len(filtered_results)} (required: {required_unit})")
                    return filtered_results[:top_k]
                else:
                    logger.warning(f"单位过滤后无结果，返回原始结果 (required: {required_unit})")
                    return results[:top_k]
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"匹配流失败: {e}")
            return []
    
    def match_process(self,
                      process_name: str,
                      location: str = None,
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        匹配到 ecoinvent processes
        
        Args:
            process_name: 工艺名称（如 "SLM printing", "powder atomization"）
            location: 地理位置
            top_k: 返回前 k 个匹配结果
            
        Returns:
            匹配结果列表
        """
        self._ensure_initialized()
        
        try:
            # 构建查询
            query_parts = [process_name]
            if location:
                query_parts.append(location)
            query = " ".join(query_parts)
            
            query_embedding = self.model.encode(query)
            
            # 获取候选 processes
            candidates = list(self.db.processes.find().limit(500))
            
            results = []
            for proc in candidates:
                proc_text = f"{proc.get('name', '')} {' '.join(proc.get('categories', []))}"
                proc_embedding = self.model.encode(proc_text)
                
                similarity = np.dot(query_embedding, proc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(proc_embedding)
                )
                
                results.append({
                    "uuid": proc.get("uuid"),
                    "name": proc.get("name"),
                    "categories": proc.get("categories", []),
                    "location": proc.get("location"),
                    "similarity": float(similarity),
                })
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"匹配工艺失败: {e}")
            return []
    
    def batch_match_session(self, session_id: str, use_llm_rewrite: bool = False) -> Dict[str, Any]:
        """
        批量匹配一个会话中的所有 LCI 数据（支持上下文增强和 LLM 重写）
        
        Args:
            session_id: 会话 ID
            use_llm_rewrite: 是否使用 LLM 辅助重写流名称（默认 False）
            
        Returns:
            匹配结果，包含每个 flow 的候选匹配
        """
        self._ensure_initialized()
        
        try:
            # 获取会话中的所有记录
            records = list(self.db.lca_actions.find({
                "session_id": session_id,
                "record_type": "flow"
            }))
            
            if not records:
                return {
                    "success": False,
                    "error": "未找到该会话的 LCI 记录"
                }
            
            # 获取 functional_unit 用于上下文增强
            scope_record = self.db.lca_actions.find_one({
                "session_id": session_id,
                "record_type": "scope"
            })
            functional_unit = ""
            if scope_record:
                functional_unit = scope_record.get("description", "") or scope_record.get("parameter_name", "")
            
            # 🔥 从 Raw Material 记录中提取材料信息，增强 functional_unit
            material_flows = [r for r in records if r.get("category") == "Raw Material"]
            if material_flows:
                material_names = [r.get("name", "") for r in material_flows]
                material_context = ", ".join(material_names)
                if functional_unit:
                    functional_unit = f"{functional_unit} (materials: {material_context})"
                else:
                    functional_unit = f"Materials: {material_context}"
                logger.info(f"增强 functional_unit: {functional_unit}")
            
            logger.info(f"批量匹配 session {session_id}: {len(records)} flows, FU: {functional_unit[:100]}...")
            
            results = []
            for record in records:
                flow_name = record.get("name")
                category = record.get("category")
                flow_type = record.get("flow_type")
                
                # Product 类型不需要 Ecoinvent 匹配（它是系统输出，不是环境流）
                if category == "Product":
                    results.append({
                        "action_id": record.get("action_id"),
                        "original": {
                            "name": flow_name,
                            "category": category,
                            "flow_type": flow_type,
                            "value": record.get("value"),
                            "unit": record.get("unit"),
                        },
                        "matches": [],
                        "skip_reason": "Product flows are system outputs, not environmental flows"
                    })
                    continue
                
                # 构建上下文信息用于增强匹配（包含 session_id 用于提取全局材料信息）
                context = {
                    "session_id": session_id,  # 🆕 添加 session_id
                    "functional_unit": functional_unit,
                    "note": record.get("note", ""),
                    "selected_chunk": record.get("selected_chunk", {}),
                    "process_name": record.get("process_name", ""),
                }
                
                matches = self.match_flow(
                    flow_name=flow_name,
                    category=category,
                    flow_type=flow_type,
                    top_k=3,
                    context=context,
                    use_llm=use_llm_rewrite,
                    required_unit=record.get("unit")  # 传递单位进行过滤
                )
                
                results.append({
                    "action_id": record.get("action_id"),
                    "original": {
                        "name": flow_name,
                        "category": category,
                        "flow_type": flow_type,
                        "value": record.get("value"),
                        "unit": record.get("unit"),
                    },
                    "matches": matches
                })
            
            return {
                "success": True,
                "session_id": session_id,
                "total_flows": len(records),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"批量匹配失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def confirm_match(self, 
                      session_id: str, 
                      action_id: str, 
                      ecoinvent_uuid: str) -> Dict[str, Any]:
        """
        确认一个匹配结果
        
        Args:
            session_id: 会话 ID
            action_id: 动作 ID
            ecoinvent_uuid: 选择的 ecoinvent flow UUID
            
        Returns:
            更新结果
        """
        try:
            # 获取 ecoinvent flow 信息
            flow = self.db.flows.find_one({"uuid": ecoinvent_uuid})
            if not flow:
                return {"success": False, "error": "未找到该 ecoinvent flow"}
            
            # 更新 lca_actions 记录
            result = self.db.lca_actions.update_one(
                {"session_id": session_id, "action_id": action_id},
                {"$set": {
                    "ecoinvent_match": {
                        "uuid": ecoinvent_uuid,
                        "name": flow.get("name"),
                        "category": flow.get("category"),
                        "confirmed": True
                    }
                }}
            )
            
            if result.modified_count > 0:
                return {
                    "success": True,
                    "message": f"已确认匹配: {flow.get('name')}"
                }
            else:
                return {"success": False, "error": "更新失败"}
                
        except Exception as e:
            logger.error(f"确认匹配失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_search_filter(self, category: str, flow_type: str) -> Dict:
        """根据 LCI 类别构建 MongoDB 查询过滤器"""
        filter_dict = {}
        
        # 根据类别映射到 ecoinvent flowType
        if category in ["Raw Material", "Gas", "Cooling Media"]:
            # 产品流或基本流
            filter_dict["flowType"] = {"$in": ["PRODUCT_FLOW", "ELEMENTARY_FLOW"]}
        elif category in ["Process Energy", "Post-processing Energy", "Feedstock Energy"]:
            # 能源相关
            filter_dict["$or"] = [
                {"name": {"$regex": "electricity|energy|power", "$options": "i"}},
                {"categories": {"$in": ["Energy carriers and technologies"]}}
            ]
        elif category == "Emission":
            filter_dict["flowType"] = "ELEMENTARY_FLOW"
            filter_dict["categories"] = {"$in": ["Emission to air", "Emission to water", "Emission to soil"]}
        elif category == "Waste":
            filter_dict["$or"] = [
                {"flowType": "WASTE_FLOW"},
                {"categories": {"$regex": "waste", "$options": "i"}}
            ]
        
        return filter_dict
    
    def _get_candidate_flows(self, filter_dict: Dict, limit: int = 500) -> List[Dict]:
        """获取候选 flows"""
        try:
            if filter_dict:
                return list(self.db.flows.find(filter_dict).limit(limit))
            else:
                return list(self.db.flows.find().limit(limit))
        except Exception as e:
            logger.error(f"获取候选 flows 失败: {e}")
            return []
    
    def _build_enhanced_query(self, flow_name: str, category: str = None, context: Dict[str, Any] = None, use_llm: bool = False) -> str:
        """
        构建上下文增强的查询字符串
        
        通过利用 note、selected_chunk、functional_unit 等上下文信息，
        构建更精确的语义查询，解决"Solid Waste"无法匹配"316L steel waste"的问题
        
        Args:
            flow_name: 原始流名称（如 "Solid Waste"）
            category: LCI 类别（如 "Waste"）
            context: 上下文信息字典，包含:
                - note: 备注信息（如 "10.2% of input material"）
                - selected_chunk: 文档片段
                - functional_unit: 功能单位（如 "316L stainless steel washers"）
                - process_name: 工艺名称
            use_llm: 是否使用 LLM 重写（默认 False）
                
        Returns:
            增强后的查询字符串
        """
        # 如果启用 LLM 重写，先尝试用 LLM 重写流名称
        # use_llm 参数直接控制是否使用 LLM 重写（不再依赖 _enable_llm_rewrite）
        if use_llm and context:
            # 跳过 Product 类型（不需要匹配 Ecoinvent）
            if category == "Product":
                logger.info(f"跳过 Product 类型的 LLM 重写: {flow_name}")
                return flow_name
            
            rewritten_name = self._llm_rewrite_flow_name_sync(flow_name, category, context)
            if rewritten_name != flow_name:
                # LLM 重写成功，直接使用重写后的名称
                logger.info(f"使用 LLM 重写结果: '{rewritten_name}'")
                return rewritten_name
        
        query_parts = [flow_name]
        
        if not context:
            # 无上下文时，仅使用类别增强
            if category:
                query_parts.append(category)
            return " ".join(query_parts)
        
        # 1. 从 functional_unit 提取材料关键词（最有价值）
        functional_unit = context.get("functional_unit", "")
        if functional_unit:
            # 提取材料相关关键词（如 "316L", "stainless steel", "aluminum"）
            material_keywords = self._extract_material_keywords(functional_unit)
            if material_keywords:
                query_parts.extend(material_keywords)
                logger.debug(f"从 functional_unit 提取材料关键词: {material_keywords}")
        
        # 2. 从 note 提取上下文（次有价值）
        note = context.get("note", "")
        if note:
            # 提取 note 中的关键信息
            note_keywords = self._extract_note_keywords(note)
            if note_keywords:
                query_parts.extend(note_keywords)
        
        # 3. 从 selected_chunk 提取材料信息（如果有实际内容）
        selected_chunk = context.get("selected_chunk", {})
        if isinstance(selected_chunk, dict):
            chunk_text = selected_chunk.get("text", "") or selected_chunk.get("content", "")
            if chunk_text and len(chunk_text) > 10:
                # 提取 chunk 中的材料关键词
                chunk_keywords = self._extract_material_keywords(chunk_text)
                if chunk_keywords:
                    query_parts.extend(chunk_keywords[:2])  # 最多取2个
        
        # 4. 添加类别（帮助区分 Input/Output）
        if category:
            query_parts.append(category)
        
        # 去重并构建查询
        seen = set()
        unique_parts = []
        for part in query_parts:
            part_lower = part.lower()
            if part_lower not in seen and len(part) > 1:
                seen.add(part_lower)
                unique_parts.append(part)
        
        enhanced_query = " ".join(unique_parts)
        logger.info(f"上下文增强查询: '{flow_name}' -> '{enhanced_query}'")
        return enhanced_query
    
    def _extract_material_keywords(self, text: str) -> List[str]:
        """
        从文本中提取材料相关关键词
        
        Args:
            text: 输入文本（如 "316L stainless steel washers via SLM"）
            
        Returns:
            材料关键词列表（如 ["316L", "stainless", "steel"]）
        """
        if not text:
            return []
        
        # 常见材料关键词模式
        material_patterns = [
            # 金属合金（包括 X2CrNiMo 系列）
            r'\b(316L|304|X2CrNiMo\d+|Ti-?6Al-?4V|Inconel|AlSi10Mg|17-?4\s?PH)\b',
            # 金属类型
            r'\b(stainless\s+steel|steel|aluminum|aluminium|titanium|copper|nickel|chromium)\b',
            # 塑料
            r'\b(PLA|ABS|PETG|nylon|polyethylene|polypropylene|plastic)\b',
            # 能源
            r'\b(electricity|power|energy|natural\s+gas)\b',
        ]
        
        import re
        keywords = []
        text_lower = text.lower()
        
        for pattern in material_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            keywords.extend(matches)
        
        # 去重并返回
        return list(dict.fromkeys(keywords))[:5]  # 最多5个关键词
    
    def _extract_note_keywords(self, note: str) -> List[str]:
        """
        从 note 字段提取有用的关键词
        
        Args:
            note: 备注信息（如 "10.2% of input material", "SLM machine"）
            
        Returns:
            关键词列表
        """
        if not note:
            return []
        
        # 提取工艺相关关键词
        import re
        keywords = []
        
        # 工艺关键词
        process_patterns = [
            r'\b(SLM|SLS|FDM|EBM|DMLS|3D\s+printing|additive\s+manufacturing)\b',
            r'\b(atomization|milling|machining|heat\s+treatment)\b',
            r'\b(laser|sintering|melting|printing)\b',
        ]
        
        for pattern in process_patterns:
            matches = re.findall(pattern, note, re.IGNORECASE)
            keywords.extend(matches)
        
        return list(dict.fromkeys(keywords))[:3]
    
    def _build_flow_text(self, flow: Dict) -> str:
        """构建 flow 的文本表示用于嵌入"""
        parts = [flow.get("name", "")]
        if flow.get("categories"):
            parts.extend(flow.get("categories", []))
        if flow.get("description"):
            parts.append(flow.get("description", "")[:200])
        return " ".join(parts)
    
    def _llm_rewrite_flow_name(self, flow_name: str, category: str, context: Dict[str, Any]) -> str:
        """
        使用 LLM 将模糊的流名称重写为精确的 Ecoinvent 搜索词
        
        Args:
            flow_name: 原始流名称（如 "Solid Waste"）
            category: LCI 类别（如 "Waste"）
            context: 上下文信息（包括 session_id 用于提取全局材料信息）
            
        Returns:
            重写后的搜索词（如 "steel scrap"）
        """
        try:
            print(f"[LLM REWRITE] 开始重写: {flow_name}")
            logger.info(f"LLM 重写开始: {flow_name}")
            # 提取上下文信息
            functional_unit = context.get("functional_unit", "")
            note = context.get("note", "")
            process_name = context.get("process_name", "")
            selected_chunk = context.get("selected_chunk", {})
            session_id = context.get("session_id")
            
            # 🆕 从 session 中提取材料信息
            material_hints = []
            if session_id:
                try:
                    # 确保数据库连接可用
                    if self.db is None:
                        from .mongodb_manager import mongodb_manager
                        self.db = mongodb_manager.get_database()
                    
                    # 查询同一 session 的所有 Raw Material 和 Process Material
                    input_flows = self.db.lci_records.find({
                        "session_id": session_id,
                        "category": {"$in": ["Raw Material", "Process Material"]}
                    })
                    
                    # 提取材料信息
                    for flow in input_flows:
                        flow_name_input = flow.get("flow_name", "")
                        note_input = flow.get("note", "")
                        combined_text = f"{flow_name_input} {note_input}".lower()
                        
                        # 提取常见材料类型
                        if "316l" in combined_text:
                            material_hints.append("316L stainless steel")
                        elif "stainless steel" in combined_text or "x2crni" in combined_text:
                            material_hints.append("stainless steel")
                        elif "steel" in combined_text and "powder" in combined_text:
                            material_hints.append("steel powder")
                        elif "titanium" in combined_text or "ti6al4v" in combined_text or "ti-6al-4v" in combined_text:
                            material_hints.append("titanium alloy")
                        elif "aluminum" in combined_text or "aluminium" in combined_text:
                            material_hints.append("aluminum")
                    
                    # 去重
                    material_hints = list(set(material_hints))
                    
                    if material_hints:
                        print(f"[LLM REWRITE] 从 session 提取到材料信息: {material_hints}")
                        logger.info(f"Session 材料信息: {material_hints}")
                    
                except Exception as e:
                    logger.warning(f"查询 session 材料信息失败: {e}")
            
            # 构建材料上下文（包含提取的材料信息）
            material_context = functional_unit or "unknown"
            if material_hints:
                material_context = f"{material_context}; Materials in this process: {', '.join(material_hints)}"
            
            # 构建工艺上下文
            process_context = process_name or note or "unknown"
            if isinstance(selected_chunk, dict):
                chunk_text = selected_chunk.get("text", "") or selected_chunk.get("content", "")
                if chunk_text and len(chunk_text) > 10:
                    process_context = f"{process_context}; {chunk_text[:100]}"
            
            # 构建 Prompt
            prompt = self.LLM_REWRITE_PROMPT.format(
                flow_name=flow_name,
                category=category or "unknown",
                material_context=material_context,
                process_context=process_context
            )
            
            # 调用 LLM
            if self._llm_client is None:
                # 延迟初始化 LLM 客户端
                try:
                    from openai import OpenAI
                    self._llm_client = OpenAI(
                        base_url="http://localhost:8080/v1",  # vLLM 服务端口
                        api_key="not-needed"
                    )
                except Exception as e:
                    logger.warning(f"LLM 客户端初始化失败: {e}")
                    return flow_name
            
            response = self._llm_client.chat.completions.create(
                model="qwen-lca",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,  # 低温度，更确定性
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}  # 禁用思考模式
            )
            
            rewritten = response.choices[0].message.content.strip()
            # 清理输出（去除引号、多余空格、思考标签等）
            if "<think>" in rewritten and "</think>" in rewritten:
                rewritten = rewritten.split("</think>")[-1].strip()
            rewritten = rewritten.strip('"\'').strip()
            
            print(f"[LLM REWRITE] 原始: {flow_name} -> 重写: {rewritten}")
            
            if rewritten and len(rewritten) < 100:
                logger.info(f"LLM 重写成功: '{flow_name}' -> '{rewritten}'")
                return rewritten
            else:
                logger.warning(f"LLM 重写结果过长或为空，使用原始名称: {flow_name}")
                return flow_name
                
        except Exception as e:
            logger.warning(f"LLM 重写失败，使用原始名称: {e}")
            return flow_name
    
    def _llm_rewrite_flow_name_sync(self, flow_name: str, category: str, context: Dict[str, Any]) -> str:
        """同步版本的 LLM 重写方法（直接调用）"""
        return self._llm_rewrite_flow_name(flow_name, category, context)
    
    def _get_flow_unit(self, flow: Dict) -> str:
        """获取 flow 的单位"""
        try:
            flow_props = flow.get("flowProperties", [])
            for prop in flow_props:
                if prop.get("isRefFlowProperty"):
                    fp = prop.get("flowProperty", {})
                    return fp.get("refUnit", "")
            return ""
        except:
            return ""


# 全局实例
_matcher_instance = None

def get_ecoinvent_matcher(enable_llm_rewrite: bool = False) -> EcoinventMatcher:
    """获取全局匹配器实例"""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = EcoinventMatcher(enable_llm_rewrite=enable_llm_rewrite)
    elif enable_llm_rewrite and not _matcher_instance._enable_llm_rewrite:
        # 如果需要启用 LLM 重写但当前实例未启用，更新设置
        _matcher_instance._enable_llm_rewrite = True
    return _matcher_instance
