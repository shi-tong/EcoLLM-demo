#!/usr/bin/env python3
"""
向量化LCI知识库
使用SentenceTransformer进行语义搜索
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from .knowledge_base import LCIDatabase
from .mongodb_manager import get_flows_collection, mongodb_manager, initialize_mongodb

logger = logging.getLogger(__name__)

class VectorizedLCIDatabase(LCIDatabase):
    """向量化LCI数据库，支持语义搜索"""
    
    def __init__(self, mongodb_uri: str = "mongodb://localhost:27017/", 
                 db_name: str = "lci_database",
                 embedding_model: str = "/home/Research_work/24_yzlin/LCA-LLM/models/Qwen3-embedding-0.6B"):
        """
        初始化向量化LCI数据库
        
        Args:
            mongodb_uri: MongoDB连接URI
            db_name: 数据库名称
            embedding_model: 嵌入模型路径
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
        # 确保MongoDB连接池已初始化
        if not mongodb_manager.client:
            initialize_mongodb(mongodb_uri, db_name)
        
        # 使用连接池获取集合
        self.flows_collection = get_flows_collection()
        
        # 嵌入模型采用延迟加载，仅在首次搜索时初始化
        # self._initialize_embedding_model()  # 注释掉立即加载
        
        logger.info("向量化LCI数据库已连接到连接池（嵌入模型延迟加载）")
    

    
    def _initialize_embedding_model(self):
        """初始化嵌入模型"""
        try:
            logger.info(f"加载嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("嵌入模型加载成功")
            
        except Exception as e:
            logger.error(f"嵌入模型加载失败: {str(e)}")
            self.embedding_model = None
    
    def vectorize_existing_data(self) -> bool:
        """
        向量化现有数据
        
        Returns:
            bool: 是否成功
        """
        if not self.client or not self.embedding_model:
            logger.error("数据库或嵌入模型未初始化")
            return False
        
        try:
            logger.info("开始向量化现有LCI数据...")
            
            # 获取所有未向量化的数据
            unvectorized_flows = list(self.flows_collection.find({
                "$or": [
                    {"embedding": {"$exists": False}},
                    {"vectorized": {"$ne": True}}
                ]
            }))
            
            if not unvectorized_flows:
                logger.info("所有数据已向量化")
                return True
            
            logger.info(f"找到{len(unvectorized_flows)}条未向量化的数据")
            
            # 批量向量化
            for i, flow in enumerate(unvectorized_flows):
                try:
                    # 创建向量化文本
                    vector_text = self._create_vector_text(flow)
                    
                    # 生成嵌入向量
                    embedding = self.embedding_model.encode(vector_text)
                    
                    # 更新数据库
                    self.flows_collection.update_one(
                        {"_id": flow["_id"]},
                        {
                            "$set": {
                                "embedding": embedding.tolist(),
                                "embedding_text": vector_text,
                                "vectorized": True
                            }
                        }
                    )
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"已向量化 {i + 1}/{len(unvectorized_flows)} 条数据")
                
                except Exception as e:
                    logger.error(f"向量化数据失败 (ID: {flow.get('_id')}): {str(e)}")
                    continue
            
            logger.info("数据向量化完成")
            return True
            
        except Exception as e:
            logger.error(f"向量化过程失败: {str(e)}")
            return False
    
    def _create_vector_text(self, flow: Dict[str, Any]) -> str:
        """
        为LCI流创建向量化文本
        
        Args:
            flow: LCI流数据
            
        Returns:
            str: 用于向量化的文本
        """
        text_parts = []
        
        # 基本信息
        if flow.get("name"):
            text_parts.append(flow["name"])
        
        if flow.get("reference_product"):
            text_parts.append(f"product: {flow['reference_product']}")
        
        if flow.get("description"):
            text_parts.append(flow["description"])
        
        # 类别信息
        if flow.get("categories"):
            categories_text = ", ".join(flow["categories"])
            text_parts.append(f"categories: {categories_text}")
        
        # 位置信息
        if flow.get("location"):
            text_parts.append(f"location: {flow['location']}")
        
        # 单位信息
        if flow.get("unit"):
            text_parts.append(f"unit: {flow['unit']}")
        
        return " | ".join(text_parts)
    
    def search_flows(self, query: str, context: List[Dict[str, Any]] = None, 
                    k: int = 5, use_hybrid: bool = True, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        搜索LCI流数据（向量化搜索）
        
        Args:
            query: 搜索查询
            context: PDF上下文信息
            k: 返回结果数量
            use_hybrid: 是否使用混合搜索（向量+关键词）
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤（默认0.3，平衡准确性和召回率）
            
        Returns:
            List[Dict[str, Any]]: 匹配的LCI流数据
        """
        if not mongodb_manager.client:
            logger.error("数据库连接池不可用")
            return []
        
        if not self.embedding_model:
            logger.info("嵌入模型未初始化，正在延迟加载...")
            self._initialize_embedding_model()
            
            # 如果初始化失败，降级到关键词搜索
            if not self.embedding_model:
                logger.warning("嵌入模型加载失败，使用关键词搜索")
                return self._keyword_search(query, k)
        
        try:
            if use_hybrid:
                # 混合搜索：向量搜索 + 关键词搜索
                vector_results = self._vector_search(query, k * 2, similarity_threshold)
                keyword_results = self._keyword_search(query, k)
                
                # 合并结果并去重
                combined_results = self._merge_search_results(vector_results, keyword_results)
                
                # 按相似度排序并限制数量
                combined_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                return combined_results[:k]
            else:
                # 纯向量搜索
                return self._vector_search(query, k, similarity_threshold)
                
        except Exception as e:
            logger.error(f"向量化搜索失败: {str(e)}")
            return []
    
    def _vector_search(self, query: str, k: int, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """向量搜索"""
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(query)
            
            # 获取所有向量化的数据
            vectorized_flows = list(self.flows_collection.find({
                "embedding_vector": {"$exists": True}
            }))
            
            if not vectorized_flows:
                logger.warning("没有找到向量化的数据")
                return []
            
            # 计算相似度
            similarities = []
            for flow in vectorized_flows:
                if "embedding_vector" in flow:
                    flow_embedding = np.array(flow["embedding_vector"])
                    similarity = np.dot(query_embedding, flow_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(flow_embedding)
                    )
                    # 只保留高于阈值的结果
                    if similarity >= similarity_threshold:
                        similarities.append((flow, similarity))
            
            # 排序并返回结果
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for flow, similarity in similarities[:k]:
                flow_copy = flow.copy()
                flow_copy.pop('_id', None)
                flow_copy.pop('embedding_vector', None)  # 移除1024维向量数据
                flow_copy['similarity_score'] = float(similarity)
                results.append(flow_copy)
            
            logger.info(f"向量搜索完成，找到{len(results)}个结果（阈值: {similarity_threshold}）")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """关键词搜索"""
        try:
            results = []
            
            # 文本搜索
            text_results = list(self.flows_collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(k))
            
            results.extend(text_results)
            
            # 类别搜索
            category_results = list(self.flows_collection.find({
                "categories": {"$regex": query, "$options": "i"}
            }).limit(k // 2))
            
            results.extend(category_results)
            
            # 产品搜索
            product_results = list(self.flows_collection.find({
                "reference_product": {"$regex": query, "$options": "i"}
            }).limit(k // 2))
            
            results.extend(product_results)
            
            # 去重
            unique_results = self._deduplicate_results(results)
            
            # 移除_id字段和向量数据
            for result in unique_results:
                result.pop('_id', None)
                result.pop('embedding_vector', None)  # 移除1024维向量数据
            
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {str(e)}")
            return []
    
    def _merge_search_results(self, vector_results: List[Dict], 
                            keyword_results: List[Dict]) -> List[Dict]:
        """合并搜索结果"""
        merged = {}
        
        # 添加向量搜索结果
        for result in vector_results:
            uuid = result.get('uuid')
            if uuid:
                merged[uuid] = result
        
        # 添加关键词搜索结果
        for result in keyword_results:
            uuid = result.get('uuid')
            if uuid and uuid not in merged:
                merged[uuid] = result
        
        return list(merged.values())
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重结果"""
        seen_uuids = set()
        unique_results = []
        
        for result in results:
            uuid = result.get('uuid')
            if uuid and uuid not in seen_uuids:
                seen_uuids.add(uuid)
                unique_results.append(result)
        
        return unique_results
    
    def get_flow_by_uuid(self, uuid: str) -> Dict[str, Any]:
        """根据UUID获取流数据"""
        if not mongodb_manager.client:
            logger.error("数据库连接池不可用")
            return {}
        
        try:
            result = self.flows_collection.find_one({"uuid": uuid})
            if result:
                result.pop('_id', None)
                result.pop('embedding', None)  # 移除向量数据
                return result
            else:
                logger.warning(f"未找到UUID为{uuid}的流数据")
                return {}
                
        except Exception as e:
            logger.error(f"获取流数据失败: {str(e)}")
            return {}
    

    


class VectorizedPermanentLCIDatabase(LCIDatabase):
    """向量化永久LCI知识库的通用接口"""
    
    def __init__(self, backend: str = "vectorized", **kwargs):
        self.backend = backend
        
        if backend == "vectorized":
            self.database = VectorizedLCIDatabase(**kwargs)
        else:
            logger.warning(f"未知的数据库后端: {backend}")
            self.database = VectorizedLCIDatabase(**kwargs)
    
    def search_flows(self, query: str, context: List[Dict[str, Any]] = None, 
                    k: int = 5, use_hybrid: bool = True, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """搜索LCI流数据"""
        return self.database.search_flows(query, context, k, use_hybrid, similarity_threshold)
    
    def get_flow_by_uuid(self, uuid: str) -> Dict[str, Any]:
        """根据UUID获取流数据"""
        return self.database.get_flow_by_uuid(uuid)
    
    def vectorize_existing_data(self) -> bool:
        """向量化现有数据"""
        return self.database.vectorize_existing_data()
