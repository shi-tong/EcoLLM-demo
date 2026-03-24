import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import List, Dict, Any
import logging
import os
import tempfile
import shutil
from abc import ABC, abstractmethod
from .mongodb_manager import get_flows_collection, mongodb_manager

logger = logging.getLogger(__name__)

class TemporaryKnowledgeBase:
    """临时RAG知识库，使用ChromaDB存储会话级别的PDF内容"""
    
    def __init__(self, collection_name: str = None):
        """
        初始化临时知识库
        
        Args:
            collection_name: ChromaDB集合名称
        """
        self.collection_name = collection_name or f"temp_session_{id(self)}"
        
        # 创建会话专用的临时目录
        self.temp_dir = tempfile.mkdtemp(prefix=f"chroma_session_{self.collection_name}_")
        
        # 创建会话隔离的ChromaDB客户端
        self.chroma_client = chromadb.PersistentClient(
            path=self.temp_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        logger.info(f"创建会话隔离的ChromaDB实例: {self.temp_dir}")
        logger.info(f"集合名称: {self.collection_name}")
        
        # 文档搜索使用all-MiniLM-L6-v2嵌入模型（384维，文本检索效果更好）
        # 注意：与LCI数据库的Qwen3-embedding分离，因为临时知识库每次重新创建
        
        # 优先使用本地模型路径
        local_model_path = "/home/Research_work/24_yzlin/LCA-LLM/models/all-MiniLM-L6-v2"
        
        try:
            # 首先尝试本地模型
            if os.path.exists(local_model_path):
                self.embeddings = SentenceTransformerEmbeddings(
                    model_name=local_model_path,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"✅ 成功加载本地all-MiniLM-L6-v2模型: {local_model_path}")
            else:
                # 如果本地模型不存在，尝试在线下载
                logger.info("本地模型不存在，尝试在线下载...")
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.embeddings = SentenceTransformerEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("✅ 成功下载并加载all-MiniLM-L6-v2模型")
                
        except Exception as e:
            logger.error(f"❌ 加载all-MiniLM-L6-v2模型失败: {e}")
            logger.info("🔄 回退到简化嵌入方案...")
            
            # 简化的备用嵌入方案
            class SimpleEmbeddings:
                def __init__(self):
                    self.model_name = "simple_hash_embedding"
                    logger.warning("使用简化哈希嵌入作为临时方案")
                
                def embed_documents(self, texts):
                    import hashlib
                    embeddings = []
                    for text in texts:
                        # 使用文本哈希生成384维向量
                        hash_obj = hashlib.md5(text.encode())
                        hash_bytes = hash_obj.digest()
                        # 扩展到384维
                        embedding = []
                        for i in range(384):
                            embedding.append((hash_bytes[i % len(hash_bytes)] / 255.0))
                        embeddings.append(embedding)
                    return embeddings
                
                def embed_query(self, text):
                    return self.embed_documents([text])[0]
            
            self.embeddings = SimpleEmbeddings()
            logger.info("✅ 临时嵌入方案已就绪")
        logger.info("临时知识库使用all-MiniLM-L6-v2嵌入模型 (384维，优化文本检索)")
            
        self.vectorstore = None
        self.documents = []
        
    def add_documents(self, documents: List[Document]):
        """添加文档到临时知识库"""
        try:
            self.documents = documents
            
            # 过滤复杂元数据，只保留简单类型
            filtered_documents = filter_complex_metadata(documents)
            
            # 创建会话隔离的ChromaDB向量存储
            self.vectorstore = Chroma.from_documents(
                documents=filtered_documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                client=self.chroma_client,
                persist_directory=self.temp_dir
            )
            
            logger.info(f"成功添加{len(filtered_documents)}个文档到临时知识库")
            
        except Exception as e:
            logger.error(f"添加文档到临时知识库失败: {str(e)}")
            raise e
    
    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """查询临时知识库"""
        if not self.vectorstore:
            return []
        
        try:
            # 相似性搜索
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                # 修复相似度分数：ChromaDB可能返回距离而非相似度
                # 将距离转换为相似度（0-1范围）
                if score > 1.0:
                    # 如果是距离度量，转换为相似度
                    normalized_score = max(0.0, 1.0 / (1.0 + score))
                else:
                    # 如果已经是相似度，确保在0-1范围内
                    normalized_score = max(0.0, min(1.0, score))
                
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": normalized_score,
                    "raw_score": score  # 保留原始分数用于调试
                })
            
            logger.info(f"临时知识库查询完成，返回{len(formatted_results)}个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"查询临时知识库失败: {str(e)}")
            return []
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索临时知识库 (兼容性方法，调用query)"""
        return self.query(query, k=top_k)
    
    def cleanup(self):
        """清理会话相关的ChromaDB数据和临时目录"""
        try:
            if hasattr(self, 'vectorstore') and self.vectorstore:
                # 清理向量存储
                try:
                    # 尝试删除集合
                    if hasattr(self.vectorstore, '_collection'):
                        self.vectorstore._collection.delete()
                except Exception as e:
                    logger.warning(f"删除向量存储集合失败: {e}")
                self.vectorstore = None
            
            if hasattr(self, 'chroma_client') and self.chroma_client:
                # 重置ChromaDB客户端
                try:
                    self.chroma_client.reset()
                except Exception as e:
                    logger.warning(f"重置ChromaDB客户端失败: {e}")
                
                # 强制关闭客户端连接
                try:
                    if hasattr(self.chroma_client, '_server'):
                        self.chroma_client._server = None
                except Exception as e:
                    logger.warning(f"关闭ChromaDB服务器连接失败: {e}")
                
                self.chroma_client = None
            
            # 强制清理临时目录（多次尝试）
            if hasattr(self, 'temp_dir') and self.temp_dir:
                temp_dir_path = self.temp_dir
                for attempt in range(3):  # 尝试3次
                    try:
                        if os.path.exists(temp_dir_path):
                            # 首先尝试修改权限
                            import stat
                            for root, dirs, files in os.walk(temp_dir_path):
                                for d in dirs:
                                    os.chmod(os.path.join(root, d), stat.S_IRWXU)
                                for f in files:
                                    os.chmod(os.path.join(root, f), stat.S_IRWXU)
                            
                            shutil.rmtree(temp_dir_path, ignore_errors=False)
                            logger.info(f"清理ChromaDB临时目录: {temp_dir_path}")
                            break
                    except Exception as e:
                        if attempt == 2:  # 最后一次尝试
                            logger.error(f"清理临时目录失败（尝试{attempt+1}次）: {e}")
                        else:
                            import time
                            time.sleep(0.1)  # 短暂等待后重试
            
            # 清理文档数据
            self.documents = []
            
            logger.info(f"会话 {self.collection_name} 的ChromaDB数据已清理")
            
        except Exception as e:
            logger.error(f"清理ChromaDB数据失败: {e}")
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"析构清理失败: {e}")

class LCIDatabase(ABC):
    """LCI数据库抽象基类"""
    
    @abstractmethod
    def search_flows(self, query: str, context: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """搜索LCI流数据"""
        pass
    
    @abstractmethod
    def get_flow_by_uuid(self, uuid: str) -> Dict[str, Any]:
        """根据UUID获取流数据"""
        pass

class EcoinventLCIDatabase(LCIDatabase):
    """Ecoinvent永久LCI知识库 - 使用连接池管理"""
    
    def __init__(self, mongodb_uri: str = "mongodb://localhost:27017/", db_name: str = "lci_database"):
        # 确保MongoDB连接池已初始化
        if not mongodb_manager.client:
            from .mongodb_manager import initialize_mongodb
            initialize_mongodb(mongodb_uri, db_name)
        
        # 使用连接池获取集合
        self.flows_collection = get_flows_collection()
        logger.info("Ecoinvent数据库已连接到连接池")
    
    def _initialize_connection(self):
        """初始化数据库连接"""
        try:
            from pymongo import MongoClient
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client[self.db_name]
            self.flows_collection = self.db.flows
            
            # 测试连接
            self.client.admin.command('ping')
            logger.info("Ecoinvent MongoDB连接成功")
            
        except Exception as e:
            logger.error(f"Ecoinvent数据库连接失败: {str(e)}")
            raise ConnectionError(f"无法连接到数据库: {str(e)}")
    
    def search_flows(self, query: str, context: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        搜索Ecoinvent流数据
        
        Args:
            query: 搜索查询
            context: PDF上下文信息
            
        Returns:
            List[Dict[str, Any]]: 匹配的LCI流数据
        """
        if not mongodb_manager.client:
            logger.error("数据库连接池不可用")
            return []
        
        try:
            results = []
            
            # 1. 文本搜索
            text_results = list(self.flows_collection.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(5))
            
            results.extend(text_results)
            
            # 2. 类别搜索
            category_results = list(self.flows_collection.find({
                "categories": {"$regex": query, "$options": "i"}
            }).limit(3))
            
            results.extend(category_results)
            
            # 3. 产品搜索  
            product_results = list(self.flows_collection.find({
                "reference_product": {"$regex": query, "$options": "i"}
            }).limit(3))
            
            results.extend(product_results)
            
            # 去重
            unique_results = self._deduplicate_results(results)
            
            logger.info(f"Ecoinvent搜索完成，找到{len(unique_results)}个匹配结果")
            return unique_results[:10]  # 限制返回数量
            
        except Exception as e:
            logger.error(f"搜索Ecoinvent数据失败: {str(e)}")
            return []
    
    def get_flow_by_uuid(self, uuid: str) -> Dict[str, Any]:
        """根据UUID获取流数据"""
        if not mongodb_manager.client:
            logger.error("数据库连接池不可用")
            return {}
        
        try:
            result = self.flows_collection.find_one({"uuid": uuid})
            if result:
                # 移除MongoDB的_id字段
                result.pop('_id', None)
                return result
            else:
                logger.warning(f"未找到UUID为{uuid}的流数据")
                return {}
                
        except Exception as e:
            logger.error(f"获取流数据失败: {str(e)}")
            return {}
    
    def search_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按类别搜索"""
        if not mongodb_manager.client:
            return []
        
        try:
            results = list(self.flows_collection.find({
                "categories": {"$in": [category]}
            }).limit(20))
            
            # 移除_id字段
            for result in results:
                result.pop('_id', None)
            
            return results
            
        except Exception as e:
            logger.error(f"按类别搜索失败: {str(e)}")
            return []
    
    def search_by_location(self, location: str) -> List[Dict[str, Any]]:
        """按地理位置搜索"""
        if not mongodb_manager.client:
            return []
        
        try:
            results = list(self.flows_collection.find({
                "location": location
            }).limit(20))
            
            # 移除_id字段
            for result in results:
                result.pop('_id', None)
            
            return results
            
        except Exception as e:
            logger.error(f"按位置搜索失败: {str(e)}")
            return []
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重结果"""
        seen_uuids = set()
        unique_results = []
        
        for result in results:
            uuid = result.get('uuid')
            if uuid and uuid not in seen_uuids:
                seen_uuids.add(uuid)
                # 移除_id字段
                result.pop('_id', None)
                unique_results.append(result)
        
        return unique_results
    

    


class PermanentLCIDatabase(LCIDatabase):
    """永久LCI知识库的通用接口，支持多种数据库后端"""
    
    def __init__(self, backend: str = "ecoinvent", **kwargs):
        self.backend = backend
        
        if backend == "ecoinvent":
            self.database = EcoinventLCIDatabase(**kwargs)
        else:
            logger.warning(f"未知的数据库后端: {backend}")
            self.database = EcoinventLCIDatabase(**kwargs)
    
    def search_flows(self, query: str, context: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """搜索LCI流数据"""
        return self.database.search_flows(query, context)
    
    def get_flow_by_uuid(self, uuid: str) -> Dict[str, Any]:
        """根据UUID获取流数据"""
        return self.database.get_flow_by_uuid(uuid)
