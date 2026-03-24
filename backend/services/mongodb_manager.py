#!/usr/bin/env python3
"""
MongoDB连接池管理器
提供高效的数据库连接管理和连接池优化
"""

import logging
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from threading import Lock
import os
from contextlib import contextmanager

try:
    from ..config.mongodb_config import get_connection_params, get_uri_and_db
except ImportError:
    # Fallback for direct execution
    def get_connection_params(environment=None):
        return {}
    def get_uri_and_db(environment=None):
        return "mongodb://localhost:27017/", "lci_database"

logger = logging.getLogger(__name__)

class MongoDBManager:
    """MongoDB连接池管理器 - 单例模式"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MongoDBManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化连接管理器"""
        if not hasattr(self, 'initialized'):
            self.client: Optional[MongoClient] = None
            self.db: Optional[Database] = None
            self._connection_params = None
            self.initialized = True
    
    def initialize(self, mongodb_uri: str = None, db_name: str = None, environment: str = None, **kwargs):
        """
        初始化MongoDB连接池
        
        Args:
            mongodb_uri: MongoDB连接URI
            db_name: 数据库名称
            environment: 环境名称 (development, testing, production, high_load)
            **kwargs: 额外的连接参数
        """
        # 使用配置文件获取参数
        if mongodb_uri is None or db_name is None:
            config_uri, config_db = get_uri_and_db(environment)
            mongodb_uri = mongodb_uri or config_uri
            db_name = db_name or config_db
        
        # 获取连接参数
        connection_params = get_connection_params(environment)
        
        # 合并用户参数
        connection_params.update(kwargs)
        self._connection_params = connection_params
        
        try:
            # 创建MongoDB客户端
            self.client = MongoClient(mongodb_uri, **connection_params)
            self.db = self.client[db_name]
            
            # 测试连接
            self.client.admin.command('ping')
            
            # 记录连接池状态
            self._log_connection_status()
            
            logger.info(f"MongoDB连接池初始化成功 - 数据库: {db_name}")
            
        except Exception as e:
            logger.error(f"MongoDB连接池初始化失败: {str(e)}")
            raise ConnectionError(f"Unable to initialize MongoDB connection pool: {str(e)}")
    
    def get_database(self) -> Database:
        """获取数据库实例"""
        if self.db is None:
            raise ConnectionError("MongoDB connection pool not initialized, please call initialize() first")
        return self.db
    
    def get_collection(self, collection_name: str) -> Collection:
        """获取集合实例"""
        if self.db is None:
            raise ConnectionError("MongoDB connection pool not initialized, please call initialize() first")
        return self.db[collection_name]
    
    @contextmanager
    def get_session(self):
        """获取MongoDB会话（用于事务）"""
        if self.client is None:
            raise ConnectionError("MongoDB connection pool not initialized")
        
        session = self.client.start_session()
        try:
            yield session
        finally:
            session.end_session()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 执行ping命令
            ping_result = self.client.admin.command('ping')
            
            # 获取服务器状态
            server_status = self.client.admin.command('serverStatus')
            
            # 获取连接池状态
            pool_stats = self._get_connection_pool_stats()
            
            return {
                "status": "healthy",
                "ping": ping_result,
                "server_info": {
                    "version": server_status.get("version"),
                    "uptime": server_status.get("uptime"),
                    "connections": server_status.get("connections", {}),
                },
                "connection_pool": pool_stats,
                "database": self.db.name if self.db is not None else None
            }
            
        except Exception as e:
            logger.error(f"MongoDB健康检查失败: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _get_connection_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        try:
            if self.client:
                # 获取基本连接信息
                topology = self.client.topology_description
                pool_stats = {
                    "topology_type": str(topology.topology_type),
                    "servers": len(topology.server_descriptions()),
                    "connection_params": self._connection_params or {}
                }
                return pool_stats
        except Exception as e:
            logger.warning(f"获取连接池统计失败: {str(e)}")
            return {"error": str(e)}
    
    def _log_connection_status(self):
        """记录连接状态"""
        try:
            params = self._connection_params
            logger.info("MongoDB连接池配置:")
            logger.info(f"  - 最大连接数: {params.get('maxPoolSize')}")
            logger.info(f"  - 最小连接数: {params.get('minPoolSize')}")
            logger.info(f"  - 连接超时: {params.get('connectTimeoutMS')}ms")
            logger.info(f"  - 空闲超时: {params.get('maxIdleTimeMS')}ms")
            logger.info(f"  - 压缩算法: {params.get('compressors')}")
            
        except Exception as e:
            logger.warning(f"记录连接状态失败: {str(e)}")
    
    def close(self):
        """关闭连接池"""
        if self.client:
            logger.info("正在关闭MongoDB连接池...")
            self.client.close()
            self.client = None
            self.db = None
            logger.info("MongoDB连接池已关闭")
    
    def __del__(self):
        """析构函数"""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except:
            # 忽略关闭时的异常，通常发生在程序退出时
            pass


# 全局连接管理器实例
mongodb_manager = MongoDBManager()


def get_mongodb_manager() -> MongoDBManager:
    """获取MongoDB管理器实例"""
    return mongodb_manager


def initialize_mongodb(mongodb_uri: str = None, db_name: str = None, environment: str = None, **kwargs):
    """初始化MongoDB连接池的便捷函数"""
    mongodb_manager.initialize(mongodb_uri, db_name, environment, **kwargs)


def get_flows_collection() -> Collection:
    """获取flows集合的便捷函数"""
    return mongodb_manager.get_collection("flows")


def get_database() -> Database:
    """获取数据库实例的便捷函数"""
    return mongodb_manager.get_database()
