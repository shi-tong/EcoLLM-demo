#!/usr/bin/env python3
"""
MongoDB连接池配置文件
可根据不同环境和需求调整连接参数
"""

import os
from typing import Dict, Any

class MongoDBConfig:
    """MongoDB配置管理类"""
    
    # 默认连接参数
    DEFAULT_CONFIG = {
        # 基础连接配置
        'mongodb_uri': 'mongodb://localhost:27017/',
        'database_name': 'lci_database',
        
        # 连接池配置
        'maxPoolSize': 50,              # 最大连接数
        'minPoolSize': 5,               # 最小连接数
        'maxIdleTimeMS': 30000,         # 空闲连接超时(30秒)
        'waitQueueTimeoutMS': 5000,     # 等待连接超时(5秒)
        
        # 连接超时配置
        'connectTimeoutMS': 10000,      # 连接超时(10秒)
        'socketTimeoutMS': 20000,       # 套接字超时(20秒)
        'serverSelectionTimeoutMS': 5000,  # 服务器选择超时(5秒)
        
        # 重连配置
        'retryWrites': True,            # 启用写入重试
        'retryReads': True,             # 启用读取重试
        
        # 心跳配置
        'heartbeatFrequencyMS': 10000,  # 心跳频率(10秒)
        
        # 其他优化
        'compressors': 'zstd,zlib,snappy',  # 启用压缩
        'readPreference': 'secondaryPreferred',  # 读取偏好
    }
    
    # 不同环境的配置
    ENVIRONMENT_CONFIGS = {
        'development': {
            'maxPoolSize': 10,
            'minPoolSize': 2,
            'connectTimeoutMS': 5000,
            'socketTimeoutMS': 10000,
        },
        
        'testing': {
            'maxPoolSize': 5,
            'minPoolSize': 1,
            'connectTimeoutMS': 3000,
            'socketTimeoutMS': 5000,
            'retryWrites': False,
        },
        
        'production': {
            'maxPoolSize': 100,
            'minPoolSize': 10,
            'maxIdleTimeMS': 60000,
            'waitQueueTimeoutMS': 10000,
            'connectTimeoutMS': 15000,
            'socketTimeoutMS': 30000,
            'heartbeatFrequencyMS': 5000,
        },
        
        'high_load': {
            'maxPoolSize': 200,
            'minPoolSize': 20,
            'maxIdleTimeMS': 120000,
            'waitQueueTimeoutMS': 15000,
            'connectTimeoutMS': 20000,
            'socketTimeoutMS': 60000,
            'heartbeatFrequencyMS': 3000,
        }
    }
    
    @classmethod
    def get_config(cls, environment: str = None) -> Dict[str, Any]:
        """
        获取指定环境的配置
        
        Args:
            environment: 环境名称 (development, testing, production, high_load)
            
        Returns:
            Dict[str, Any]: 完整配置字典
        """
        # 从环境变量获取环境名称
        if environment is None:
            environment = os.getenv('MONGODB_ENV', 'development')
        
        # 获取基础配置
        config = cls.DEFAULT_CONFIG.copy()
        
        # 从环境变量覆盖配置
        config.update(cls._get_env_config())
        
        # 应用环境特定配置
        if environment in cls.ENVIRONMENT_CONFIGS:
            config.update(cls.ENVIRONMENT_CONFIGS[environment])
        
        return config
    
    @classmethod
    def _get_env_config(cls) -> Dict[str, Any]:
        """从环境变量获取配置"""
        env_config = {}
        
        # 基础配置
        if os.getenv('MONGODB_URI'):
            env_config['mongodb_uri'] = os.getenv('MONGODB_URI')
        
        if os.getenv('MONGODB_DATABASE'):
            env_config['database_name'] = os.getenv('MONGODB_DATABASE')
        
        # 连接池配置
        if os.getenv('MONGODB_MAX_POOL_SIZE'):
            env_config['maxPoolSize'] = int(os.getenv('MONGODB_MAX_POOL_SIZE'))
        
        if os.getenv('MONGODB_MIN_POOL_SIZE'):
            env_config['minPoolSize'] = int(os.getenv('MONGODB_MIN_POOL_SIZE'))
        
        if os.getenv('MONGODB_CONNECT_TIMEOUT'):
            env_config['connectTimeoutMS'] = int(os.getenv('MONGODB_CONNECT_TIMEOUT'))
        
        if os.getenv('MONGODB_SOCKET_TIMEOUT'):
            env_config['socketTimeoutMS'] = int(os.getenv('MONGODB_SOCKET_TIMEOUT'))
        
        return env_config
    
    @classmethod
    def get_connection_params(cls, environment: str = None) -> Dict[str, Any]:
        """
        获取PyMongo连接参数（排除URI和数据库名）
        
        Args:
            environment: 环境名称
            
        Returns:
            Dict[str, Any]: PyMongo连接参数
        """
        config = cls.get_config(environment)
        
        # 移除非PyMongo参数
        connection_params = config.copy()
        connection_params.pop('mongodb_uri', None)
        connection_params.pop('database_name', None)
        
        return connection_params
    
    @classmethod
    def get_uri_and_db(cls, environment: str = None) -> tuple:
        """
        获取URI和数据库名
        
        Args:
            environment: 环境名称
            
        Returns:
            tuple: (mongodb_uri, database_name)
        """
        config = cls.get_config(environment)
        return config['mongodb_uri'], config['database_name']


# 便捷函数
def get_mongodb_config(environment: str = None) -> Dict[str, Any]:
    """获取MongoDB配置的便捷函数"""
    return MongoDBConfig.get_config(environment)


def get_connection_params(environment: str = None) -> Dict[str, Any]:
    """获取连接参数的便捷函数"""
    return MongoDBConfig.get_connection_params(environment)


def get_uri_and_db(environment: str = None) -> tuple:
    """获取URI和数据库名的便捷函数"""
    return MongoDBConfig.get_uri_and_db(environment)
