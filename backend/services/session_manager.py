import asyncio
import logging
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import uuid
import os

from .knowledge_base import TemporaryKnowledgeBase

logger = logging.getLogger(__name__)

class SessionManager:
    """会话管理器，负责会话生命周期管理和资源清理"""
    
    def __init__(self, session_timeout: int = 7200, cleanup_interval: int = 300):
        """
        初始化会话管理器
        
        Args:
            session_timeout: 会话超时时间（秒），默认2小时
            cleanup_interval: 清理检查间隔（秒），默认5分钟
        """
        self.session_timeout = session_timeout
        self.cleanup_interval = cleanup_interval
        self.sessions: Dict[str, SessionData] = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "lca_llm_sessions"
        self.temp_dir.mkdir(exist_ok=True)
        
        # 后台清理任务将在首次需要时启动
        self._cleanup_task = None
    
    def start_cleanup_task(self):
        """启动后台清理任务"""
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                logger.info("会话清理任务已启动")
        except RuntimeError:
            # 如果没有运行的事件循环，则稍后启动
            logger.info("会话清理任务将在FastAPI启动时启动")
    
    async def _periodic_cleanup(self):
        """定期清理过期会话"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"清理任务出错: {e}")
    
    def create_session(self, original_filename: str) -> str:
        """
        创建新会话
        
        Args:
            original_filename: 原始文件名
            
        Returns:
            str: 会话ID
        """
        session_id = str(uuid.uuid4())
        session_data = SessionData(
            session_id=session_id,
            original_filename=original_filename,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        self.sessions[session_id] = session_data
        logger.info(f"创建会话: {session_id}, 文件: {original_filename}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional['SessionData']:
        """获取会话数据并更新访问时间"""
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
            session_data.last_accessed = datetime.now()
            return session_data
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        删除指定会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 删除是否成功
        """
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
            
            # 清理会话资源
            self._cleanup_session_resources(session_data)
            
            # 从内存中移除
            del self.sessions[session_id]
            
            logger.info(f"删除会话: {session_id}")
            return True
        
        return False
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            if (current_time - session_data.last_accessed).seconds > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
            logger.info(f"清理过期会话: {session_id}")
        
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
    
    def _cleanup_session_resources(self, session_data: 'SessionData'):
        """清理会话相关资源"""
        try:
            # 1. 清理ChromaDB数据
            if session_data.knowledge_base is not None:
                # 显式调用cleanup方法清理ChromaDB数据和临时目录
                session_data.knowledge_base.cleanup()
                session_data.knowledge_base = None
            
            # 2. 清理临时文件目录
            if session_data.temp_dir and session_data.temp_dir.exists():
                shutil.rmtree(session_data.temp_dir, ignore_errors=True)
                logger.debug(f"清理临时目录: {session_data.temp_dir}")
            
            # 3. 清理其他资源
            session_data.documents = []
            
        except Exception as e:
            logger.error(f"清理会话资源失败: {e}")
    
    def get_session_stats(self) -> Dict:
        """获取会话统计信息"""
        current_time = datetime.now()
        active_sessions = 0
        total_documents = 0
        
        for session_data in self.sessions.values():
            if (current_time - session_data.last_accessed).seconds < self.session_timeout:
                active_sessions += 1
                total_documents += len(session_data.documents)
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_documents": total_documents,
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量（MB）"""
        total_size = 0
        for session_data in self.sessions.values():
            # 估算文档大小
            for doc in session_data.documents:
                total_size += len(doc.page_content.encode('utf-8'))
        
        return total_size / (1024 * 1024)  # 转换为MB

class SessionData:
    """会话数据类"""
    
    def __init__(self, session_id: str, original_filename: str, created_at: datetime, last_accessed: datetime):
        self.session_id = session_id
        self.original_filename = original_filename
        self.created_at = created_at
        self.last_accessed = last_accessed
        
        # 会话相关数据
        self.knowledge_base: Optional[TemporaryKnowledgeBase] = None
        self.documents = []
        self.temp_dir: Optional[Path] = None
        self.metadata = {}
    
    def is_expired(self, timeout_seconds: int) -> bool:
        """检查会话是否过期"""
        return (datetime.now() - self.last_accessed).seconds > timeout_seconds
    
    def get_age(self) -> timedelta:
        """获取会话年龄"""
        return datetime.now() - self.created_at
    
    def get_idle_time(self) -> timedelta:
        """获取会话空闲时间"""
        return datetime.now() - self.last_accessed
