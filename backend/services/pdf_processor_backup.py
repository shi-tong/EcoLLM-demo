#!/usr/bin/env python3
"""
PDF处理器 - 增强版
包含智能文本提取功能
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import logging
import re
from .lci_extractor import LCIInformationExtractor

logger = logging.getLogger(__name__)

class PDFProcessor:
    """增强版PDF处理器，包含智能文本提取"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.lci_extractor = LCIInformationExtractor(use_nlp=False)
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """处理PDF文件"""
        try:
            # 加载PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # 分块处理
            documents = self.text_splitter.split_documents(pages)
            
            logger.info(f"PDF处理完成，共{len(pages)}页，生成{len(documents)}个文档块")
            return documents
            
        except Exception as e:
            logger.error(f"PDF处理失败: {str(e)}")
            raise e
    
    def extract_lci_information(self, documents: List[Document]) -> dict:
        """从文档中提取LCI相关信息"""
        return self.lci_extractor.extract_lci_information(documents)
    
    def extract_key_information(self, text: str, query: str) -> str:
        """
        智能提取关键信息
        
        Args:
            text: 原始文本
            query: 查询内容
            
        Returns:
            str: 提取的关键信息
        """
        try:
            # 1. 基于查询的关键词提取
            keywords = self._extract_keywords(query)
            
            # 2. 查找包含关键词的句子
            sentences = self._split_into_sentences(text)
            relevant_sentences = []
            
            for sentence in sentences:
                # 检查句子是否包含查询关键词
                if any(keyword.lower() in sentence.lower() for keyword in keywords):
                    relevant_sentences.append(sentence)
            
            # 3. 如果找到相关句子，返回前3个
            if relevant_sentences:
                return " ".join(relevant_sentences[:3])
            
            # 4. 如果没有找到，返回前100个字符的摘要
            return text[:100] + "..." if len(text) > 100 else text
            
        except Exception as e:
            logger.error(f"关键信息提取失败: {str(e)}")
            return text[:100] + "..." if len(text) > 100 else text
    
    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        # 简单的关键词提取
        keywords = []
        
        # 移除常见停用词
        stop_words = {"的", "是", "在", "有", "和", "与", "或", "但", "而", "了", "着", "过"}
        
        # 分词并过滤
        words = re.findall(r'\w+', query)
        for word in words:
            if word not in stop_words and len(word) > 1:
                keywords.append(word)
        
        return keywords
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 简单的句子分割
        sentences = re.split(r'[。！？；\n]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_search_results(self, search_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        处理搜索结果，提取关键信息
        
        Args:
            search_results: 原始搜索结果
            query: 查询内容
            
        Returns:
            List[Dict[str, Any]]: 处理后的结果
        """
        processed_results = []
        
        for result in search_results:
            content = result.get("content", "")
            
            # 提取关键信息
            key_info = self.extract_key_information(content, query)
            
            # 创建处理后的结果
            processed_result = {
                "content": key_info,  # 使用提取的关键信息
                "full_content": content,  # 保留完整内容
                "metadata": result.get("metadata", {}),
                "similarity_score": result.get("similarity_score", 0),
                "extracted_keywords": self._extract_keywords(query)
            }
            
            processed_results.append(processed_result)
        
        return processed_results
