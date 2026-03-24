#!/usr/bin/env python3
"""
增强版PDF处理器 - 集成pdfplumber支持表格提取
"""

import pdfplumber
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any, Optional
import logging
import re
from .lci_extractor import LCIInformationExtractor

logger = logging.getLogger(__name__)

class EnhancedPDFProcessor:
    """增强版PDF处理器，支持文本+表格提取"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.lci_extractor = LCIInformationExtractor(use_nlp=False)
        
        # 尝试导入pdfplumber，如果失败则降级到基础功能
        try:
            import pdfplumber
            self.pdfplumber_available = True
            logger.info("pdfplumber可用，启用增强表格提取功能")
        except ImportError:
            self.pdfplumber_available = False
            logger.warning("pdfplumber不可用，使用基础PDF处理功能")
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        处理PDF文件，支持文本和表格提取
        
        Returns:
            List[Document]: 增强的文档列表，包含表格信息
        """
        try:
            if self.pdfplumber_available:
                return self._process_pdf_enhanced(file_path)
            else:
                return self._process_pdf_basic(file_path)
                
        except Exception as e:
            logger.error(f"PDF处理失败: {str(e)}")
            # 如果增强处理失败，降级到基础处理
            if self.pdfplumber_available:
                logger.info("尝试降级到基础PDF处理")
                return self._process_pdf_basic(file_path)
            raise e
    
    def _process_pdf_basic(self, file_path: str) -> List[Document]:
        """基础PDF处理（使用PyPDFLoader）"""
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        documents = self.text_splitter.split_documents(pages)
        
        logger.info(f"基础PDF处理完成，共{len(pages)}页，生成{len(documents)}个文档块")
        return documents
    
    def _process_pdf_enhanced(self, file_path: str) -> List[Document]:
        """增强PDF处理（使用pdfplumber + PyPDFLoader）"""
        # 1. 使用PyPDFLoader获取基础文档
        base_documents = self._process_pdf_basic(file_path)
        
        # 2. 使用pdfplumber提取表格和增强信息
        enhanced_info = self._extract_enhanced_info_with_pdfplumber(file_path)
        
        # 3. 合并信息
        enhanced_documents = self._merge_document_info(base_documents, enhanced_info)
        
        logger.info(f"增强PDF处理完成，提取到{enhanced_info['total_tables']}个表格")
        return enhanced_documents
    
    def _extract_enhanced_info_with_pdfplumber(self, file_path: str) -> Dict:
        """使用pdfplumber提取增强信息"""
        tables_data = []
        page_summaries = {}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"开始pdfplumber增强处理，共{len(pdf.pages)}页")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_info = {
                        "tables": [],
                        "structured_data": [],
                        "lca_indicators": []
                    }
                    
                    # 提取表格
                    tables = page.extract_tables()
                    if tables:
                        for table_idx, table in enumerate(tables):
                            processed_table = self._process_table_for_lca(
                                table, page_num, table_idx
                            )
                            if processed_table:
                                tables_data.append(processed_table)
                                page_info["tables"].append(processed_table)
                    
                    # 提取结构化数值数据
                    structured_data = self._extract_numerical_data(page)
                    if structured_data:
                        page_info["structured_data"] = structured_data
                    
                    # 识别LCA相关指标
                    lca_indicators = self._identify_lca_indicators_on_page(page)
                    if lca_indicators:
                        page_info["lca_indicators"] = lca_indicators
                    
                    page_summaries[page_num] = page_info
                
                return {
                    "tables": tables_data,
                    "page_summaries": page_summaries,
                    "total_tables": len(tables_data),
                    "total_pages": len(pdf.pages)
                }
                
        except Exception as e:
            logger.error(f"pdfplumber增强处理失败: {str(e)}")
            return {"tables": [], "page_summaries": {}, "total_tables": 0, "total_pages": 0}
    
    def _process_table_for_lca(self, raw_table: List[List], page_num: int, table_idx: int) -> Optional[Dict]:
        """处理表格，专门针对LCA数据优化"""
        try:
            if not raw_table or len(raw_table) < 2:
                return None
            
            # 清理表格数据
            cleaned_table = []
            for row in raw_table:
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                if any(cleaned_row):
                    cleaned_table.append(cleaned_row)
            
            if len(cleaned_table) < 2:
                return None
            
            # 识别LCA相关数据模式
            lca_data = self._extract_lca_data_from_table(cleaned_table)
            
            return {
                "table_id": f"page_{page_num}_table_{table_idx}",
                "page_number": page_num,
                "headers": cleaned_table[0],
                "data_rows": cleaned_table[1:],
                "lca_data": lca_data,
                "summary": self._generate_table_summary(cleaned_table, lca_data)
            }
            
        except Exception as e:
            logger.error(f"表格处理失败: {str(e)}")
            return None
    
    def _extract_lca_data_from_table(self, table_data: List[List]) -> Dict:
        """从表格中提取LCA相关数据"""
        lca_data = {
            "energy_consumption": [],
            "material_flows": [],
            "emissions": [],
            "process_parameters": []
        }
        
        if not table_data:
            return lca_data
        
        headers = [str(h).lower() for h in table_data[0]]
        
        # LCA数据识别模式
        patterns = {
            "energy_consumption": [
                r"电[力耗]|energy|electricity|kwh|mj|功率|power|瓦特|watt",
                r"耗电|用电|电量|能源|能耗|消耗"
            ],
            "material_flows": [
                r"材料|material|substance|kg|ton|gram|质量|重量",
                r"原料|物料|成分|组分|投入|input"
            ],
            "emissions": [
                r"排放|emission|co2|ghg|carbon|废气|污染",
                r"温室气体|碳排放|二氧化碳|甲烷|氧化亚氮"
            ],
            "process_parameters": [
                r"工艺|process|temperature|pressure|时间|温度|压力",
                r"参数|条件|设定|操作|运行"
            ]
        }
        
        # 分析每一行数据
        for row_idx, row in enumerate(table_data[1:], 1):
            for col_idx, cell in enumerate(row):
                if not cell:
                    continue
                
                cell_str = str(cell).lower()
                header = headers[col_idx] if col_idx < len(headers) else ""
                
                # 尝试提取数值和单位
                numerical_matches = re.findall(r'(\d+\.?\d*)\s*([a-zA-Z/]+)', cell_str)
                
                for category, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        if re.search(pattern, cell_str, re.IGNORECASE) or re.search(pattern, header, re.IGNORECASE):
                            data_entry = {
                                "row": row_idx,
                                "col": col_idx,
                                "header": headers[col_idx] if col_idx < len(headers) else "",
                                "value": cell,
                                "category": category
                            }
                            
                            # 如果找到数值，添加结构化信息
                            if numerical_matches:
                                for value, unit in numerical_matches:
                                    data_entry.update({
                                        "numerical_value": float(value),
                                        "unit": unit,
                                        "formatted": f"{value} {unit}"
                                    })
                            
                            lca_data[category].append(data_entry)
                            break
        
        return lca_data
    
    def _generate_table_summary(self, table_data: List[List], lca_data: Dict) -> str:
        """生成表格摘要"""
        if not table_data:
            return "空表格"
        
        rows = len(table_data) - 1
        cols = len(table_data[0]) if table_data else 0
        
        summary_parts = [f"表格包含{rows}行{cols}列数据"]
        
        # 添加LCA数据摘要
        lca_summary = []
        for category, data_list in lca_data.items():
            if data_list:
                category_names = {
                    "energy_consumption": "能源消耗",
                    "material_flows": "材料流",
                    "emissions": "排放",
                    "process_parameters": "工艺参数"
                }
                lca_summary.append(f"{category_names.get(category, category)}: {len(data_list)}项")
        
        if lca_summary:
            summary_parts.append("LCA数据: " + ", ".join(lca_summary))
        
        return " | ".join(summary_parts)
    
    def _extract_numerical_data(self, page) -> List[Dict]:
        """从页面提取数值数据"""
        structured_data = []
        
        try:
            # 提取页面文本
            text = page.extract_text()
            if not text:
                return structured_data
            
            # 查找数值模式: 数字 + 单位
            numerical_patterns = [
                r'(\d+\.?\d*)\s*(kWh/kg|MJ/kg|kg/m²|g/片|kg|ton|kWh|MJ|℃|°C|MPa|bar)',
                r'(\d+\.?\d*)\s*(kwh|mj|kg|g|ton|celsius|pascal|watt|w)',
                r'(\d+\.?\d*)\s*%',  # 百分比
                r'(\d+\.?\d*)\s*(小时|hour|h|分钟|min|秒|s)'  # 时间
            ]
            
            for pattern in numerical_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(1)
                    unit = match.group(2) if len(match.groups()) > 1 else "%"
                    
                    # 获取上下文（前后20个字符）
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end].strip()
                    
                    structured_data.append({
                        "value": float(value),
                        "unit": unit,
                        "formatted": f"{value} {unit}",
                        "context": context,
                        "position": {"start": match.start(), "end": match.end()}
                    })
            
        except Exception as e:
            logger.error(f"数值数据提取失败: {str(e)}")
        
        return structured_data
    
    def _identify_lca_indicators_on_page(self, page) -> List[str]:
        """识别页面上的LCA指标"""
        indicators = []
        
        try:
            text = page.extract_text()
            if not text:
                return indicators
            
            text_lower = text.lower()
            
            # LCA常见指标模式
            lca_indicators = {
                "GWP": [r"global warming potential|gwp|全球变暖潜值|温室气体"],
                "CED": [r"cumulative energy demand|ced|累积能源需求|能源消耗"],
                "AP": [r"acidification potential|ap|酸化潜值"],
                "EP": [r"eutrophication potential|ep|富营养化潜值"],
                "POCP": [r"photochemical ozone creation potential|pocp|光化学臭氧"],
                "能耗": [r"energy consumption|能源消耗|电力消耗|燃料消耗"],
                "碳足迹": [r"carbon footprint|碳足迹|碳排放|co2.*排放"],
                "生命周期": [r"life cycle|lifecycle|生命周期|全生命周期"]
            }
            
            for indicator, patterns in lca_indicators.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        if indicator not in indicators:
                            indicators.append(indicator)
                        break
            
        except Exception as e:
            logger.error(f"LCA指标识别失败: {str(e)}")
        
        return indicators
    
    def _merge_document_info(self, base_documents: List[Document], enhanced_info: Dict) -> List[Document]:
        """合并基础文档和增强信息"""
        enhanced_documents = []
        page_summaries = enhanced_info.get("page_summaries", {})
        
        for doc in base_documents:
            page_num = doc.metadata.get('page', 1)
            
            # 获取该页面的增强信息
            page_info = page_summaries.get(page_num, {})
            
            # 增强元数据
            enhanced_metadata = doc.metadata.copy()
            enhanced_metadata.update({
                "enhanced_processing": True,
                "tables_count": len(page_info.get("tables", [])),
                "lca_indicators": page_info.get("lca_indicators", []),
                "structured_data_count": len(page_info.get("structured_data", [])),
                "pdfplumber_processed": True
            })
            
            # 增强内容
            content = doc.page_content
            
            # 添加表格摘要
            if page_info.get("tables"):
                table_summaries = []
                for table in page_info["tables"]:
                    summary = f"\n[表格 {table['table_id']}]: {table['summary']}"
                    
                    # 添加关键LCA数据
                    if table.get("lca_data"):
                        lca_highlights = []
                        for category, data_list in table["lca_data"].items():
                            if data_list:
                                for item in data_list[:2]:  # 只显示前2项
                                    if "formatted" in item:
                                        lca_highlights.append(item["formatted"])
                        if lca_highlights:
                            summary += f" | 关键数据: {', '.join(lca_highlights)}"
                    
                    table_summaries.append(summary)
                
                content += "\n".join(table_summaries)
            
            # 添加结构化数据摘要
            if page_info.get("structured_data"):
                structured_summaries = []
                for data in page_info["structured_data"][:3]:  # 只显示前3项
                    structured_summaries.append(data["formatted"])
                
                if structured_summaries:
                    content += f"\n[结构化数据]: {', '.join(structured_summaries)}"
            
            enhanced_doc = Document(
                page_content=content,
                metadata=enhanced_metadata
            )
            enhanced_documents.append(enhanced_doc)
        
        return enhanced_documents
    
    # 保持与原始PDFProcessor兼容的方法
    def extract_lci_information(self, documents: List[Document]) -> dict:
        """从文档中提取LCI相关信息"""
        return self.lci_extractor.extract_lci_information(documents)
    
    def extract_key_information(self, text: str, query: str) -> str:
        """智能提取关键信息"""
        try:
            keywords = self._extract_keywords(query)
            sentences = self._split_into_sentences(text)
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword.lower() in sentence.lower() for keyword in keywords):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                return " ".join(relevant_sentences[:3])
            
            return text[:100] + "..." if len(text) > 100 else text
            
        except Exception as e:
            logger.error(f"关键信息提取失败: {str(e)}")
            return text[:100] + "..." if len(text) > 100 else text
    
    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词"""
        keywords = []
        stop_words = {"的", "是", "在", "有", "和", "与", "或", "但", "而", "了", "着", "过"}
        
        words = re.findall(r'\w+', query)
        for word in words:
            if word not in stop_words and len(word) > 1:
                keywords.append(word)
        
        return keywords
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        sentences = re.split(r'[。！？；\n]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_search_results(self, search_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """处理搜索结果，提取关键信息"""
        processed_results = []
        
        for result in search_results:
            content = result.get("content", "")
            key_info = self.extract_key_information(content, query)
            
            processed_result = {
                "content": key_info,
                "full_content": content,
                "metadata": result.get("metadata", {}),
                "similarity_score": result.get("similarity_score", 0),
                "extracted_keywords": self._extract_keywords(query)
            }
            
            processed_results.append(processed_result)
        
        return processed_results


# 为了向后兼容，创建一个包装类
class PDFProcessor(EnhancedPDFProcessor):
    """向后兼容的PDF处理器，自动使用增强功能"""
    pass
