#!/usr/bin/env python3
"""
表格感知文档分块器
实现表格优先的文档分块策略，将表格序列化为LLM友好的格式
"""

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class TableAwareChunker:
    """表格感知的文档分块器"""
    
    def __init__(self, 
                 chunk_size: int = 600, 
                 chunk_overlap: int = 150,
                 table_serialization_format: str = "markdown"):
        """
        初始化表格感知分块器
        
        Args:
            chunk_size: 常规文本块大小
            chunk_overlap: 文本块重叠大小
            table_serialization_format: 表格序列化格式 ("markdown", "keyvalue", "json")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_format = table_serialization_format
        
        # 常规文本分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "。", "; ", "；", " ", ""],
            keep_separator=True
        )
    
    def process_pdf_with_table_awareness(self, file_path: str) -> List[Document]:
        """
        表格感知的PDF处理
        
        Returns:
            List[Document]: 包含表格块和文本块的文档列表
        """
        try:
            with pdfplumber.open(file_path) as pdf:
                all_documents = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # 1. 提取页面表格
                    tables = page.extract_tables()
                    table_regions = self._get_table_regions(page, tables)
                    
                    # 2. 提取非表格文本
                    non_table_text = self._extract_non_table_text(page, table_regions)
                    
                    # 3. 处理表格为独立块
                    page_documents = []
                    
                    if tables:
                        table_documents = self._process_tables_as_chunks(
                            tables, page_num, table_regions
                        )
                        page_documents.extend(table_documents)
                    
                    # 4. 处理常规文本（直接分块，不进行启发式表格检测）
                    if non_table_text.strip():
                        # 🔥 关键修复：已经通过pdfplumber检测过表格，非表格文本直接分块
                        # 不再进行启发式检测，避免误判
                        text_documents = self._process_regular_text(non_table_text, page_num)
                        page_documents.extend(text_documents)
                    
                    # 5. 🎯 按照真实文档顺序排序
                    sorted_page_documents = self._sort_documents_by_position(page_documents, page)
                    all_documents.extend(sorted_page_documents)
                
                logger.info(f"表格感知处理完成: 总共{len(all_documents)}个文档块")
                return all_documents
                
        except Exception as e:
            logger.error(f"表格感知PDF处理失败: {str(e)}")
            raise e
    
    def _get_table_regions(self, page, tables: List[List]) -> List[Dict]:
        """
        获取表格在页面中的区域信息
        
        🔥 关键修复：确保获取到表格的bbox信息
        """
        table_regions = []
        
        try:
            # 使用pdfplumber的find_tables方法获取表格位置
            table_objects = page.find_tables()
            
            if not table_objects:
                logger.warning(f"页面{page.page_number}: find_tables()未返回表格对象")
                raise ValueError("No table objects found")
            
            for idx, table_obj in enumerate(table_objects):
                if idx < len(tables):  # 确保有对应的表格数据
                    bbox = table_obj.bbox
                    if bbox:
                        region = {
                            "bbox": bbox,  # (x0, top, x1, bottom)
                            "table_data": tables[idx],
                            "table_index": idx
                        }
                        table_regions.append(region)
                        logger.debug(f"页面{page.page_number}: 表格{idx} bbox={bbox}")
                    else:
                        logger.warning(f"页面{page.page_number}: 表格{idx}缺少bbox")
                        raise ValueError(f"Table {idx} missing bbox")
                    
        except Exception as e:
            logger.warning(f"页面{page.page_number}: 获取表格bbox失败: {str(e)}，使用文本匹配回退方案")
            # 🔥 回退方案：不过滤表格内容，让启发式检测处理
            # 这样可以避免误删段落
            for idx, table in enumerate(tables):
                table_regions.append({
                    "bbox": None,
                    "table_data": table,
                    "table_index": idx
                })
        
        return table_regions
    
    def _extract_non_table_text(self, page, table_regions: List[Dict]) -> str:
        """
        🔥 关键修复：使用位置信息过滤表格区域，避免误删段落
        
        提取非表格区域的文本，确保表格内容从文本流中完全移除
        """
        try:
            # 如果没有表格，直接返回全文
            if not table_regions:
                full_text = page.extract_text() or ""
                return full_text
            
            # 🔥 改进方案：使用位置信息过滤表格区域
            # 获取页面中所有文本对象及其位置
            words = page.extract_words()
            
            if not words:
                logger.warning(f"页面{page.page_number}: 无法提取文本对象，返回空文本")
                return ""
            
            # 构建表格区域的边界框列表
            table_bboxes = []
            for region in table_regions:
                if 'bbox' in region:
                    table_bboxes.append(region['bbox'])
            
            # 如果没有边界框信息，回退到简单方法
            if not table_bboxes:
                logger.warning(f"页面{page.page_number}: 表格区域缺少bbox信息，使用全文")
                return page.extract_text() or ""
            
            # 过滤掉位于表格区域内的文本
            non_table_words = []
            for word in words:
                word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
                
                # 检查该词是否在任何表格区域内
                is_in_table = False
                for table_bbox in table_bboxes:
                    if self._bbox_overlaps(word_bbox, table_bbox):
                        is_in_table = True
                        break
                
                if not is_in_table:
                    non_table_words.append(word)
            
            # 重建文本（保持原有的空格和换行）
            if not non_table_words:
                return ""
            
            # 按位置排序（从上到下，从左到右）
            non_table_words.sort(key=lambda w: (w['top'], w['x0']))
            
            # 重建文本，保持合理的空格和换行
            # 🔥 改进：使用更智能的换行判断，避免错误断行
            lines = []
            current_line = []
            current_top = non_table_words[0]['top']
            line_height = 12  # 估计的行高（像素）
            
            for word in non_table_words:
                # 如果Y坐标变化超过半个行高，说明换行了
                if abs(word['top'] - current_top) > line_height * 0.6:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word['text']]
                    current_top = word['top']
                else:
                    current_line.append(word['text'])
            
            # 添加最后一行
            if current_line:
                lines.append(' '.join(current_line))
            
            filtered_text = '\n'.join(lines)
            
            logger.info(f"页面{page.page_number}: 基于位置过滤表格，保留{len(non_table_words)}个词，{len(lines)}行")
            return filtered_text
                
        except Exception as e:
            logger.error(f"表格内容过滤失败: {str(e)}")
            # 发生错误时，返回全文以避免丢失内容
            logger.warning("过滤失败，返回全文")
            return page.extract_text() or ""
    
    def _bbox_overlaps(self, bbox1, bbox2, threshold=0.5) -> bool:
        """
        判断两个边界框是否重叠
        
        Args:
            bbox1, bbox2: (x0, y0, x1, y1) 格式的边界框
            threshold: 重叠阈值（0-1），表示重叠面积占bbox1面积的比例
        """
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2
        
        # 计算重叠区域
        x0_overlap = max(x0_1, x0_2)
        y0_overlap = max(y0_1, y0_2)
        x1_overlap = min(x1_1, x1_2)
        y1_overlap = min(y1_1, y1_2)
        
        # 如果没有重叠
        if x0_overlap >= x1_overlap or y0_overlap >= y1_overlap:
            return False
        
        # 计算重叠面积
        overlap_area = (x1_overlap - x0_overlap) * (y1_overlap - y0_overlap)
        bbox1_area = (x1_1 - x0_1) * (y1_1 - y0_1)
        
        if bbox1_area == 0:
            return False
        
        overlap_ratio = overlap_area / bbox1_area
        return overlap_ratio >= threshold
    
    def _process_tables_as_chunks(self, tables: List[List], page_num: int, 
                                  table_regions: List[Dict]) -> List[Document]:
        """将表格处理为独立的文档块"""
        table_documents = []
        
        for table_idx, table_data in enumerate(tables):
            if not table_data or len(table_data) < 2:
                continue
            
            # 清理表格数据
            cleaned_table = self._clean_table_data(table_data)
            if not cleaned_table:
                continue
            
            headers = cleaned_table[0]
            data_rows = cleaned_table[1:]
            
            # 根据配置选择序列化格式
            if self.table_format == "markdown":
                table_chunks = self._serialize_table_as_markdown(
                    headers, data_rows, page_num, table_idx
                )
            elif self.table_format == "keyvalue":
                table_chunks = self._serialize_table_as_keyvalue(
                    headers, data_rows, page_num, table_idx
                )
            else:  # json格式
                table_chunks = self._serialize_table_as_json(
                    headers, data_rows, page_num, table_idx
                )
            
            table_documents.extend(table_chunks)
        
        return table_documents
    
    def _serialize_table_as_markdown(self, headers: List[str], data_rows: List[List],
                                     page_num: int, table_idx: int) -> List[Document]:
        """
        将表格序列化为Markdown格式的文档块
        
        采用专家建议的"表头重复"子分块策略：
        - 统一使用table_section类型
        - 每个块都包含完整表头
        - 大表格按行数分段，小表格保持完整
        """
        documents = []
        
        # 创建表格头部信息块
        table_header = f"## Table {page_num}.{table_idx + 1}\n\n"
        table_header += "| " + " | ".join(headers) + " |\n"
        table_header += "|" + "---|" * len(headers) + "\n"
        
        # 计算表头大小（用于chunk大小控制）
        header_size = len(table_header)
        
        # 估算每行大小
        avg_row_size = 0
        if data_rows:
            sample_row = "| " + " | ".join(str(cell) for cell in data_rows[0]) + " |\n"
            avg_row_size = len(sample_row)
        
        # 计算每个chunk能容纳的最大行数
        available_size = self.chunk_size - header_size - 100  # 预留100字符缓冲
        max_rows_per_chunk = max(1, available_size // avg_row_size) if avg_row_size > 0 else 10
        
        # 如果整个表格能放入一个chunk，则创建单个table_section
        total_content_size = header_size + len(data_rows) * avg_row_size
        if total_content_size <= self.chunk_size:
            # 小表格：单个table_section
            full_content = table_header
            for row in data_rows:
                # 处理单元格内的换行符，保持markdown表格格式
                cleaned_cells = [str(cell).replace('\n', ' ').replace('\r', ' ') if cell else '' for cell in row]
                row_content = "| " + " | ".join(cleaned_cells) + " |"
                full_content += row_content + "\n"
            
            doc = Document(
                page_content=full_content,
                metadata={
                    "page": page_num,
                    "chunk_type": "table_section",
                    "table_id": f"page_{page_num}_table_{table_idx}",
                    "table_headers": headers,
                    "section_info": "1_of_1",
                    "row_count": len(data_rows),
                    "is_complete_table": True
                }
            )
            documents.append(doc)
        else:
            # 大表格：分段处理，每段都包含表头
            total_sections = (len(data_rows) + max_rows_per_chunk - 1) // max_rows_per_chunk
            
            for section_idx in range(total_sections):
                start_row = section_idx * max_rows_per_chunk
                end_row = min(start_row + max_rows_per_chunk, len(data_rows))
                
                # 构建这一段的内容（表头 + 对应行）
                section_content = table_header
                for row_idx in range(start_row, end_row):
                    row = data_rows[row_idx]
                    # 处理单元格内的换行符，保持markdown表格格式
                    cleaned_cells = [str(cell).replace('\n', ' ').replace('\r', ' ') if cell else '' for cell in row]
                    row_content = "| " + " | ".join(cleaned_cells) + " |"
                    section_content += row_content + "\n"
                
                doc = Document(
                    page_content=section_content,
                    metadata={
                        "page": page_num,
                        "chunk_type": "table_section",
                        "table_id": f"page_{page_num}_table_{table_idx}",
                        "table_headers": headers,
                        "section_info": f"{section_idx + 1}_of_{total_sections}",
                        "row_range": f"{start_row + 1}-{end_row}",
                        "total_rows": len(data_rows),
                        "is_complete_table": False
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _sort_documents_by_position(self, documents: List[Document], page) -> List[Document]:
        """
        🎯 按照文档中的真实位置排序chunks
        
        Args:
            documents: 页面中的所有文档块
            page: pdfplumber页面对象
            
        Returns:
            List[Document]: 按位置排序后的文档块
        """
        if not documents:
            return documents
        
        # 为每个文档添加位置信息
        documents_with_position = []
        
        for doc in documents:
            metadata = doc.metadata
            position_y = 0  # 默认位置
            
            # 获取表格的位置信息
            if metadata.get('chunk_type', '').startswith('table'):
                table_idx = metadata.get('table_index', 0)
                try:
                    tables = page.extract_tables()
                    if table_idx < len(tables) and tables[table_idx]:
                        # 尝试获取表格的边界框
                        table_bbox = self._get_table_bbox(page, table_idx)
                        if table_bbox:
                            position_y = table_bbox[1]  # 使用top坐标
                        else:
                            # 回退：使用表格在页面中的估计位置
                            position_y = page.height * 0.3 * (table_idx + 1)
                except Exception as e:
                    logger.debug(f"获取表格位置失败: {str(e)}")
                    position_y = 100 * (table_idx + 1)  # 简单估计
            
            # 对于文本块，尝试通过内容匹配获取位置
            else:
                try:
                    # 使用文档内容的前50个字符在页面中查找位置
                    content_preview = doc.page_content[:50].strip()
                    if content_preview:
                        # 在页面文本中查找这段内容的位置
                        page_text = page.extract_text() or ""
                        if content_preview in page_text:
                            # 简单估计：根据在文本中的相对位置估算Y坐标
                            text_position = page_text.find(content_preview)
                            total_text_length = len(page_text)
                            if total_text_length > 0:
                                relative_position = text_position / total_text_length
                                position_y = page.height * (1 - relative_position)  # Y坐标从上到下递减
                            else:
                                position_y = page.height * 0.5  # 默认中间位置
                        else:
                            position_y = page.height * 0.8  # 默认靠下位置
                    else:
                        position_y = page.height * 0.9  # 空内容放在最后
                except Exception as e:
                    logger.debug(f"获取文本位置失败: {str(e)}")
                    position_y = page.height * 0.7  # 默认位置
            
            documents_with_position.append((position_y, doc))
        
        # 按Y坐标排序（从上到下，Y坐标递减）
        documents_with_position.sort(key=lambda x: -x[0])  # 负号表示从大到小（从上到下）
        
        # 返回排序后的文档列表
        sorted_documents = [doc for _, doc in documents_with_position]
        
        logger.debug(f"页面{page.page_number}: 按位置排序了{len(sorted_documents)}个文档块")
        return sorted_documents
    
    def _get_table_bbox(self, page, table_idx: int):
        """获取表格的边界框"""
        try:
            # 尝试从表格对象获取边界框
            tables = page.extract_tables()
            if table_idx < len(tables):
                # pdfplumber的表格通常没有直接的bbox，需要通过其他方式估算
                # 这里使用简单的估算方法
                table_height_estimate = page.height / (len(tables) + 1)
                top_y = page.height - (table_idx + 1) * table_height_estimate
                return (0, top_y, page.width, top_y + table_height_estimate)
        except Exception as e:
            logger.debug(f"获取表格边界框失败: {str(e)}")
        return None
    
    def _process_text_with_heuristic_table_detection(self, text: str, page_num: int) -> List[Document]:
        """
        🔥 关键改进：启发式检测文本中的表格
        
        当pdfplumber无法检测到表格时，使用文本模式识别
        """
        # 检测文本中是否包含表格模式
        if self._is_text_likely_table(text):
            logger.info(f"页面{page_num}: 启发式检测到表格模式，使用表格处理")
            return self._process_text_as_table(text, page_num)
        else:
            # 使用常规文本处理
            return self._process_regular_text(text, page_num)
    
    def _is_text_likely_table(self, text: str) -> bool:
        """
        启发式判断文本是否可能是表格
        
        基于专家建议中提到的表格特征
        """
        lines = text.strip().split('\n')
        if len(lines) < 3:  # 表格至少需要3行（标题+分隔+数据）
            return False
        
        # 检测表格指示符
        table_score = 0
        
        # 1. 表格标题检测
        if re.search(r'(?:Table|表格|表)\s*\d+', text, re.IGNORECASE):
            table_score += 3
        
        # 2. 竖线分隔符检测
        pipe_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
        if pipe_lines >= len(lines) * 0.3:  # 30%以上的行包含竖线
            table_score += 2
        
        # 3. 数值密度检测
        numeric_lines = 0
        for line in lines:
            numbers = re.findall(r'\d+(?:\.\d+)?', line)
            if len(numbers) >= 2:
                numeric_lines += 1
        
        if numeric_lines >= len(lines) * 0.4:  # 40%以上的行包含多个数字
            table_score += 2
        
        # 4. 参数-值对模式检测
        param_value_lines = 0
        for line in lines:
            if re.search(r'\w+\s*[:\|]\s*\d+', line):
                param_value_lines += 1
        
        if param_value_lines >= 3:
            table_score += 1
        
        # 5. 表格分隔符检测
        separator_lines = sum(1 for line in lines if re.match(r'^[\s\-\|=]+$', line.strip()))
        if separator_lines >= 1:
            table_score += 1
        
        logger.debug(f"启发式表格检测得分: {table_score}")
        return table_score >= 3  # 阈值：3分认为是表格
    
    def _process_text_as_table(self, text: str, page_num: int) -> List[Document]:
        """
        将检测到的表格文本按照专家建议的table_section格式处理
        """
        # 解析表格内容
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        # 查找表格标题
        table_title = "## Table (Detected)"
        for line in lines:
            if re.search(r'(?:Table|表格|表)\s*\d+', line, re.IGNORECASE):
                table_title = f"## {line.strip()}"
                break
        
        # 构建表格内容（使用专家建议的格式）
        table_content = f"{table_title}\n\n"
        
        # 如果有竖线分隔的内容，保持原格式
        if any('|' in line for line in lines):
            table_content += '\n'.join(lines)
        else:
            # 转换为简单的表格格式
            table_content += "| Parameter | Value | Unit | Description |\n"
            table_content += "|-----------|-------|------|-------------|\n"
            
            for line in lines:
                if line and not re.search(r'(?:Table|表格|表)\s*\d+', line, re.IGNORECASE):
                    # 尝试解析参数-值对
                    match = re.search(r'(.+?)\s*[:\|]\s*(.+)', line)
                    if match:
                        param, value = match.groups()
                        table_content += f"| {param.strip()} | {value.strip()} | | |\n"
                    else:
                        table_content += f"| {line} | | | |\n"
        
        # 创建table_section格式的文档
        doc = Document(
            page_content=table_content,
            metadata={
                "page": page_num,
                "chunk_type": "table_section",  # 🔥 使用专家建议的类型
                "section_info": "1_of_1",
                "is_complete_table": True,
                "detection_method": "heuristic",  # 标记检测方法
                "original_format": "text_table"
            }
        )
        
        logger.info(f"启发式表格处理: 创建table_section块")
        return [doc]
    
    def _serialize_table_as_keyvalue(self, headers: List[str], data_rows: List[List],
                                     page_num: int, table_idx: int) -> List[Document]:
        """将表格序列化为键值对格式"""
        documents = []
        
        for row_idx, row in enumerate(data_rows):
            # 创建键值对字符串
            kv_pairs = []
            for i, (header, value) in enumerate(zip(headers, row)):
                if value and str(value).strip():
                    kv_pairs.append(f"{header}: {value}")
            
            if not kv_pairs:
                continue
            
            content = f"[Table {page_num}.{table_idx + 1} - Row {row_idx + 1}]\n"
            content += "\n".join(kv_pairs)
            
            doc = Document(
                page_content=content,
                metadata={
                    "page": page_num,
                    "chunk_type": "table_keyvalue",
                    "table_id": f"page_{page_num}_table_{table_idx}",
                    "row_index": row_idx,
                    "table_headers": headers
                }
            )
            documents.append(doc)
        
        return documents
    
    
    def _clean_table_data(self, raw_table: List[List]) -> List[List]:
        """清理表格数据"""
        cleaned_table = []
        
        for row in raw_table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    cleaned_cell = str(cell).strip()
                    cleaned_row.append(cleaned_cell)
            
            # 只保留非空行
            if any(cell for cell in cleaned_row):
                cleaned_table.append(cleaned_row)
        
        return cleaned_table if len(cleaned_table) >= 2 else []
    
    def _process_regular_text(self, text: str, page_num: int) -> List[Document]:
        """处理常规文本"""
        if not text.strip():
            return []
        
        # 创建临时文档进行分块
        temp_doc = Document(
            page_content=text,
            metadata={"page": page_num}
        )
        
        # 使用标准分块器
        text_chunks = self.text_splitter.split_documents([temp_doc])
        
        # 更新元数据
        for chunk in text_chunks:
            chunk.metadata.update({
                "chunk_type": "text",
                "page": page_num
            })
        
        return text_chunks


# 向后兼容的集成类
class EnhancedTableAwarePDFProcessor:
    """集成表格感知功能的PDF处理器"""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 150):
        self.chunker = TableAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            table_serialization_format="markdown"
        )
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """处理PDF文件，返回表格感知的文档块"""
        return self.chunker.process_pdf_with_table_awareness(file_path)
    
    def process_pdf_and_get_full_text(self, file_path: str, session_id: str = None) -> Tuple[str, List[Document]]:
        """处理PDF并返回完整文本和文档片段"""
        documents = self.process_pdf(file_path)
        
        # 为文档分配稳定的chunk ID
        documents = self._assign_stable_chunk_ids(documents, file_path, session_id)
        
        full_text = "\n\n".join([doc.page_content for doc in documents])
        return full_text, documents
    
    def _assign_stable_chunk_ids(self, documents: List[Document], file_path: str, session_id: str = None) -> List[Document]:
        """为文档块分配稳定的全局chunk ID"""
        import logging
        import hashlib
        import os
        
        logger = logging.getLogger(__name__)
        logger.info(f"分配chunk ID: session_id={session_id}, 文档数={len(documents)}")
        
        for i, doc in enumerate(documents):
            # 新格式：纯数字（与 LLM 推理时格式一致）
            if session_id:
                chunk_id = str(i)
                logger.info(f"使用纯数字格式: {chunk_id}")
            else:
                # 向后兼容的fallback格式
                filename = os.path.basename(file_path)
                content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]
                chunk_id = f"{filename}_{i:03d}_{content_hash}"
                logger.info(f"使用fallback格式: {chunk_id}")
            
            # 将chunk_id添加到metadata中
            if not hasattr(doc, 'metadata') or doc.metadata is None:
                doc.metadata = {}
            doc.metadata['chunk_id'] = chunk_id
            doc.metadata['chunk_index'] = i
            doc.metadata['source_file'] = os.path.basename(file_path) if session_id else file_path
            
        return documents
    
    def extract_lci_information(self, documents: List[Document]) -> dict:
        """
        提取LCI信息（兼容性方法）
        
        注意：在1.0阶段，我们不再自动提取LCI数据，而是通过工具调用记录。
        此方法保留是为了向后兼容，返回空结构。
        
        Args:
            documents: 文档列表
            
        Returns:
            dict: 空的LCI信息结构
        """
        return {
            "flows": [],
            "parameters": [],
            "processes": [],
            "metadata": {
                "extraction_method": "manual_tool_calling",
                "note": "LCI data extraction is now handled through tool calling (record_parameter, record_calculation, record_process_flow)"
            }
        }
