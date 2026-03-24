from typing import Dict, Any, List
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

logger = logging.getLogger(__name__)

class CoderLLMService:
    """Coder-LLM推理服务，生成pyLCA代码"""
    
    def __init__(self, model_name: str = "microsoft/CodeT5-large", max_length: int = 2048):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型（实际部署中应该使用微调后的模型）
        # self._load_model()
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info(f"成功加载模型: {self.model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise e
    
    def generate_pylca_code(self, context: Dict[str, Any]) -> str:
        """
        根据上下文生成pyLCA代码
        
        Args:
            context: 包含PDF上下文、LCI数据、用户指令等信息
            
        Returns:
            str: 生成的pyLCA代码
        """
        try:
            # 构建提示词
            prompt = self._build_prompt(context)
            
            # 目前使用规则生成，实际部署中应该使用微调的LLM
            if self.model is None:
                return self._generate_code_with_template(context)
            else:
                return self._generate_code_with_llm(prompt)
                
        except Exception as e:
            logger.error(f"生成pyLCA代码失败: {str(e)}")
            raise e
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """构建LLM提示词"""
        pdf_context = context.get("pdf_context", [])
        lci_data = context.get("lci_data", [])
        instruction = context.get("instruction", "")
        
        # 提取关键信息
        pdf_summary = self._summarize_pdf_context(pdf_context)
        lci_summary = self._summarize_lci_data(lci_data)
        
        prompt = f"""
You are a professional LCA (Life Cycle Assessment) code generation assistant. Please generate pyLCA code based on the following information:

User instruction: {instruction}

PDF document context:
{pdf_summary}

Available LCI flow data:
{lci_summary}

Please generate complete pyLCA code including:
1. Import necessary libraries
2. Connect to openLCA database
3. Create or retrieve relevant flows and processes
4. Set up input-output relationships
5. Perform calculations
6. Output results

The code should be executable Python code:

```python
"""
        return prompt
    
    def _summarize_pdf_context(self, pdf_context: List[Dict[str, Any]]) -> str:
        """总结PDF上下文"""
        if not pdf_context:
            return "No PDF context information available"
        
        summary_parts = []
        for i, ctx in enumerate(pdf_context[:3]):  # 只取前3个最相关的
            content = ctx.get("content", "")[:200]  # 截取前200个字符
            summary_parts.append(f"Section {i+1}: {content}...")
        
        return "\n".join(summary_parts)
    
    def _summarize_lci_data(self, lci_data: List[Dict[str, Any]]) -> str:
        """总结LCI数据"""
        if not lci_data:
            return "No available LCI flow data"
        
        summary_parts = []
        for flow in lci_data:
            summary_parts.append(
                f"- {flow.get('name', '')}, UUID: {flow.get('uuid', '')}, "
                f"Unit: {flow.get('unit', '')}, Location: {flow.get('location', '')}"
            )
        
        return "\n".join(summary_parts)
    
    def _generate_code_with_template(self, context: Dict[str, Any]) -> str:
        """使用模板生成代码（临时方案）"""
        instruction = context.get("instruction", "")
        lci_data = context.get("lci_data", [])
        
        # 简单的模板生成
        code_template = f'''
import olca
import uuid
from decimal import Decimal

# 连接到openLCA数据库
client = olca.Client()

def create_lca_model():
    """根据用户指令创建LCA模型: {instruction}"""
    
    # 创建新的产品系统
    product_system = olca.ProductSystem()
    product_system.id = str(uuid.uuid4())
    product_system.name = "用户定制LCA模型"
    product_system.description = "{instruction}"
    
    # 添加参考流程'''
        
        # 根据LCI数据添加流
        if lci_data:
            for flow in lci_data:
                flow_uuid = flow.get('uuid', '')
                flow_name = flow.get('name', '')
                code_template += f'''
    
    # 添加流: {flow_name}
    flow_{flow_uuid.replace('-', '_')} = client.get(olca.Flow, "{flow_uuid}")
    if flow_{flow_uuid.replace('-', '_')}:
        print(f"成功获取流: {flow_name}")
    else:
        print(f"未找到流: {flow_name}")'''
        
        code_template += '''
    
    # 创建计算设置
    calculation_setup = olca.CalculationSetup()
    calculation_setup.calculation_type = olca.CalculationType.CONTRIBUTION_ANALYSIS
    calculation_setup.product_system = product_system
    
    # 执行计算
    result = client.calculate(calculation_setup)
    
    # 输出结果
    print("LCA计算完成")
    print(f"结果ID: {result.id}")
    
    return result

if __name__ == "__main__":
    try:
        result = create_lca_model()
        print("LCA模型创建和计算成功")
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
'''
        
        return code_template
    
    def _generate_code_with_llm(self, prompt: str) -> str:
        """使用LLM生成代码"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=self.max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取代码部分
            if "```python" in generated_code:
                code_start = generated_code.find("```python") + 9
                code_end = generated_code.find("```", code_start)
                if code_end != -1:
                    return generated_code[code_start:code_end].strip()
            
            return generated_code
            
        except Exception as e:
            logger.error(f"LLM代码生成失败: {str(e)}")
            # 降级到模板生成
            return self._generate_code_with_template({"instruction": "LLM生成失败，使用模板"})
