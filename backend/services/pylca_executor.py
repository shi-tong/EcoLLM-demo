import subprocess
import tempfile
import os
import sys
import logging
from typing import Dict, Any
import json
import signal
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PyLCAExecutor:
    """pyLCA代码执行器，安全执行生成的代码"""
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout  # 执行超时时间（秒）
        self.allowed_imports = {
            'olca', 'uuid', 'decimal', 'json', 'math', 'datetime',
            'numpy', 'pandas', 'brightway2', 'bw2data', 'bw2calc'
        }
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        在安全环境中执行pyLCA代码
        
        Args:
            code: 要执行的Python代码
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        try:
            # 验证代码安全性
            if not self._validate_code_safety(code):
                return {
                    "success": False,
                    "error": "代码包含不安全的操作",
                    "output": "",
                    "execution_time": 0
                }
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            try:
                # 执行代码
                result = self._execute_in_subprocess(temp_file_path)
                return result
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"执行pyLCA代码失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "execution_time": 0
            }
    
    def _validate_code_safety(self, code: str) -> bool:
        """验证代码安全性"""
        try:
            # 检查危险操作
            dangerous_patterns = [
                'import os', 'import subprocess', 'import shutil',
                'import socket', 'import urllib', 'import requests',
                'open(', 'file(', 'exec(', 'eval(',
                '__import__', 'globals()', 'locals()',
                'rm ', 'del ', 'rmdir', 'unlink'
            ]
            
            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if pattern in code_lower:
                    logger.warning(f"代码包含危险模式: {pattern}")
                    return False
            
            # 检查允许的导入
            lines = code.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    # 提取模块名
                    if line.startswith('import '):
                        module = line.split('import ')[1].split()[0].split('.')[0]
                    else:  # from ... import
                        module = line.split('from ')[1].split()[0].split('.')[0]
                    
                    if module not in self.allowed_imports:
                        logger.warning(f"不允许的模块导入: {module}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"代码安全验证失败: {str(e)}")
            return False
    
    def _execute_in_subprocess(self, script_path: str) -> Dict[str, Any]:
        """在子进程中执行代码"""
        import time
        start_time = time.time()
        
        try:
            # 设置环境变量
            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(sys.path)
            
            # 执行代码
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                
                return {
                    "success": process.returncode == 0,
                    "output": stdout,
                    "error": stderr if process.returncode != 0 else "",
                    "return_code": process.returncode,
                    "execution_time": execution_time
                }
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return {
                    "success": False,
                    "error": f"代码执行超时（{self.timeout}秒）",
                    "output": "",
                    "execution_time": self.timeout
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"执行过程中出错: {str(e)}",
                "output": "",
                "execution_time": time.time() - start_time
            }
    
    def test_openlca_connection(self) -> Dict[str, Any]:
        """测试openLCA连接"""
        test_code = """
try:
    import olca
    client = olca.Client()
    print("openLCA连接测试成功")
    print(f"客户端版本: {olca.__version__ if hasattr(olca, '__version__') else '未知'}")
except ImportError:
    print("错误: 未安装olca库")
except Exception as e:
    print(f"openLCA连接失败: {str(e)}")
"""
        return self.execute_code(test_code)
    
    def validate_generated_code(self, code: str) -> Dict[str, Any]:
        """验证生成的代码语法正确性"""
        try:
            # 语法检查
            compile(code, '<generated_code>', 'exec')
            
            # 基本结构检查
            has_import = 'import' in code
            has_function_or_main = 'def ' in code or '__main__' in code
            
            return {
                "valid": True,
                "has_import": has_import,
                "has_function_or_main": has_function_or_main,
                "message": "代码语法验证通过"
            }
            
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"语法错误: {str(e)}",
                "line": e.lineno,
                "message": "代码包含语法错误"
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "message": "代码验证失败"
            }
