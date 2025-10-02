# utils/__init__.py
import os
import importlib
from glob import glob

# 自动导入所有py文件中的类
py_files = glob(os.path.join(os.path.dirname(__file__), "*.py"))
all_classes = {}

for py_file in py_files:
    file_name = os.path.basename(py_file)
    if file_name.startswith("__") or file_name == "__init__.py":
        continue
    
    module_name = file_name[:-3]  # 去掉.py后缀
    try:
        module = importlib.import_module(f".{module_name}", package="utils")
        
        # 获取模块中的所有类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and not attr_name.startswith("_"):
                all_classes[attr_name] = attr
                globals()[attr_name] = attr  # 直接添加到全局命名空间
                
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")

# 定义__all__以便from utils import *使用
__all__ = list(all_classes.keys())