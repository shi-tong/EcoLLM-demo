"""
单位相关API端点
Unit-related API Endpoints

提供单位选择和验证的API支持
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os

# 添加utils路径
sys.path.append(os.path.dirname(__file__))

try:
    from ..utils.unit_processor import (
        get_unit_dropdown_options, 
        get_grouped_unit_options,
        search_units, 
        validate_unit, 
        normalize_unit
    )
except ImportError:
    # 备用导入
    from utils.unit_processor import (
        get_unit_dropdown_options, 
        get_grouped_unit_options,
        search_units, 
        validate_unit, 
        normalize_unit
    )

router = APIRouter(prefix="/api/units", tags=["units"])

class UnitSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class UnitValidationRequest(BaseModel):
    unit: str

class UnitValidationResponse(BaseModel):
    is_valid: bool
    message: str
    normalized_unit: Optional[str] = None

class UnitOption(BaseModel):
    value: str
    label: str
    category: str

class UnitSearchResponse(BaseModel):
    options: List[UnitOption]
    total_count: int

@router.get("/dropdown-options", response_model=List[UnitOption])
async def get_dropdown_options():
    """获取下拉菜单选项"""
    try:
        options = get_unit_dropdown_options()
        return [UnitOption(**opt) for opt in options]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading unit options: {str(e)}")

@router.get("/grouped-options")
async def get_grouped_options():
    """获取按类别分组的选项"""
    try:
        return get_grouped_unit_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading grouped options: {str(e)}")

@router.post("/search", response_model=UnitSearchResponse)
async def search_unit_options(request: UnitSearchRequest):
    """搜索单位选项"""
    try:
        options = search_units(request.query, request.limit)
        return UnitSearchResponse(
            options=[UnitOption(**opt) for opt in options],
            total_count=len(options)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching units: {str(e)}")

@router.post("/validate", response_model=UnitValidationResponse)
async def validate_unit_input(request: UnitValidationRequest):
    """验证单位输入"""
    try:
        is_valid, message = validate_unit(request.unit)
        normalized = normalize_unit(request.unit) if is_valid else None
        
        return UnitValidationResponse(
            is_valid=is_valid,
            message=message,
            normalized_unit=normalized
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating unit: {str(e)}")

@router.get("/normalize/{unit}")
async def normalize_unit_endpoint(unit: str):
    """标准化单位"""
    try:
        normalized = normalize_unit(unit)
        return {
            "original_unit": unit,
            "normalized_unit": normalized,
            "is_standard": normalized != unit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error normalizing unit: {str(e)}")

@router.get("/categories")
async def get_unit_categories():
    """获取单位类别列表"""
    try:
        options = get_unit_dropdown_options()
        categories = list(set(opt['category'] for opt in options))
        categories.sort()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading categories: {str(e)}")

# 健康检查端点
@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "unit_api"}
