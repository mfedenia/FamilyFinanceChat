from fastapi import APIRouter, Depends, HTTPException, Request
from open_webui.utils.auth import get_verified_user
from pydantic import BaseModel
import logging

log = logging.getLogger(__name__)
router = APIRouter()

class CustomActionRequest(BaseModel):
    data: dict = {}

@router.post("/execute")
async def execute_custom_action(
    request: Request,
    form_data: CustomActionRequest,
    user=Depends(get_verified_user)
):
    try:
        result = {
            "status": "success", 
            "message": "Custom action executed",
            "user": user.name,
            "data": form_data.data
        }
        log.info(f"Custom action executed by user {user.name}")
        return result
    except Exception as e:
        log.error(f"Error in custom action: {e}")
        raise HTTPException(status_code=400, detail=str(e))
