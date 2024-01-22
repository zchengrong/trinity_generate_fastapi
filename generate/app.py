# main.py
import json

from fastapi import FastAPI
from celery import Celery
from celery.result import AsyncResult
from pydantic import BaseModel
from tasks import generate_image

app = FastAPI()

# 初始化Celery
celery = Celery(
    "tasks",
    broker="redis://127.0.0.1:6379/0",
    backend="redis://127.0.0.1:6379/1"
)


# API模型


class RequestData(BaseModel):
    mode: int
    category: str
    content: str
    user_id: int
    image_url: str
    version: str


# API端点
@app.post("/generate_image")
async def generate_image_endpoint(Data: RequestData):
    # 将任务放入Celery队列
    print(Data)
    # json_data = json.dumps(Data)
    task = generate_image.delay(Data.dict())
    return {"task_id": str(task)}


# 获取任务状态端点
@app.get("/get-task-status/{task_id}")
async def get_task_status(task_id: str):
    result = AsyncResult(task_id, app=celery)
    return {"status": result.status, "result": result.result}


@app.get("/cancel_task/{task_id}")
async def get_task_status(task_id: str):
    result = AsyncResult(task_id)
    result.revoke(terminate=True)
    return {"status": result.status, "result": result.result}
