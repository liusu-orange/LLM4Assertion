#uvicorn app:app --reload --host 0.0.0.0 --port 8000
#http://<你的服务器IP>:8000
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# 模板引擎（用于渲染 HTML）
templates = Jinja2Templates(directory="templates")

# 加载模型和分词器
model_path = "/home/featurize/work/unsloth/unsloth_sft_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

# 定义输入数据模型
class QuestionRequest(BaseModel):
    Code: str
    Synchronous: str
    Clock: Optional[str] = None

# Alpaca 提示模板
alpaca_prompt = """Below is a piece of hardware design code (in Verilog) and its corresponding verification assertions (in SystemVerilog). Generate the assertions based on the given design code.
### Design Code:
{}
### Assertions:
{}"""

# API 路由
@app.post("/generate")
async def generate_answer(request: QuestionRequest):
    try:
        input_text = alpaca_prompt.format(request.Code, "")
        inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            use_cache=True,
        )
        full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generated_answer = full_response.split("### Assertions:")[-1].strip()
        
        return {"Assertions": generated_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 首页路由（返回 HTML）
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


