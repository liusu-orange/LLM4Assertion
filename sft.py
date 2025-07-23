# pip --default-timeout=1000 install "unsloth[121]" -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install trl transformers accelerate peft bitsandbytes
# pip install tf-keras
# pip install pandas openpyxl
# pip install nltk rouge
# pip install scikit-learn
# pip install wandb
# pip install uvicorn
# pip install fastapi
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import torch
import pandas as pd
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import re
from tqdm import tqdm

# 模型和训练参数
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 加载模型和Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 配置LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# 定义模板（instruction为空，input=Question，output=Answer）
# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that
#  provides further context. Write a response that appropriately completes the request.
#  ### Instruction:
#  Analyze and generate assertions for processor verification based on the given dataset.
#  ### Input:
#  {}
#  ### Response:
#  {}"""

alpaca_prompt = """Below is a piece of hardware design code (in Verilog) and its corresponding verification assertions (in SystemVerilog). Generate the assertions based on the given design code.
### Design Code:
{}
### Assertions:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
# 数据格式化函数
def formatting_prompts_func(examples):
    Question = examples["Question"]
    Answer = examples["Answer"]
    texts = []
    for input_text, output_text in zip(Question, Answer):
        text=(alpaca_prompt.format(input_text, output_text))+EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# 加载Excel数据
df = pd.read_excel("/home/featurize/work/unsloth/VERT.xlsx", engine="openpyxl")

# 检查列名
if "Question" not in df.columns or "Answer" not in df.columns:
    raise ValueError("Excel文件必须包含 'Question' 和 'Answer' 列")

# 转换为Hugging Face Dataset
dataset = Dataset.from_pandas(df)
# 第一次拆分：分出测试集（10%）
dataset_split = dataset.train_test_split(
    test_size=0.1,          # 10%测试集
    seed=3407,             # 随机种子确保可复现性
    shuffle=True           # 打乱数据顺序
)

# 第二次拆分：从剩余数据中分出验证集（10%）
train_val_split = dataset_split["train"].train_test_split(
    test_size=0.1,         # 10%验证集（相对于剩余数据的10%）
    seed=3407,
    shuffle=True
)

# 最终数据集
dataset = {
    "train": train_val_split["train"],
    "validation": train_val_split["test"],
    "test": dataset_split["test"]
}
# 应用格式化函数
for key in dataset:
    dataset[key] = dataset[key].map(formatting_prompts_func, batched=True)
# 查看拆分结果


# 定义训练参数
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=2000,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    eval_strategy="steps",       # 按步评估
    eval_steps=1000,                    # 每20步验证一次
    save_strategy="steps", 
    save_steps=1000,           # 同步保存检查点
    load_best_model_at_end=True,      # 训练结束时加载最佳模型
    metric_for_best_model="eval_loss", # 根据验证损失选择最佳模型
    output_dir="unsloth_sft_outputs",
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="wandb",    
)


# 初始化SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],  # 训练集
    eval_dataset=dataset["validation"],    # 验证集
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
    # compute_metrics=compute_metrics,  # 添加计算指标的函数
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics,  # 添加预处理函数
)


# 开始训练
trainer.train()

# 保存模型和Tokenizer
trainer.model.save_pretrained("unsloth_sft_model")
tokenizer.save_pretrained("unsloth_sft_model")

# 加载训练好的模型和Tokenizer
model_path = "/home/featurize/work/unsloth/unsloth_sft_model"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 启用推理模式（2倍加速）
model = FastLanguageModel.for_inference(model)

# 定义推理函数（直接返回完整生成内容）
def generate_full_response(question):
    # 构造输入 prompt（Answer 部分留空）
    input_text = alpaca_prompt.format(question, "")  # Answer 留空，让模型生成
    inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
    
    # 生成完整文本
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,  # 控制生成的最大长度
        use_cache=True,
    )
    
    # 解码为完整字符串（包括问题和答案）
    full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return full_response

# 测试推理
test_samples = dataset["test"].shuffle(seed=3407).select(range(5))
for sample in test_samples:
    question = sample["Question"]
    ground_truth = sample["Answer"]  # 真实答案
    full_response = generate_full_response(question)
    print(f"Generated Response:\n{full_response}\n{'-'*50}")
    print(f"Ground Truth Answer:\n{ground_truth}\n")

test_samples = dataset["test"]

# 存储真实答案和生成答案
ground_truths = []
generated_answers = []

for sample in tqdm(test_samples, desc="Processing samples"):
    question = sample["Question"]
    ground_truth = sample["Answer"]
    full_response = generate_full_response(question)
    
    # 提取生成的断言部分（从 "### Assertions:" 开始）
    generated_answer = full_response.split("### Assertions:")[-1].strip()
    
    ground_truths.append(ground_truth)
    generated_answers.append(generated_answer)

# 1. 计算生成任务指标（BLEU、ROUGE）
smoothie = SmoothingFunction().method4
bleu_scores = [
    sentence_bleu([gt.split()], gen.split(), smoothing_function=smoothie)
    for gt, gen in zip(ground_truths, generated_answers)
]
avg_bleu = sum(bleu_scores) / len(bleu_scores)

rouge = Rouge()
rouge_scores = rouge.get_scores(generated_answers, ground_truths, avg=True)

def extract_property_content(text):
    """提取所有 property 和 endproperty 之间的内容，跳过每个块中第一个分号之前的部分"""
    pattern = r'property\s+(.*?)\s*endproperty'
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for match in matches:
        content = match.strip()
        first_semicolon_pos = content.find(';')
        if first_semicolon_pos != -1:
            results.append(content[first_semicolon_pos + 1:].strip())
    return results
ground_truth_contents = []
generated_contents = []

for gt, gen in tqdm(zip(ground_truths, generated_answers), desc="Extracting content"):
    gt_content = extract_property_content(gt)
    gen_content = extract_property_content(gen)
    
    
    ground_truth_contents.append(gt_content)
    generated_contents.append(gen_content)

# 计算准确率
correct = 0
for gt_list, gen_list in zip(ground_truth_contents, generated_contents):
    # 确保 gt_list 和 gen_list 是列表且长度一致
    if isinstance(gt_list, list) and isinstance(gen_list, list):
        all_match = True
        for gt_str, gen_str in zip(gt_list, gen_list):
            if gt_str.replace(" ", "") != gen_str.replace(" ", ""):
                all_match = False
                print(f"Ground Truth: {gt_str}")
                print(f"Generated: {gen_str}")
                print("-" * 50)  # 分隔线
                break
        if all_match:
            correct += 1

accuracy = correct / len(ground_truth_contents) if ground_truth_contents else 0

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Average BLEU: {avg_bleu:.4f}")
print(f"ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}")
print(f"ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}")
print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")





