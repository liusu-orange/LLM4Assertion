# Verilog Assertion Training & Inference aaa

This project is designed for training and inference using Verilog code assertion pairs. It includes scripts for fine-tuning models and a local server for deployment.

---

## 📦 Installation

Install the required packages using the following commands:

```bash
pip --default-timeout=1000 install "unsloth[121]" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install trl transformers accelerate peft bitsandbytes
pip install tf-keras
pip install pandas openpyxl
pip install nltk rouge
pip install scikit-learn
pip install wandb
pip install uvicorn
pip install fastapi

Dataset
The dataset (vert.xlsx) contains 20,000 Verilog code assertion pairs.

---

## 🚀 Usage

### Training and Inference

Run the following script to start training and inference:

```bash
python sft.py
```

### Local Server

Start the local server with the following command:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## 🖥️ Frontend

The frontend code is located in the `template` directory (`index.html`).

---

## 📁 Project Structure

- `vert.xlsx`: Dataset file
- `sft.py`: Script for training and inference
- `app.py`: Script to start the local server
- `template/`: Directory containing frontend code (`index.html`)

---

## ⚠️ Notes

- Ensure all dependencies are installed before running the scripts.

