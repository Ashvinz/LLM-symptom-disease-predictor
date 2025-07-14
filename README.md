# 🩺 Symptom → Disease Predictor (LLM powered)

A Language Model-powered medical assistant that predicts diseases from user-input symptoms using models like **FLAN-T5**, **Mistral**, or your own **fine-tuned LLMs**.

---

## 🔍 What This Project Does

This app takes symptom input in natural language (even **Thonglish**) and predicts the most likely disease using LLMs (Large Language Models). You can use:

- ✅ Open-source pre-trained models (FLAN-T5, Falcon, etc.)
- ✅ Your **fine-tuned Mistral** model trained with **Unsloth**
- ✅ Real-time prediction using **Streamlit UI**

---

## 🧰 Packages & Tools Used

| Package             | Purpose |
|---------------------|---------|
| [`transformers`](https://huggingface.co/docs/transformers/index) | Load, run, and fine-tune LLMs |
| [`torch`](https://pytorch.org/) | Core deep learning backend |
| [`unsloth`](https://github.com/unslothai/unsloth) | Fast, memory-efficient fine-tuning of Mistral/LLMs |
| [`streamlit`](https://streamlit.io/) | Interactive frontend for real-time prediction |
| [`datasets`](https://huggingface.co/docs/datasets) | Dataset preparation & loading |
| [`json`](https://docs.python.org/3/library/json.html) | Reading JSONL input for training |
| [`traceback`](https://docs.python.org/3/library/traceback.html) | Error handling and debug info |

---

## 🎯 Project Objective

> To demonstrate how fine-tuned LLMs can assist in disease prediction based on natural language symptom descriptions.

---

## 📦 Installation

```bash
pip install -r requirements.txt
