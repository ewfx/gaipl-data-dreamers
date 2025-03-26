# 🚀 SupportGenie

## 📌 Table of Contents
- [Introduction](#introduction)
- [What It Does](#what-it-does)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## 🎯 Introduction
This project aims to develop a chatbot powered by Generative AI to assist IT support teams in incident management. The chatbot leverages a trained AI model to analyze incident reports, suggest potential solutions, and provide troubleshooting steps based on historical data and best practices.

## ⚙️ What It Does
The system is designed to:
1. Provide real-time AI-driven suggestions for resolving IT incidents.
2. Improve response time and efficiency in IT support operations.
3. Continuously learn from past incidents to enhance its recommendation accuracy.
4. Integrate with existing ITSM (IT Service Management) tools for seamless workflow support.

## 🏃 How to Run
- Install dependencies pip install -r requirements.txt
- Set up `.env` file for sensitive configurations
- Configure model paths
- Set up Hugging Face authentication if required
- Model Configuration
```python

model_config = {
&#39;model_name&#39;: &#39;bert-base-uncased&#39;,
&#39;max_length&#39;: 128,
&#39;num_labels&#39;: 4, # Based on incident statuses
&#39;learning_rate&#39;: 2e-5
}
```
- Local Deployment
```bash
# Run Streamlit application
streamlit run incidentchatbot.py
```

## 🏗️ Tech Stack
- **Core Libraries**:
- PyTorch
- Transformers (Hugging Face)
- LangChain
- Streamlit
- **Machine Learning**:
- BERT-based sequence classification
- Text generation pipeline
- **Data Processing**:
- Pandas
- scikit-learn
