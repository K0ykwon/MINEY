# 🛠️ M$ney

## 📌 Overview
This project was developed as part of the **AGI Agent Application Hackathon**.  
It aims to support victims of **rental fraud** by providing AI-powered legal document analysis and actionable legal guidance by leveraging advanced AI agent technologies.

---

## 🚀 Key Features
✅ **Contract Analysis**: Uses Upstage OCR and GPT models to extract and analyze key clauses from rental contracts, identifying hidden risks and unfavorable terms.
✅ **Actionable Legal Guidance**: Users can describe their legal situation, and based on BERT-powered analysis, the system provides relevant laws, precedents, required documents, and step-by-step instructions for legal response.
✅ **Contextual Q&A**: Users can ask questions in natural language, and the system — fine-tuned on legal texts using BERT — returns document-specific, context-aware legal answers.

---

## 🖼️ Demo / Screenshots
![Landing](./readmeimage/landing.png)
![Functions](./readmeimage/readme.png)
  
[👉 demo video](https://www.youtube.com/watch?v=GuP2Haq75-Y)
[👉 live demo website link ](https://iamanidiot.streamlit.app/)

---

## 🧩 Tech Stack

**Frontend**: Streamlit  
**Backend**: Python (custom scripts & API integrations) 
**Database**: FAISS (Vector Store for Legal Document Retrieval)  
**Others**: OpenAI API / Upstage OCR API/ LangChain / HuggingFace Transformers

---

## 🏗️ Project Structure
```
📁MINEY/
├── .devcontainer/ 
├── docs/ 
├── Predict/ 
├── readmeiamge/ 
├── referencnes/ 
├── 계약서input예시(테스트용)/ 
├── .gitattributes 
├── .gitignore 
├── answer.py 
├── contract_analysis.py 
├── main.py 
├── rag_law_current.py 
├── README.md 
├── requirements.txt 
├── risk_assessor.py 
├── test.py 
├── tutorial1.png 
├── tutorial2.png 
├── tutorial3.png 
└── upstage.py
```

---

## 🔧 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/UpstageAI/cookbook/usecase/agi-agent-application/miney.git

# run
pip install -r requirements.txt
streamlit run main.py
```

---

## 📁 Dataset & References

### 📊 Dataset Used

- **LBox Open Dataset**  
  Korean court rulings and legal clauses used to fine-tune the BERT model for legal prediction tasks.

- **Custom OCR Samples**  
  Real-world rental contracts, including the official government-standard lease agreement, were collected and anonymized for OCR testing, parsing, and downstream legal analysis.

### 🔗 References / Resources

- [LangChain](https://www.langchain.com/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [LBox](https://lbox.kr/v2)


---

## 🙌 Team Members

| Name        | Role               | GitHub         |
|-------------|--------------------|----------------|
| Ko Youngkwon | AI Developer | [@k0ykwon](https://github.com/k0ykwon) |
| Seo Suyeon | Frontend Developer  | [@ellie3413](https://github.com/ellie3413) |
| Yeom Seo Kyung | Backend Developer  | [@skyyeom](https://github.com/skyyeom) |
| Yoon Tae Du | Backend Developer  | [@taedooit](https://github.com/taedooit) |
| Lim Chaeyoon | Backend Developer  | [@2022148081](https://github.com/2022148081) |

---

## ⏰ Development Period
Last updated: 2025-04-05

---

## 📄 License
This project is licensed under the MIT license.  
See the [LICENSE](./LICENSE) file for more details.

---

## 💬 Additional Notes
- This project was developed as an MVP during the AGI Agent Application Hackathon.
- All legal responses are generated by AI models and are intended for informational purposes only.
- It is not a substitute for professional legal advice. Please consult a licensed attorney for critical legal decisions.
- We plan to expand the service to other legal domains (e.g., labor law, consumer protection) in the future.




