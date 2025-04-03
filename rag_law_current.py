
import os
import openai
import pandas as pd
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  
from langchain_openai import OpenAIEmbeddings  
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 2. Load dataset (casename classification)
dataset = load_dataset("lbox/lbox_open", "casename_classification_plus", trust_remote_code=True)
df = pd.DataFrame(dataset['train'])

# 3. Filter only deposit-related cases
deposit_cases = df[df['casename'].isin(['보증금반환', '임대차보증금'])]

# 4. Prepare text chunks for retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=32,
    separators=["\n\n", "\n", ".", " "]
)

texts = [
    f"사례와 관련 법령: {row['facts']}\n"
    for _, row in deposit_cases.iterrows()
    if isinstance(row['facts'], str)
]

documents = text_splitter.split_text("\n".join(texts))

# 5. Create FAISS Vector Store using OpenAI Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    openai_api_key=os.getenv("OPENAI_API_KEY") 
)
vector_store = FAISS.from_texts(documents, embedding=embedding_model)

# 6. Define Prompt Template for GPT-4
template = """[법률 분석 요청]
상황: {context}
질문: {question}
답변:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 7. Function to Generate Answer Using OpenAI's New API
def generate_answer(query, context):
    input_text = QA_PROMPT.format(context=context, question=query)
    
    response = openai.chat.completions.create( 
        model="gpt-4",
        messages=[
            {"role": "system", "content": "법률 질문을 도와드립니다."},
            {"role": "user", "content": input_text}
        ],
        max_tokens=1000, 
        temperature=0.7,
        top_p=0.9
    )
    
    return response.choices[0].message.content

# 8. Query Execution
def rag(question):
    contexts = vector_store.similarity_search(question, k=2)
    return generate_answer(question, "\n".join([c.page_content for c in contexts]))

    # print("\n[AI 법률 분석]\n", answer)
