from .test import predict
from .distill_sit import cleaner
from dotenv import load_dotenv
load_dotenv()

def bertPredict(text):
    cleaned_text, enough = cleaner(text)
    if(enough): return predict(cleaned_text)
    return 0
