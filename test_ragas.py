"""Quick RAGAS answer_relevancy debug test"""
import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

d = Dataset.from_dict({
    "question": ["눈에 도움되는 약이 있나요?"],
    "answer": ["네, 눈 건강에 도움되는 제품이 있습니다. 루테인은 눈 건강에 도움을 줍니다. [문서 1]"],
    "contexts": [["루테인은 눈 건강에 도움을 주는 건강기능식품 성분입니다. 1일 1~2정 복용합니다."]]
})

llm = ChatOpenAI(model="gpt-5.2")
emb = OpenAIEmbeddings(model="text-embedding-3-small")

print("Testing with raise_exceptions=True...")
try:
    r = evaluate(dataset=d, metrics=[answer_relevancy], llm=llm, embeddings=emb, raise_exceptions=True)
    df = r.to_pandas()
    print("Columns:", df.columns.tolist())
    print("Values:", df.to_dict())
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\nTesting with raise_exceptions=False...")
r2 = evaluate(dataset=d, metrics=[answer_relevancy], llm=llm, embeddings=emb, raise_exceptions=False)
df2 = r2.to_pandas()
print("Columns:", df2.columns.tolist())
print("Values:", df2.to_dict())
