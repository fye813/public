# import sys
# lib_path = r"D:\work\00_git\private\work\llm_test2\env\Lib\site-packages"
# sys.path.append(lib_path)
import os
import platform

import openai
import chromadb
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

# LangChain における LLM のセットアップ
os.environ["OPENAI_API_KEY"] ='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'
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# テキストの分割
loader = PyPDFLoader(r"D:\work\tmp\1411.4555v2.pdf")
pages = loader.load_and_split()
# pages[10].page_content

# 分割したテキストの情報をベクターストアに格納する
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".")
vectorstore.persist()

# PDF ドキュメントへ自然言語で問い合わせる
pdf_qa = ConversationalRetrievalChain.from_llm(
    llm, vectorstore.as_retriever(), return_source_documents=True)

# query = "What is US Cloud Act? Answer within 50 words in Japanese."
chat_history = []

# result = pdf_qa({"question": query, "chat_history": chat_history})

# result["answer"]
# result['source_documents']

# chat_history = [(query, result["answer"])]

query2 = "このPDFに記載されている内容を簡単にまとめてください"

result2 = pdf_qa({"question": query2, "chat_history": chat_history})

result2["answer"]

# # ドキュメントを読み込み
# file_path = r'D:\work\00_git\private\work\llm_test3\env\sample.txt'
# documents = TextLoader(file_path).load()

# # 改行でチャンクに分割
# text_splitter = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=0
# )
# documents = text_splitter.split_documents(documents)

# # 分割したテキストをベクターストア (Chroma）に保存する
# store = Chroma.from_documents(documents, OpenAIEmbeddings())

# # キーワードで類似検索する
# question = "勤務時間について教えてください。"
# keyword = "労働時間"

# docs = store.similarity_search(keyword)
# related_content = docs[0].page_content

# print(related_content)


# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": f"あなたは就業規則に詳しいアシスタントです"},
#         {"role": "system", "content": f"以下のテキストをもとにユーザーの質問に回答してください: {related_content}"},
#         {"role": "user", "content": question},
#     ],
# ) 

# print(response['choices'][0]['message']['content'])
# print(related_content)
