from langchain.document_loaders import WebBaseLoader

import os
import platform

import openai
import chromadb
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders import PyPDFLoader

# LangChain における LLM のセットアップ
os.environ["OPENAI_API_KEY"] ='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'
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Webサイトから取得したテキストを分割
target_url = r'https://www.ark-gold.com/'
loader = WebBaseLoader(
    target_url,
    # プロキシの使用が必要となる場合は以下を記述
    # proxies={
    #     "http": "http://{username}:{password}:@proxy.service.com:6666/",
    #     "https": "https://{username}:{password}:@proxy.service.com:6666/",
    # },
)
documents = loader.load()
documents
# print(documents[0].dict()["page_content"])

# # 分割したテキストの情報をベクターストアに格納する
# embeddings = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=".")
# vectorstore.persist()

# 分割したテキストをベクターストア (Chroma）に保存する
store = Chroma.from_documents(documents, OpenAIEmbeddings())

# キーワードで類似検索する
question = "このサイトには金の価格についてどのように書かれていますか"
keyword = "金"

docs = store.similarity_search(keyword)
related_content = docs[0].page_content

# print(related_content)


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"あなたはAIです"},
        {"role": "system", "content": f"以下のテキストは、ユーザーが指定したWebサイトからキーワードを類似検索したものになります。このテキストをもとにユーザーの質問に回答してください: {related_content}"},
        {"role": "user", "content": question},
    ],
) 

print(response['choices'][0]['message']['content'])
#print(related_content)

####################################
# PDF ドキュメントへ自然言語で問い合わせる
pdf_qa = ConversationalRetrievalChain.from_llm(
    llm, vectorstore.as_retriever(), return_source_documents=True)
chat_history = []

query = "結束バンドとは何か、要約してください。"
result = pdf_qa({"question": query, "chat_history": chat_history})
result["answer"]
