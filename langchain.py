import io,os
os.environ["OPENAI_API_KEY"] ='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'

from langchain.llms import OpenAIChat
from langchain import PromptTemplate, LLMChain

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import openai


# ドキュメントを読み込み
file_path = r'D:\work\00_git\private\work\llm_test3\env\sample.txt'
documents = TextLoader(file_path).load()

# 改行でチャンクに分割
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
documents = text_splitter.split_documents(documents)

# 分割したテキストをベクターストア (Chroma）に保存する
store = Chroma.from_documents(documents, OpenAIEmbeddings())

# キーワードで類似検索する
question = "勤務時間について教えてください。"
keyword = "労働時間"

docs = store.similarity_search(keyword)
related_content = docs[0].page_content

print(related_content)


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"あなたは就業規則に詳しいアシスタントです"},
        {"role": "system", "content": f"以下のテキストをもとにユーザーの質問に回答してください: {related_content}"},
        {"role": "user", "content": question},
    ],
) 

print(response['choices'][0]['message']['content'])
print(related_content)

# for index, doc in enumerate(docs):
#     print(f"No{index+1}. {doc.page_content}")

# # LLMの準備
# llm_openai = OpenAIChat(temperature=0)

# # プロンプトテンプレートの準備
# template = """Q: {question}

# A: 一歩一歩、考えていきましょう。"""
# prompt = PromptTemplate(
#     template=template, 
#     input_variables=["question"]
# )

# # チェーンの準備
# llm_chain = LLMChain(
#     prompt=prompt, 
#     llm=llm_openai
# )

# # チェーンの実行
# question = "藤井聡太が生まれた年のワールドカップの優勝チームは？"
# llm_chain.run(question)