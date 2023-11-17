# モジュールをインポート
import openai

# API-KEYを設定
openai.api_key = 'YOUR_API_KEY'

# 関数を使用してChatGPTに問い合わせを送信
response_data = openai.ChatCompletion.create(

    # 使用するモデルの指定
    model="gpt-3.5-turbo",

    # サンプリング温度。値が高いほど、多様な文章が生成される
    # 0～1の間で指定、デフォルトでは1
    temperature = 0.8,

    # ストップシーケンスを指定するパラメータ
    # トークンの生成を停止するシーケンス、4つまで指定可能
    # stop = '',

    # レスポンスのトークン最大数を指定するパラメータ
    # デフォルトでは、4096token - 入力token 
    # max_tokens = 5,

    # 繰り返し同じ文章が生成されるのを制御するパラメータ
    # -2.0～2.0の間で値を指定
    # frequency_penalty = -2,

    # 特定の単語の出現率を設定することができるパラメータ
    # logit_bias = {96096:20},

    messages=[
        {"role": "user", "content": "Pythonで1から10までを出力するプログラムを書いてください" }
    ]
)
# print(type(response_data))
print(response_data["choices"][0]["message"]["content"])


#this is change
#aaaaaaa

#change2
#aaaaaaaaaaa
