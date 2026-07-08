import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 现在可以直接读取，或者很多大模型 SDK 会自动寻找这些环境变量
api_key = os.getenv("OPENAI_API_KEY")