import os
from dotenv import load_dotenv
from langchain_gigachat import GigaChat

load_dotenv()

print("cred:", os.getenv("GIGACHAT_CREDENTIALS")[:20])
print("scope:", os.getenv("GIGACHAT_SCOPE"))

model = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
    verify_ssl_certs=False,
)

response = model.invoke("Привет! Ответь одним словом: работает.")
print(response.content)