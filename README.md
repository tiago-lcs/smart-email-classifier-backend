# Smart Email Classifier – Backend

API Flask que processa e-mails, faz pré-processamento de texto, classifica com **Hugging Face Inference API** e gera uma resposta automática.

## 🛠️ Tecnologias

- Flask + Flask-CORS

- Hugging Face Inference API (zero-shot + geração de texto)

- Deploy no Render (Free Tier)


## 🚀 Endpoints
- `GET /health` → checar status da API  
- `POST /classify`  
  Body JSON:
  ```json
  { "text": "Olá, preciso do status do ticket #123." }

Retorno:
```json{
  "category": "Produtivo",
  "confidence": 0.87,
  "suggested_reply": "Resposta automática sugerida..."
}




