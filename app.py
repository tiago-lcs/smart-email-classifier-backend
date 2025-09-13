import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from ai_client import zero_shot_classify, generate_reply
from nlp import preprocess

app = Flask(__name__)
CORS(app)  # permite chamadas do Netlify (origem diferente)

# Prompt base para geração da resposta
REPLY_SYSTEM = """
Você é um assistente de atendimento por e-mail de uma empresa do setor financeiro.
Responda de modo educado, objetivo e profissional em português do Brasil.
Se for improdutivo (ex.: felicitações), agradeça de forma breve.
Se for produtivo, peça as informações mínimas ou confirme o próximo passo, de modo curto.
Assine como Equipe de Suporte.
"""

def make_generation_prompt(category: str, original_text: str) -> str:
    if category.lower().startswith("produt"):
        instr = "Classifiquei o e-mail como PRODUTIVO. Escreva uma resposta curta, pedindo o que falta ou indicando o próximo passo."
    else:
        instr = "Classifiquei o e-mail como IMPRODUTIVO. Escreva um agradecimento curto e cordial, sem abrir ticket."
    prompt = (
        f"{REPLY_SYSTEM}\n\n"
        f"Instrução: {instr}\n\n"
        f"E-mail do cliente:\n\"\"\"\n{original_text}\n\"\"\"\n\n"
        f"Resposta proposta:"
    )
    return prompt

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Texto vazio."}), 400

    # 1) pré-processa (leve)
    cleaned = preprocess(text)

    # 2) classifica (Zero-Shot na HF API)
    z = zero_shot_classify(
        cleaned,
        candidate_labels=["Produtivo", "Improdutivo"],
        multi_label=False
    )
    category = z["label"]
    score = z["score"]

    # 3) gera resposta (FLAN-T5 small via HF API)
    gen_prompt = make_generation_prompt(category, text)
    reply = generate_reply(gen_prompt, max_new_tokens=80, temperature=0.3)

    return jsonify({
        "category": category,
        "confidence": round(score, 4),
        "suggested_reply": reply
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)