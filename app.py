import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from ai_client import zero_shot_classify, generate_reply, HFError
from nlp import preprocess

app = Flask(__name__)
CORS(app)

REPLY_SYSTEM = """
Você é um assistente de atendimento por e-mail de uma empresa do setor financeiro.
Responda de modo educado, objetivo e profissional em português do Brasil.
Se for improdutivo (ex.: felicitações), agradeça de forma breve.
Se for produtivo, peça as informações mínimas ou confirme o próximo passo, de modo curto.
Assine como Equipe de Suporte.
"""

@app.route("/", methods=["GET"])
def root():
    return (
        "<h1>Smart Email Classifier - Backend</h1>"
        "<p>API online 🚀</p>"
        "<p>Use <code>/health</code> e <code>POST /classify</code>.</p>"
    ), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/classify", methods=["GET"])
def classify_info():
    return jsonify({"message": "Use POST /classify com JSON: {\"text\": \"...\"}"}), 200

def _fallback_rule_based(original_text: str):
    """
    Regrinhas simples para nunca quebrar o MVP quando a HF falhar.
    """
    t = original_text.lower()
    palavras_prod = ["status", "chamado", "ticket", "protocolo", "anexo", "fatura", "boleto",
                     "reembolso", "erro", "suporte", "prazo", "previsão", "documento", "acesso"]
    produtivo = any(p in t for p in palavras_prod)
    categoria = "Produtivo" if produtivo else "Improdutivo"
    if produtivo:
        reply = ("Olá! Para agilizar, poderia confirmar o número do chamado/protocolo "
                 "e, se houver, anexar os documentos pendentes? Assim seguimos com o próximo passo.\n\n"
                 "Atenciosamente,\nEquipe de Suporte")
    else:
        reply = ("Olá! Obrigado pela mensagem. Ficamos à disposição.\n\n"
                 "Atenciosamente,\nEquipe de Suporte")
    return {"category": categoria, "confidence": 0.51 if produtivo else 0.51, "suggested_reply": reply}

def make_generation_prompt(category: str, original_text: str) -> str:
    if category.lower().startswith("produt"):
        instr = "Classifiquei o e-mail como PRODUTIVO. Escreva uma resposta curta, pedindo o que falta ou indicando o próximo passo."
    else:
        instr = "Classifiquei o e-mail como IMPRODUTIVO. Escreva um agradecimento curto e cordial, sem abrir ticket."
    return (
        f"{REPLY_SYSTEM}\n\nInstrução: {instr}\n\n"
        f"E-mail do cliente:\n\"\"\"\n{original_text}\n\"\"\"\n\nResposta proposta:"
    )

@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Texto vazio."}), 400

        cleaned = preprocess(text)

        try:
            z = zero_shot_classify(cleaned, ["Produtivo", "Improdutivo"], multi_label=False)
            category = z["label"]
            score = float(z["score"])
            gen_prompt = make_generation_prompt(category, text)
            reply = generate_reply(gen_prompt, max_new_tokens=80, temperature=0.3)
            return jsonify({
                "category": category,
                "confidence": round(score, 4),
                "suggested_reply": reply
            })
        except HFError as e:
            # Fallback para garantir resposta mesmo sem HF
            fb = _fallback_rule_based(text)
            fb["note"] = f"fallback:true; reason:{str(e)}"
            return jsonify(fb), 200

    except Exception as e:
        # Última linha de defesa: nunca 500 opaco
        return jsonify({"error": "Falha inesperada no servidor.", "detail": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
