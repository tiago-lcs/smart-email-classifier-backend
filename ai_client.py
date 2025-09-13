import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")  # defina no Render/Deta como variável de ambiente

# Modelos escolhidos
ZERO_SHOT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2"
GEN_MODEL = "google/flan-t5-small"

HF_API_BASE = "https://api-inference.huggingface.co/models"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

def zero_shot_classify(text: str, candidate_labels=None, multi_label=False):
    """
    Usa a Inference API (Zero-Shot Classification).
    """
    if candidate_labels is None:
        candidate_labels = ["Produtivo", "Improdutivo"]

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": candidate_labels,
            "multi_label": multi_label
        }
    }
    url = f"{HF_API_BASE}/{ZERO_SHOT_MODEL}"
    r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Resposta esperada: {labels: [...], scores: [...]}
    labels = data.get("labels", [])
    scores = data.get("scores", [])
    if not labels or not scores:
        return {"label": "Improdutivo", "score": 0.0}
    # Pega o topo
    top_label = labels[0]
    top_score = float(scores[0])
    return {"label": top_label, "score": top_score}

def generate_reply(prompt: str, max_new_tokens=80, temperature=0.3):
    """
    Gera resposta curta com FLAN-T5 small (barato e ok para MVP).
    """
    url = f"{HF_API_BASE}/{GEN_MODEL}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
    }
    r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Formato: [{"generated_text": "..."}] OU {"error":...}
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    # Alguns modelos retornam {"generated_text": "..."} direto
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    return "Obrigado pelo contato. Em breve retornaremos."