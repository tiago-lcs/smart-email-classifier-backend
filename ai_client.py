import os
import time
import requests

HF_TOKEN = os.getenv("HF_TOKEN")

ZERO_SHOT_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2"
GEN_MODEL = "google/flan-t5-small"
HF_API_BASE = "https://api-inference.huggingface.co/models"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

class HFError(Exception):
    pass

def _post_with_retry(url, payload, retries=3, backoff=2.0, timeout=30):
    last_err = None
    for i in range(retries):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            # Se o modelo estiver “carregando”, a HF pode retornar 503/200 com estimated_time
            if r.status_code == 503:
                # aguarda e tenta de novo
                time.sleep(backoff)
                continue
            r.raise_for_status()
            data = r.json()
            # alguns modelos retornam {"error":"Model ... is currently loading","estimated_time":...}
            if isinstance(data, dict) and data.get("error") and "loading" in data["error"].lower():
                time.sleep(backoff)
                continue
            return data
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    raise HFError(str(last_err) if last_err else "Unknown HF error")

def zero_shot_classify(text: str, candidate_labels=None, multi_label=False):
    if not HF_TOKEN:
        raise HFError("HF_TOKEN ausente")
    if candidate_labels is None:
        candidate_labels = ["Produtivo", "Improdutivo"]
    url = f"{HF_API_BASE}/{ZERO_SHOT_MODEL}"
    payload = {"inputs": text, "parameters": {
        "candidate_labels": candidate_labels,
        "multi_label": multi_label
    }}
    data = _post_with_retry(url, payload)
    # formatos possíveis
    labels = []
    scores = []
    if isinstance(data, dict):
        labels = data.get("labels") or []
        scores = data.get("scores") or []
    elif isinstance(data, list) and data:
        item = data[0]
        labels = item.get("labels") or []
        scores = item.get("scores") or []
    if not labels or not scores:
        # fallback neutro
        return {"label": "Improdutivo", "score": 0.0}
    return {"label": labels[0], "score": float(scores[0])}

def generate_reply(prompt: str, max_new_tokens=80, temperature=0.3):
    if not HF_TOKEN:
        raise HFError("HF_TOKEN ausente")
    url = f"{HF_API_BASE}/{GEN_MODEL}"
    payload = {"inputs": prompt, "parameters": {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature
    }}
    data = _post_with_retry(url, payload)
    # formatos possíveis
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    return "Obrigado pelo contato. Em breve retornaremos."
