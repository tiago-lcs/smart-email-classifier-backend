import re

_pt_stop = {
    "a","o","as","os","de","da","do","das","dos","em","no","na","nos","nas",
    "para","por","com","sem","um","uma","uns","umas","e","ou","que","se",
    "sobre","ao","à","aos","às","é","ser","está","estao","estão"
}

def preprocess(text: str) -> str:
    """
    MVP: limpeza leve + normalização. (Sem libs pesadas)
    """
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"https?://\S+", " ", t)           # remove URLs
    t = re.sub(r"[^\wÀ-ÿ@#\s]", " ", t)           # mantém letras acentuadas
    tokens = [tok for tok in t.split() if tok not in _pt_stop]
    return " ".join(tokens)
