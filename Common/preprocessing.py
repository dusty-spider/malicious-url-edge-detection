import re

# url canonicalisation
def canonicalize_url(url: str) -> str:
    url = url.lower().strip()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    url = url.rstrip('/')
    return url

# tokenisation function used
def tokenize(url: str, char_to_idx: dict) -> list:
    return [char_to_idx.get(char, 1) for char in url]  # 1 = <UNK>

