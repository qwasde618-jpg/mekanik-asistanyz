#!/usr/bin/env python3
"""Mekanik Kolaylıklar - yapay zeka asistanı sunucusu (Google Gemini).
POST /ask  {"q":"...", "x":"...", "lang":"tr", "h":[{"r":"u"|"a","t":"..."}]}  ->  {"a":"cevap"}
GET  /health
Sadece Python standart kütüphanesi gerekir (requests da gerekmez, urllib kullanılır)."""
import json, os, re, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8080"))
ALLOW = os.environ.get("ALLOW_ORIGIN", "*")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
RATE = int(os.environ.get("RATE_PER_MIN", "30"))
GLOBAL_RATE = int(os.environ.get("GLOBAL_PER_MIN", "180"))
TIMEOUT = int(os.environ.get("ASK_TIMEOUT", "55"))

SYS = {
    "tr": "Sen Mekanik adında, Mekanik Kolaylıklar adlı elektronik/Arduino/mekanik öğrenme sitesinin "
          "yardımcı asistanısın. Dostça, kısa (en fazla 4-5 cümle) ve net cevap ver. Emin olmadığın "
          "teknik bilgi (pin numarası, gerilim gibi) verme; sitenin ilgili sayfasına bakmasını söyle. "
          "Elektronik, Arduino, kodlama ve sitenin kendisiyle ilgili sorulara odaklan; sohbet isteyen "
          "sorulara da sıcak ve kısa cevap ver. Zararlı, yasa dışı ya da müstehcen isteklere yardım etme.",
    "en": "You are Mekanik, the friendly assistant of the Mekanik Kolaylıklar electronics/Arduino/mechanics "
          "learning site. Answer briefly (max 4-5 sentences) and clearly. Don't state technical specifics "
          "(pin numbers, voltages) you're not sure about; point to the site's relevant page instead. Focus on "
          "electronics, Arduino, coding and the site itself; still reply warmly to casual chat. Refuse harmful, "
          "illegal or explicit requests.",
    "de": "Du bist Mekanik, der freundliche Assistent der Lernseite Mekanik Kolaylıklar für Elektronik, "
          "Arduino und Mechanik. Antworte kurz (max. 4-5 Sätze) und klar. Nenne keine technischen Details "
          "(Pins, Spannungen), bei denen du unsicher bist; verweise stattdessen auf die passende Seite. "
          "Lehne schädliche, illegale oder anzügliche Anfragen ab.",
    "fr": "Tu es Mekanik, l'assistant amical du site d'apprentissage Mekanik Kolaylıklar (électronique, "
          "Arduino, mécanique). Réponds brièvement (4-5 phrases max) et clairement. Ne donne pas de détails "
          "techniques (broches, tensions) dont tu n'es pas sûr ; renvoie plutôt vers la page du site. Refuse "
          "les demandes nuisibles, illégales ou explicites.",
}
SYS_DEFAULT = SYS["en"]

hits, global_hits = {}, []
lock = threading.Lock()


def rate_ok(ip):
    now = time.time()
    with lock:
        l = [t for t in hits.get(ip, []) if now - t < 60]
        global global_hits
        global_hits = [t for t in global_hits if now - t < 60]
        if len(l) >= RATE or len(global_hits) >= GLOBAL_RATE:
            hits[ip] = l
            return False
        l.append(now)
        global_hits.append(now)
        hits[ip] = l
        return True


def ask_gemini(q, x, lang, hist):
    sys_prompt = SYS.get(lang, SYS_DEFAULT)
    if x:
        sys_prompt += ("\n\nKullanıcının o an baktığı sayfa/proje: " if lang == "tr" else "\n\nUser's current page/project: ") + x
    contents = []
    for m in hist[-6:]:
        role = "user" if m.get("r") == "u" else "model"
        t = str(m.get("t", ""))[:400]
        if t:
            contents.append({"role": role, "parts": [{"text": t}]})
    contents.append({"role": "user", "parts": [{"text": q[:600]}]})
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": sys_prompt}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 300, "temperature": 0.6},
    }).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            j = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"gemini-http-{e.code}: {detail}")
    cands = j.get("candidates") or []
    if not cands:
        raise RuntimeError("no-candidates: " + json.dumps(j)[:200])
    parts = cands[0].get("content", {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("empty-response")
    return text[:900]


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", ALLOW)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        self._send(200, {"ok": True, "configured": bool(API_KEY)}) if self.path == "/health" else self._send(404, {"ok": False})

    def do_POST(self):
        if self.path != "/ask":
            return self._send(404, {"ok": False})
        ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
        if not rate_ok(ip):
            return self._send(429, {"a": ""})
        if not API_KEY:
            return self._send(200, {"a": ""})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 20000:
                return self._send(413, {"a": ""})
            j = json.loads(self.rfile.read(n))
            q = str(j.get("q", "")).strip()
            x = str(j.get("x", ""))
            lang = str(j.get("lang", "tr"))
            hist = j.get("h") or []
            if not isinstance(hist, list):
                hist = []
        except Exception:
            return self._send(400, {"a": ""})
        if not q:
            return self._send(200, {"a": ""})
        try:
            a = ask_gemini(q, x, lang, hist)
            self._send(200, {"a": a})
        except Exception as e:
            print("ask error:", e, flush=True)
            self._send(200, {"a": ""})  # site hazır cevaplara geri döner

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("Dinleniyor: :%d (gemini key set: %s)" % (PORT, bool(API_KEY)), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
