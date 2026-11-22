#!/usr/bin/env python3
"""Mekanik Kolaylıklar - yapay zeka asistanı sunucusu (Google Gemini).
POST /ask  {"q":"...", "x":"...", "lang":"tr", "h":[{"r":"u"|"a","t":"..."}]}  ->  {"a":"cevap"}
GET  /health
Sadece Python standart kütüphanesi gerekir (requests da gerekmez, urllib kullanılır)."""
import json, os, re, smtplib, threading, time, urllib.request, urllib.error
from email.mime.text import MIMEText
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8080"))
ALLOW = os.environ.get("ALLOW_ORIGIN", "*")
API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
RATE = int(os.environ.get("RATE_PER_MIN", "30"))
GLOBAL_RATE = int(os.environ.get("GLOBAL_PER_MIN", "180"))
TIMEOUT = int(os.environ.get("ASK_TIMEOUT", "25"))
MAX_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "900"))
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
NOTIFY_TO = os.environ.get("NOTIFY_TO", GMAIL_USER)

SYS = {
    "tr": "Sen Mekanik adında, Mekanik Kolaylıklar adlı elektronik/Arduino/mekanik öğrenme sitesinin "
          "yardımcı asistanısın. Dostça, kısa (en fazla 4-5 cümle) ve net cevap ver. Bildiğin elektronik/Arduino "
          "bilgisini çekinmeden anlat; yalnızca gerçekten emin olmadığın kart pin numarası ya da gerilim gibi "
          "ayrıntıyı uydurma, onun için sitenin ilgili sayfasına bakmasını söyle. "
          "Elektronik, Arduino, kodlama ve sitenin kendisiyle ilgili sorulara odaklan; sohbet isteyen "
          "sorulara da sıcak ve kısa cevap ver. Zararlı, yasa dışı ya da müstehcen isteklere yardım etme.",
    "en": "You are Mekanik, the friendly assistant of the Mekanik Kolaylıklar electronics/Arduino/mechanics "
          "learning site. Answer briefly (max 4-5 sentences) and clearly. Share what you know about electronics/Arduino "
          "confidently; only skip details you are truly unsure of (exact pin numbers, voltages) and point to the site instead. Focus on "
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

THINK_OPTS = [{"thinkingBudget": 0}, {"thinkingLevel": "minimal"}, None]
THINK_IDX = 0
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
    if lang not in SYS:
        sys_prompt += f"\nAlways answer in the user's language (language code: {lang})."
    if x:
        sys_prompt += ("\n\nKullanıcının o an baktığı sayfa/proje: " if lang == "tr" else "\n\nUser's current page/project: ") + x
    contents = []
    for m in hist[-6:]:
        role = "user" if m.get("r") == "u" else "model"
        t = str(m.get("t", ""))[:400]
        if t:
            contents.append({"role": role, "parts": [{"text": t}]})
    contents.append({"role": "user", "parts": [{"text": q[:600]}]})
    safety = [{"category": c, "threshold": "BLOCK_ONLY_HIGH"} for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

    def call(conts, temp):
        """Düşünmeyi kapatmayı dener (hız + boş cevap sorununu çözer). Model desteklemezse
        sıradaki ayara, o da olmazsa ayarsız isteğe düşer; çalışan ayar hatırlanır."""
        global THINK_IDX
        last = None
        for idx in range(THINK_IDX, len(THINK_OPTS)):
            gc = {"maxOutputTokens": MAX_TOKENS, "temperature": temp}
            if THINK_OPTS[idx] is not None:
                gc["thinkingConfig"] = THINK_OPTS[idx]
            body = json.dumps({"systemInstruction": {"parts": [{"text": sys_prompt}]},
                               "contents": conts, "generationConfig": gc,
                               "safetySettings": safety}).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    THINK_IDX = idx
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:400]
                last = RuntimeError(f"gemini-http-{e.code}: {detail}")
                if e.code == 400 and "think" in detail.lower() and idx + 1 < len(THINK_OPTS):
                    continue
                raise last
        raise last

    def text_of(j):
        c = (j.get("candidates") or [{}])[0]
        return "".join(p.get("text", "") for p in (c.get("content", {}).get("parts") or [])).strip(), c

    j = call(contents, 0.7)
    if not (j.get("candidates") or []):
        raise RuntimeError("no-candidates: " + json.dumps(j)[:200])
    text, cand = text_of(j)
    if not text:
        fr = cand.get("finishReason", "?")
        if fr == "RECITATION":
            retry_q = q + ("\n\n(Kendi cümlelerinle, kısaca özetleyerek anlat.)" if lang == "tr" else "\n\n(Summarize briefly in your own words.)")
            contents[-1] = {"role": "user", "parts": [{"text": retry_q[:650]}]}
            try:
                text, _ = text_of(call(contents, 0.9))
            except Exception:
                pass
        if not text:
            raise RuntimeError(f"empty-response: finishReason={fr}")
    return text[:1200]


def send_notify_email(kind, text, by, lang):
    if not (GMAIL_USER and GMAIL_APP_PASSWORD and NOTIFY_TO):
        raise RuntimeError("mail-not-configured")
    body = f"Tür: {kind}\nGönderen: {by or '(bilinmiyor)'}\nDil: {lang}\n\n{text}"
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = f"[Mekanik Kolaylıklar] Yeni öneri/şikayet: {kind}"
    msg["From"] = GMAIL_USER
    msg["To"] = NOTIFY_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.sendmail(GMAIL_USER, [NOTIFY_TO], msg.as_string())


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
        if self.path == "/health":
            return self._send(200, {"ok": True, "configured": bool(API_KEY), "mail_configured": bool(GMAIL_USER and GMAIL_APP_PASSWORD)})
        self._send(404, {"ok": False})

    def do_POST(self):
        if self.path == "/notify":
            return self._handle_notify()
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

    def _handle_notify(self):
        ip = (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()
        if not rate_ok(ip):
            return self._send(429, {"ok": False})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 10000:
                return self._send(413, {"ok": False})
            j = json.loads(self.rfile.read(n))
            kind = str(j.get("type", "Öneri"))[:60]
            text = str(j.get("text", ""))[:2000]
            by = str(j.get("by", ""))[:60]
            lang = str(j.get("lang", "tr"))[:10]
        except Exception:
            return self._send(400, {"ok": False})
        if not text.strip():
            return self._send(200, {"ok": False})
        try:
            send_notify_email(kind, text, by, lang)
            self._send(200, {"ok": True})
        except Exception as e:
            print("notify error:", e, flush=True)
            self._send(200, {"ok": False})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("Dinleniyor: :%d (gemini key set: %s)" % (PORT, bool(API_KEY)), flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
