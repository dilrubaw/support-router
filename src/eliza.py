import re

REFLECTIONS = {
    "i": "you", "me": "you", "my": "your", "am": "are", "m": "are",
    "you": "I", "your": "my", "myself": "yourself",
}


def reflect(fragment):
    return " ".join(REFLECTIONS.get(w, w) for w in fragment.split())


RULES = [
    (r"^(hi|hello|hey|good morning|good evening)( there)?$",
     ["Hello! How can I help you with your appliance today?"]),
    (r"^(merhaba|selam|selamlar|günaydın|iyi günler|iyi akşamlar)$",
     ["Merhaba! Cihazınızla ilgili size nasıl yardımcı olabilirim?"]),
    (r"^((hi|hello) )?how are you( doing)?( today)?$",
     ["I'm fine, thank you! What can I do for you?"]),
    (r"^(merhaba |selam )?(nasılsın|nasılsınız|naber|ne haber)$",
     ["İyiyim, teşekkürler! Size nasıl yardımcı olabilirim?"]),
    (r"^(thanks|thank you|thank you very much|thanks a lot|many thanks)$",
     ["You're welcome! Anything else I can help with?"]),
    (r"^(teşekkürler|teşekkür ederim|çok teşekkürler|sağ ol|sağ olun|eyvallah)$",
     ["Rica ederim! Başka bir konuda yardımcı olabilir miyim?"]),
    (r"^(bye|goodbye|see you|see you later)$", ["Goodbye! Have a nice day."]),
    (r"^(görüşürüz|hoşça kal|hoşça kalın|iyi günler dilerim)$", ["Görüşmek üzere, iyi günler!"]),
    (r"^(who are you|are you a (bot|robot|human))$",
     ["I'm an automatic support assistant. Tell me about your appliance problem."]),
    (r"^(sen kimsin|bot musun|robot musun|insan mısın)$",
     ["Ben otomatik bir destek asistanıyım. Cihazınızdaki sorunu yazabilirsiniz."]),
    # Klasik ELIZA "yansıtma" kuralları
    (r"^i (?:am |m )?(?:feel|feeling) (.+)$", ["Why do you feel {0}?"]),
    (r"^i (?:am|m) (so |very )?(sad|angry|upset|bored|tired|frustrated|lonely|happy)(.*)$",
     ["How long have you been {1}{2}?"]),
    (r"^(?:çok )?(sıkıldım|yoruldum|sinirliyim|mutluyum|bunaldım)$",
     ["Neden {0} olduğunu biraz anlatır mısın?"]),
]
COMPILED = [(re.compile(p), r) for p, r in RULES]


def eliza_reply(normalized_text, has_domain_signal=False):
    """Mesaj sohbet/selamlaşma ise kural tabanlı cevap, değilse None döner."""
    if has_domain_signal:
        return None
    text = normalized_text.strip()
    for pattern, responses in COMPILED:
        m = pattern.match(text)
        if m:
            groups = [reflect(g.strip()) if g else "" for g in m.groups()]
            template = responses[len(text) % len(responses)]
            return template.format(*groups).replace("  ", " ")
    return None
