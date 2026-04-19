"""
Hand-seeded Urdu POS lexicon (≥200 NOUN / VERB / ADJ) + NER gazetteers (CS-4063 Part 2).
Extended at runtime from corpus frequency in part2_dataset.py.
"""
from __future__ import annotations

# --- POS: closed-class and common function words (Urdu) ---
PRON_TOKENS = """
یہ وہ اس ان ہم تم آپ کوئی کچھ سب کسی کسی کو سے میں نے کو انھیں ان کو جس کیسا کیسی کیسے جہاں جب
کہاں کیوں کیسے خود اپنا اپنی اپنے
""".split()

DET_TOKENS = """
ایک کچھ ہر کوئی کئی بعض تمام
""".split()

CONJ_TOKENS = """
اور لیکن مگر تاہم یا تو نیز پھر سوائے
""".split()

POST_TOKENS = """
میں پر سے کو کا کی کے نے والا والی والے
""".split()

# Additional ADV / misc for lexicon lookup
ADV_TOKENS = """
بہت زیادہ کم جلدی دیر سے اب پھر ہمیشہ کبھی
ساتھ ساتھ پہلے بعد آگے پیچھے دوبارہ
""".split()

# Seed lists (Urdu) — expanded below via corpus in part2_dataset
NOUN_SEEDS = """
پاکستان انڈیا کرکٹ میچ ٹیم کھلاڑی حکومت وزیر عوام عدالت قانون
سیاست انتخاب پارلیمنٹ صوبہ شہر صدر وزرات معیشت بینک
روپیہ ڈالر تجارت بجٹ ہسپتال بیماری تعلیم اسکول یونیورسٹی
فوج فوجی جنگ امن سلامتی پولیس عدالت جج وکیل
زمین پانی ہوا آگ موسم بارش سیلاب زلزلہ
کھیل ورلڈ کپ سیریز ٹورنامنٹ فائنل رنز وکٹ
""".split()

VERB_SEEDS = """
ہے ہیں تھا تھے ہوگا ہوگی ہونا تھا ہے
کرنا گیا گیاں جاتے کہنا کہتے بتانا دینا لینا
آنا جانا دیکھنا سننا پڑھنا لکھنا ملنا
کھیلنا جیتنا ہارنا بنانا بننا
""".split()

ADJ_SEEDS = """
بڑا بڑی بڑے چھوٹا نیا پرانا اچھا برا
تیز سست لمبا موٹا پتلا
اہم ضروری ممکن ناممکن
""".split()

PUNC_CHARS = set("،۔؛:!؟()[]{}\"'«»٫٬٭….")


def merge_frequency_buckets(
    counter: dict[str, int],
    min_each: int = 200,
) -> dict[str, str]:
    """
    Build token->POS map with ≥ min_each NOUN/VERB/ADJ using heuristics + frequency.
    """
    from collections import Counter

    c = Counter(counter)
    noun: set[str] = set(NOUN_SEEDS)
    verb: set[str] = set(VERB_SEEDS)
    adj: set[str] = set(ADJ_SEEDS)

    for w, _ in c.most_common(12000):
        if w in ("<NUM>", "<PAD>", "<UNK>") or len(w) < 2:
            continue
        if len(verb) < min_each + 50 and w.endswith("نا"):
            verb.add(w)
        elif len(adj) < min_each + 80 and w.endswith("ی") and len(w) > 2:
            adj.add(w)
        elif len(noun) < min_each + 100:
            noun.add(w)

    # Ensure minimum sizes
    for w, _ in c.most_common(20000):
        if len(noun) >= min_each and len(verb) >= min_each and len(adj) >= min_each:
            break
        if w in ("<NUM>", "<PAD>", "<UNK>"):
            continue
        if len(verb) < min_each and (w.endswith("نا") or w.endswith("تے") or w.endswith("گی")):
            verb.add(w)
        elif len(adj) < min_each and w.endswith("ی"):
            adj.add(w)
        else:
            noun.add(w)

    token2pos: dict[str, str] = {}
    # Priority: VERB, ADJ, NOUN (infinitives and adjectives before default noun)
    for s, tag in (
        (verb, "VERB"),
        (adj, "ADJ"),
        (noun, "NOUN"),
    ):
        for w in s:
            if w not in token2pos:
                token2pos[w] = tag
    for w in PRON_TOKENS:
        token2pos[w] = "PRON"
    for w in DET_TOKENS:
        token2pos[w] = "DET"
    for w in CONJ_TOKENS:
        token2pos[w] = "CONJ"
    for w in POST_TOKENS:
        token2pos[w] = "POST"
    for w in ADV_TOKENS:
        token2pos[w] = "ADV"
    return token2pos


# --- NER gazetteers: one phrase per line (tokens split on whitespace) ---
PER_LINES = """
عمران خان
بابر اعظم
محمد رضوان
شاہد آفریدی
وسیم اکرم
یونس خان
مصباح الحق
سلمان بٹ
شعیب ملک
کامران اکمل
عبدالرزاق
محمد آصف
عمر گل
سہیل تنویر
نواز شریف
آصف زرداری
بلاول بھٹو
مریم نواز
شہباز شریف
قائداعظم
علامہ اقبال
ڈونلڈ ٹرمپ
جو بائیڈن
ولادیمیر پوتن
نریندر مودی
شی جن پنگ
عمران نذیر
محمد حفیظ
سرفراز احمد
حسن علی
شاہین آفریدی
عثمان خواجہ
رانا ثناء اللہ
مولانا فضل الرحمان
عاصم منیر
قمر جاوید باجوہ
راحیل شریف
اشفاق پرویز کیانی
عبدالستار ایدھی
بلقیس ایدھی
ملالہ یوسفزئی
حافظ سعید
مولانا طارق جمیل
وسیم
عمران
حفیظ
نذیر
سلمان
شعیب
مصباح
عمر
گل
""".strip().splitlines()

LOC_LINES = """
کراچی
لاہور
اسلام آباد
پشاور
کوئٹہ
حیدرآباد
ملتان
سندھ
پنجاب
خیبر پختونخوا
بلوچستان
گلگت
انڈیا
چین
افغانستان
ایران
امریکہ
برطانیہ
دریائے سندھ
دریائے جہلم
کشمیر
غزہ
فلسطین
یروشلم
رفح
خان یونس
جنوبی افریقہ
نیوزی لینڈ
آسٹریلیا
سوات
چترال
ہنزہ
سری نگر
ممبئی
دہلی
کابل
قندھار
تہران
دبئی
ابوظبی
لندن
نیو یارک
واشنگٹن
مقبوضہ کشمیر
آزاد کشمیر
راولپنڈی
سیالکوٹ
گوجرانوالہ
فیصل آباد
بہاولپور
ٹھٹھہ
بدین
سوات وادی
ہنزا وادی
دریائے چناب
دریائے راوی
خیبر درہ
بولان درہ
""".strip().splitlines()

ORG_LINES = """
پی سی بی
آئی سی سی
اقوام متحدہ
بی بی سی
جیو نیوز
اے آر وائی
پاکستان مسلم لیگ
نون
پیپلز پارٹی
تحریک انصاف
مسلم لیگ
آئی ایم ایف
ورلڈ بینک
سپریم کورٹ
ہائی کورٹ
نیب
ایف آئی اے
پی آئی اے
پاکستان ریلویز
پی ٹی وی
ڈان اخبار
جنگ گروپ
نیشنل بینک
اسٹیٹ بینک
ہب بینک
آئی ایس پی آر
فاٹا
القاعدہ
داعش
""".strip().splitlines()

MISC_LINES = """
کرکٹ ورلڈ کپ
چیمپئنز ٹرافی
ایشیا کپ
ٹی ٹوئنٹی
ون ڈے
""".strip().splitlines()


def gazetteer_phrases() -> list[tuple[tuple[str, ...], str]]:
    """Return list of (token_tuple, entity_type) for longest-match."""
    out: list[tuple[tuple[str, ...], str]] = []
    for label, lines in (
        ("PER", PER_LINES),
        ("LOC", LOC_LINES),
        ("ORG", ORG_LINES),
        ("MISC", MISC_LINES),
    ):
        for line in lines:
            toks = line.strip().split()
            if toks:
                out.append((tuple(toks), label))
    out.sort(key=lambda x: -len(x[0]))
    return out


def tag_sentence_ner(tokens: list[str], phrases: list[tuple[tuple[str, ...], str]]) -> list[str]:
    """BIO tagging for tokens (longest phrase wins)."""
    n = len(tokens)
    tags = ["O"] * n
    i = 0
    while i < n:
        matched = False
        for phrase, et in phrases:
            L = len(phrase)
            if i + L <= n and tuple(tokens[i : i + L]) == phrase:
                tags[i] = f"B-{et}"
                for k in range(1, L):
                    tags[i + k] = f"I-{et}"
                i += L
                matched = True
                break
        if not matched:
            i += 1
    return tags


def rule_pos(token: str, lex: dict[str, str]) -> str:
    """Lexicon first (caller), then rules, then UNK."""
    if token == "<NUM>":
        return "NUM"
    if len(token) == 1 and token in PUNC_CHARS:
        return "PUNC"
    if all(ch in PUNC_CHARS or ch in '٫٬' for ch in token) and len(token) <= 2:
        return "PUNC"
    if token in lex:
        return lex[token]
    if token.endswith("نا") and len(token) > 3:
        return "VERB"
    if token.endswith("ی") and len(token) > 2:
        return "ADJ"
    return "NOUN"
