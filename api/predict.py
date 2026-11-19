
import json
import re
import unicodedata
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
bundle = joblib.load(ROOT / "cosmetic_function_svm_final.joblib")


def normalize_name(name):
    name = unicodedata.normalize("NFKC", name)
    return re.sub(r"\s+", " ", name).strip().upper()


# Match formatting differences only. Do not guess chemical identities from spelling.
def name_key(name):
    return re.sub(r"[^A-Z0-9]", "", normalize_name(name))


name_index = {}
for registered_name in bundle['description_lookup']:
    name_index.setdefault(name_key(registered_name), set()).add(registered_name)


def registered_matches(text):
    matches = name_index.get(name_key(text), set())
    if not matches:
        # Explicit botanical common-name annotations seen on the supplied label.
        # Never remove general parentheses: they may be part of a chemical name.
        without_annotation = re.sub(r'\((?:OAT|ARTICHOKE)\)', '', text)
        matches = name_index.get(name_key(without_annotation), set())
    return sorted(matches)


def clean_piece(text):
    text = normalize_name(text)
    text = re.sub(r'^\s*[|.»~\\]+\s*', '', text)
    text = re.sub(r'^\d+\s+(?=[A-Z])', '', text)
    # Keep decimal/numeric ingredient prefixes such as 1,2-HEXANEDIOL.
    if re.search('[A-Z]', text):
        text = re.sub(r'\s*[^\x00-\x7F].*$', '', text).strip()
    return text.strip(' |.\\~')


def prepare_ocr_text(text):
    text = unicodedata.normalize('NFKC', text).upper()
    heading = re.search(r'\b(?:INGREDIENTS|INCI)\s*[:：]', text)
    if heading:
        text = text[heading.end():]
    # A physical OCR line is a soft boundary; commas/semicolons are hard boundaries.
    # Parentheses can be broken by OCR, so don't let unmatched brackets swallow the list.
    groups, current = [], ''
    for i, char in enumerate(text):
        numeric_comma = (char == ',' and i > 0 and i + 1 < len(text)
                         and text[i-1].isdigit() and text[i+1].isdigit())
        if char in ',;；，' and not numeric_comma:
            groups.append(current); current = ''
        else:
            current += char
    groups.append(current)
    records = []
    for group in groups:
        pieces = [clean_piece(line) for line in group.splitlines()]
        pieces = [p for p in pieces if p]
        i = 0
        while i < len(pieces):
            if pieces[i] == 'WATER' and 'AQUA' in bundle['description_lookup']:
                records.append({'source': 'WATER', 'name': 'AQUA',
                                'status': 'alias', 'suggestions': []})
                i += 1
                continue
            # Merge wrapped lines only if the full concatenation is a unique registered name.
            candidates = []
            for end in range(i+1, min(len(pieces), i+5)+1):
                combined = ' '.join(pieces[i:end])
                matches = registered_matches(combined)
                if len(matches) == 1:
                    candidates.append((end, combined, matches[0]))
            if candidates:
                end, source, canonical = candidates[-1]
                records.append({'source': source, 'name': canonical,
                                'status': 'matched' if source == canonical else 'normalized',
                                'suggestions': []})
                i = end
                continue
            source = pieces[i]; i += 1
            if not re.search('[A-Z가-힣]', source):
                continue
            suggestions = set(registered_matches(source))
            # Prefix noise may be present, but deleting words could change identity.
            # Suggest exact suffix matches for explicit user confirmation only.
            words = source.split()
            for cut in range(1, min(3, len(words))):
                suffix = ' '.join(words[cut:])
                if len(suffix) >= 8:
                    suggestions.update(registered_matches(suffix))
            records.append({'source': source, 'name': source,
                            'status': 'review', 'suggestions': sorted(suggestions)[:3]})
    unique = {}
    for record in records:
        if record['status'] != 'review' and not bundle['description_lookup'].get(record['name'], '').strip():
            record['status'] = 'description_unavailable'
        unique.setdefault(record['name'], record)
    return {'records': list(unique.values()),
            'names': [r['name'] for r in unique.values()
                      if r['status'] in {'matched', 'normalized', 'alias'}],
            'note': '등록명 또는 설명을 확인하지 못한 항목은 분석 목록에서 제외했습니다. 원문은 별도로 보존됩니다.'}


def analyze(names):
    names = list(dict.fromkeys(
        normalize_name(name) for name in names if name.strip()
    ))

    results = []

    for name in names:
        description = bundle["description_lookup"].get(name)

        if not description or not description.strip():
            results.append({
                "ingredient": name,
                "status": "description_unavailable",
                "message": "설명을 찾지 못해 예측을 보류합니다.",
                "functions": []
            })
            continue

        text = name + " " + description
        features = bundle["vectorizer"].transform([text])
        scores = bundle["model"].decision_function(features)[0]

        functions = []

        for label, score in zip(bundle["labels"], scores):
            functions.append({
                "label": label,
                "name": bundle["function_names"].get(label, label),
                "score": float(score),
                "predicted": bool(score >= bundle["threshold"])
            })

        results.append({
            "ingredient": name,
            "status": "predicted",
            "message": "학습한 SVM으로 예측했습니다.",
            "functions": functions
        })

    return results


class handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_json(200, {
            "message": "화장품 성분 기능 예측 API",
            "model": "TF-IDF + Linear SVM",
            "threshold": float(bundle["threshold"]),
            "samples": bundle.get("sample_ingredients", []),
            "sample_note": "검증 데이터에서 선정한 시연용 사례"
        })

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))

            if length <= 0 or length > 50000:
                self.send_json(400, {"error": "입력 크기를 확인해 주세요."})
                return

            payload = json.loads(self.rfile.read(length))
            if isinstance(payload, dict) and payload.get('action') == 'prepare_ocr':
                text = payload.get('text')
                if not isinstance(text, str) or not text.strip() or len(text) > 20000:
                    self.send_json(400, {'error': '정리할 원문을 20,000자 이내로 입력해 주세요.'})
                    return
                self.send_json(200, prepare_ocr_text(text))
                return
            names = payload.get("ingredients") if isinstance(payload, dict) else None

            if (
                not isinstance(names, list)
                or not 1 <= len(names) <= 100
                or not all(
                    isinstance(name, str) and 0 < len(name.strip()) <= 3000
                    for name in names
                )
            ):
                self.send_json(400, {
                    "error": "성분명을 문자열 목록으로 1~100개 입력해 주세요."
                })
                return

            self.send_json(200, {"results": analyze(names)})

        except (ValueError, UnicodeDecodeError):
            self.send_json(400, {"error": "입력 형식을 확인해 주세요."})
        except Exception:
            self.send_json(500, {"error": "모델 예측 처리 중 오류가 발생했습니다."})
