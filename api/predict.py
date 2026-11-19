
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
