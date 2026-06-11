"""
Модуль ранжирования (тональности) новости от негативной до позитивной.
Использует RuBERT или простой rule-based fallback.
"""

from typing import Dict, Tuple
import re

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install torch transformers")


class SentimentRanker:
    """Анализ тональности текста, возвращает score от -1 (негатив) до +1 (позитив)"""

    def __init__(self, model_name: str = "blanchefort/rubert-base-cased-sentiment"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

        # Словарь простых эмоциональных слов (fallback)
        self.positive_words = ['хорош', 'отличн', 'успех', 'рост', 'прибыль', 'выиграл', 'лауреат']
        self.negative_words = ['убыток', 'падени', 'кризис', 'штраф', 'иск', 'проблем', 'снижени']

    def _load_model(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Failed to load RuBERT model: {e}. Using fallback.")
                self.model = None

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Возвращает словарь:
        {
            'score': float от -1 до +1,
            'label': 'negative' / 'neutral' / 'positive',
            'confidence': float от 0 до 1 (если доступно)
        }
        """
        if not text or len(text.strip()) < 3:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.5}

        # Если модель загружена – используем её
        if self.model is not None and self.tokenizer is not None:
            return self._predict_with_model(text)
        else:
            return self._predict_rule_based(text)

    def _predict_with_model(self, text: str) -> Dict[str, float]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        # Предполагаем порядок классов: 0 - негатив, 1 - нейтраль, 2 - позитив (зависит от модели)
        # Для модели blanchefort/rubert-base-cased-sentiment: негатив=0, нейтраль=1, позитив=2
        neg, neu, pos = probs[0], probs[1], probs[2]
        score = (pos - neg)  # от -1 до +1
        if score > 0.3:
            label = 'positive'
        elif score < -0.3:
            label = 'negative'
        else:
            label = 'neutral'
        confidence = max(neg, neu, pos)
        return {'score': float(score), 'label': label, 'confidence': float(confidence)}

    def _predict_rule_based(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        total = pos_count + neg_count
        if total == 0:
            score = 0.0
        else:
            score = (pos_count - neg_count) / total
        # Ограничиваем от -1 до +1
        score = max(-1.0, min(1.0, score))
        if score > 0.3:
            label = 'positive'
        elif score < -0.3:
            label = 'negative'
        else:
            label = 'neutral'
        return {'score': score, 'label': label, 'confidence': 0.6}  # условная уверенность


# Пример использования
if __name__ == "__main__":
    ranker = SentimentRanker()
    test_texts = [
        "Компания Газпром сообщила о рекордной прибыли в этом году",
        "Сбербанк оштрафован на 10 млн рублей за нарушение прав потребителей",
        "Сегодня в Москве открылась конференция по цифровой экономике"
    ]
    for t in test_texts:
        result = ranker.analyze(t)
        print(f"Text: {t[:50]}... -> {result}")