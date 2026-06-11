"""
Модуль кластеризации новостей по трём атрибутам:
- название компании
- местоположение
- отрасль
"""

import re
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np

try:
    from natasha import (
        NewsNERToponymExtractor, NewsMorphTagger, NewsSyntaxParser,
        Doc, Segmenter, MorphVocab
    )
    NATASHA_AVAILABLE = True
except ImportError:
    NATASHA_AVAILABLE = False
    print("Warning: Natasha not installed. Install with: pip install natasha")


class NewsClusterer:
    """Кластеризатор новостей на основе TF-IDF и DBSCAN"""

    # Словарь ключевых слов для определения отрасли (можно расширять)
    INDUSTRY_KEYWORDS = {
        'IT': ['it', 'software', 'программное обеспечение', 'облачные технологии', 'data center'],
        'finance': ['банк', 'финансы', 'кредит', 'инвестиции', 'страхование'],
        'retail': ['ритейл', 'магазин', 'торговля', 'e-commerce', 'маркетплейс'],
        'energy': ['энергетика', 'нефть', 'газ', 'электроэнергия', 'атомная'],
        'manufacturing': ['производство', 'завод', 'фабрика', 'промышленность'],
        'telecom': ['телеком', 'связь', 'оператор', 'интернет-провайдер'],
        'transport': ['транспорт', 'логистика', 'авиа', 'жд', 'доставка'],
    }

    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        """
        Args:
            eps: Максимальное расстояние между точками в одном кластере (DBSCAN)
            min_samples: Минимальное количество точек для формирования кластера
        """
        self.eps = eps
        self.min_samples = min_samples
        self.vectorizer = TfidfVectorizer(stop_words='russian')
        self._init_natasha()

    def _init_natasha(self):
        if NATASHA_AVAILABLE:
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            self.ner_tagger = NewsNERToponymExtractor(self.morph_vocab)
            self.morph_tagger = NewsMorphTagger(self.morph_vocab)
        else:
            self.segmenter = None

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Извлекает компании (ORG), локации (LOC) и отрасль.
        Возвращает словарь с ключами: companies, locations, industry
        """
        entities = {'companies': [], 'locations': [], 'industry': []}
        if not NATASHA_AVAILABLE or not self.segmenter:
            return entities

        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)
        for span in doc.spans:
            if span.type == 'ORG':
                entities['companies'].append(span.text)
            elif span.type == 'LOC':
                entities['locations'].append(span.text)

        # Определение отрасли по ключевым словам
        text_lower = text.lower()
        for industry, keywords in self.INDUSTRY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    entities['industry'].append(industry)
                    break
        # Уникальные значения
        for key in entities:
            entities[key] = list(set(entities[key]))
        return entities

    def build_feature_vector(self, news_list: List[Dict[str, Any]]) -> np.ndarray:
        """
        Строит TF-IDF вектор на основе объединённого текста
        (заголовок + описание + извлечённые сущности)
        """
        texts = []
        for news in news_list:
            title = news.get('title', '')
            description = news.get('description', '')
            entities = self.extract_entities(title + ' ' + description)
            # Добавляем сущности как отдельные слова для усиления
            entity_text = ' '.join(entities['companies'] + entities['locations'] + entities['industry'])
            combined = f"{title} {description} {entity_text}"
            texts.append(combined)
        return self.vectorizer.fit_transform(texts).toarray()

    def cluster(self, news_list: List[Dict[str, Any]]) -> List[int]:
        """
        Выполняет кластеризацию списка новостей.
        Возвращает список cluster_id для каждой новости (-1 означает шум/выброс)
        """
        if len(news_list) < self.min_samples:
            return [-1] * len(news_list)

        X = self.build_feature_vector(news_list)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        labels = clustering.fit_predict(X)
        return labels.tolist()


# Пример использования (для тестирования)
if __name__ == "__main__":
    sample_news = [
        {"title": "Газпром открыл новое месторождение в ЯНАО", "description": "Компания Газпром увеличила добычу газа"},
        {"title": "Сбербанк запустил платформу для бизнеса", "description": "Финансовый гигант представляет новые IT-решения"},
        {"title": "Ямальское месторождение принесло прибыль", "description": "Дочернее предприятие Газпрома отчиталось о рекордах"},
    ]
    clusterer = NewsClusterer(eps=0.5, min_samples=2)
    labels = clusterer.cluster(sample_news)
    print("Cluster labels:", labels)
    # Ожидаем: первые две новости в разных кластерах? (зависит от текста)