from fastapi import FastAPI
from clustering import NewsClusterer
from ranking import SentimentRanker

app = FastAPI(title="News Analyzer")

# Инициализация (можно с кэшированием, но для прототипа достаточно)
clusterer = NewsClusterer()
ranker = SentimentRanker()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze_news(news: list[dict]):
    """
    Ожидает список новостей с полями title, description.
    Возвращает кластеры и тональность.
    """
    clusters = clusterer.cluster(news)
    sentiments = [ranker.analyze(item.get("title", "") + " " + item.get("description", "")) for item in news]
    results = []
    for i, item in enumerate(news):
        results.append({
            "title": item.get("title"),
            "cluster_id": clusters[i],
            "sentiment": sentiments[i]
        })
    return {"results": results}