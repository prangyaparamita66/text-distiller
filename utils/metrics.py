import textstat
from sklearn.feature_extraction.text import TfidfVectorizer

def get_metrics(original_text, summary_text):
    metrics = {
        "original_word_count": len(original_text.split()),
        "summary_word_count": len(summary_text.split()),
        "readability_score": textstat.flesch_reading_ease(summary_text),
        "reduction_percentage": round((1 - len(summary_text) / len(original_text)) * 100, 2) if len(original_text) > 0 else 0
    }
    return metrics

def extract_keywords(text, top_n=5):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top N indices sorted by tf-idf score
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[-top_n:][::-1]
        
        return [feature_names[i] for i in top_indices]
    except ValueError:
        return []
