import os
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from groq import Groq

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"NLTK download failed: {e}")

class Summarizer:
    def __init__(self):
        self.groq_client = None
        self.model_name = "llama-3.1-8b-instant"
        
    def _init_groq(self):
        if self.groq_client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is missing")
            self.groq_client = Groq(api_key=api_key)

    def summarize_abstractive(self, text, max_length=150, min_length=50):
        self._init_groq()
        
        chunks = self._chunk_text(text, max_chunk_len=15000)
        summaries = []
        
        for chunk in chunks:
            prompt = f"Summarize the following text in an abstractive, human-like manner. Aim for around {max_length} words.\n\nText:\n{chunk}"
            
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional summarizer. Provide concise, clear, and accurate summaries directly without introductory phrases like 'Here is a summary'."
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=self.model_name,
                    temperature=0.3,
                    max_tokens=1024,
                )
                summary = chat_completion.choices[0].message.content
                summaries.append(summary.strip())
            except Exception as e:
                print(f"Groq API Error: {e}")
                summaries.append(f"[Summarization failed for this section: {str(e)}]")
            
        return " ".join(summaries)

    def summarize_extractive(self, text, ratio=0.3):
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return text
            
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return text
            
        # Calculate sentence scores based on TF-IDF
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Select top N sentences
        num_sentences = max(1, int(len(sentences) * ratio))
        top_sentence_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_sentence_indices.sort()
        
        summary = [sentences[i] for i in top_sentence_indices]
        return " ".join(summary)

    def _chunk_text(self, text, max_chunk_len=15000):
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback if sent_tokenize fails
            return [text[i:i+max_chunk_len] for i in range(0, len(text), max_chunk_len)]
            
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_len:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return [c for c in chunks if c.strip()]

# Singleton instance
summarizer_instance = Summarizer()
