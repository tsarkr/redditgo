import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import pickle

# Spacy 모델 로드 및 lemmatization 함수 정의
nlp = spacy.load("en_core_web_sm")

# 다중 언어 임베딩 모델 로드
multi_lang_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 감정분석 결과 파일 경로
pickle_files = {
    'korea': 'models/reddit_korea.pickle',
    'china': 'models/reddit_china.pickle',
    'japan': 'models/reddit_japan.pickle'
}

emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']

# Lemmatization 함수
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# 감정별 BERTopic 모델 생성 및 저장 함수
def create_bertopic_model(df_emotion, emotion, nation):
    texts = df_emotion['text'].tolist()
    if not texts:
        return None
    texts = [lemmatize_text(text) for text in texts]  # Lemmatization 적용
    embeddings = multi_lang_model.encode(texts, show_progress_bar=True)
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    topic_model = BERTopic(language='multilingual', nr_topics='auto', calculate_probabilities=True, vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(texts, embeddings)
    topic_model.save(f'models/lemma_{nation}_{emotion}_bertopic_model')
    return topic_model

# 국가별로 데이터 처리
for nation, file_path in pickle_files.items():
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    # 감정별 데이터 분리
    emotion_dfs = {label: df[df['label'] == label] for label in emotion_labels}

    # 모든 감정에 대해 BERTopic 모델 생성 및 저장
    for label in emotion_labels:
        create_bertopic_model(emotion_dfs[label], label, nation)