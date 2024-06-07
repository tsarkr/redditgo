import os
import pandas as pd
import json
import nltk
from transformers import pipeline
import langid
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy
from tqdm import tqdm

# NLTK 데이터 다운로드
nltk.download('punkt')
nltk.download('stopwords')

# Spacy 모델 로드 및 lemmatization 함수 정의
nlp = spacy.load("en_core_web_sm")

# 다중 언어 임베딩 모델 로드
multi_lang_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 감정분석 파이프라인 초기화
emotion_classifier_en = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')
emotion_classifier_ko = pipeline('text-classification', model='sangrimlee/bert-base-multilingual-cased-nsmc')
emotion_classifier_zh = pipeline('text-classification', model='uer/roberta-base-finetuned-jd-binary-chinese')
emotion_classifier_ja = pipeline('text-classification', model='cl-tohoku/bert-base-japanese')

# 전역 변수 설정
nations = ['korea', 'china', 'japan']
emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']

# 댓글과 답글을 구분하는 함수
def separate_post_comments_replies(data):
    posts, comments, replies = [], [], []

    def process_comments(comment_list, parent_id=None, post_id=None):
        for comment in comment_list:
            comment_data = {
                'author': comment.get('author'),
                'body': comment.get('body'),
                'created_utc': comment.get('created_utc'),
                'id': comment.get('id'),
                'parent_id': parent_id or comment.get('parent_id'),
                'post_id': post_id,
                'category': 'comment' if not parent_id else 'reply'
            }
            if parent_id:
                replies.append(comment_data)
            else:
                comments.append(comment_data)
                if 'replies' in comment and comment['replies']:
                    process_comments(comment['replies'], parent_id=comment['id'], post_id=post_id)

    post_metadata = data['data']['submission_metadata']
    post_data = {
        'author': post_metadata.get('author'),
        'body': post_metadata.get('selftext'),
        'created_utc': post_metadata.get('created_utc'),
        'id': post_metadata.get('id'),
        'parent_id': None,
        'post_id': post_metadata.get('id'),
        'category': 'post'
    }
    posts.append(post_data)

    process_comments(data['data']['comments'], post_id=post_metadata.get('id'))

    return posts, comments, replies

# POST, COMMENTS, REPLIES를 경로에서 읽어들여 하나의 DF로 생성하는 함수
def post_comments_replies(comm_path):
    posts, comments, replies = [], [], []

    for file_name in os.listdir(comm_path):
        file_path = os.path.join(comm_path, file_name)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            try:
                json_data = json.load(file)
                post, cmt, repl = separate_post_comments_replies(json_data)
                if isinstance(post, list):
                    posts.extend(post)
                    comments.extend(cmt)
                    replies.extend(repl)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {file_path}")

    df_posts = pd.DataFrame(posts)
    df_comments = pd.DataFrame(comments)
    df_replies = pd.DataFrame(replies)

    df = pd.concat([df_posts, df_comments, df_replies], ignore_index=True)
    print('posts + comments + replies = :'+df.count().to_string())

    return df

# 멀티스레딩을 사용한 언어 감지 함수
def detect_languages(texts):
    def detect_language(text):
        try:
            return langid.classify(text)[0]
        except:
            return 'unknown'

    with ThreadPoolExecutor(max_workers=128) as executor:
        future_to_text = {executor.submit(detect_language, text): text for text in texts}
        languages = []
        for future in as_completed(future_to_text):
            text = future_to_text[future]
            try:
                lang = future.result()
            except Exception as exc:
                lang = 'unknown'
            languages.append(lang)
    return languages

# 감정 분석 및 결과 반환 함수
def analyze_emotions_by_language(df, classifiers, max_length=256):
    def truncate_text(text, max_length):
        return text.encode('utf-8')[:max_length].decode('utf-8', errors='ignore')
    
    results = []
    texts_by_lang = {lang: (df[df['language'] == lang]['body'].tolist(), classifier) 
                     for lang, classifier in classifiers.items()}
    
    for lang, (texts, classifier) in texts_by_lang.items():
        for text in texts:
            truncated_text = truncate_text(text, max_length)
            result = classifier(truncated_text)[0]
            results.append({
                'text': truncated_text,
                'label': result['label'],
                'score': result['score']
            })
    
    return pd.DataFrame(results)

# 감정별 BERTopic 모델 생성 및 저장 함수
def create_bertopic_model(df_emotion, emotion, nation):
    texts = df_emotion['text'].tolist()
    if not texts:
        return None
    embeddings = multi_lang_model.encode(texts, show_progress_bar=True)
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    topic_model = BERTopic(language='multilingual', nr_topics='auto', calculate_probabilities=True, vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(texts, embeddings)
    topic_model.save(f'models/{nation}_{emotion}_bertopic_model')
    return topic_model

# 국가별로 데이터 처리
for nation in nations:
    comm_path = "URS/scrapes/" + nation + "/"

    # 데이터 로드 및 처리
    df = post_comments_replies(comm_path)

    # 언어 감지 수행
    df['language'] = detect_languages(df['body'].tolist())

    # 언어별 감정 분석기 사전
    classifiers = {
        'en': emotion_classifier_en,
        'ko': emotion_classifier_ko,
        'zh': emotion_classifier_zh,
        'ja': emotion_classifier_ja
    }

    # 감정 분석 수행
    emotion_df = analyze_emotions_by_language(df, classifiers)

    # 원래 데이터프레임에 감정 분석 결과 추가
    df.reset_index(drop=True, inplace=True)
    emotion_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, emotion_df], axis=1)

    # 결과 저장
    with open("models/reddit_" + nation + ".pickle", "wb") as f:
        pickle.dump(df, f)

    # 감정별 데이터 분리
    emotion_dfs = {label: df[df['label'] == label] for label in emotion_labels}

    # 모든 감정에 대해 BERTopic 모델 생성 및 저장
    for label in emotion_labels:
        create_bertopic_model(emotion_dfs[label], label, nation)
