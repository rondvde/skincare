import pandas as pd
import numpy as np
import os
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy

#### Setting working directory ####
# from google.colab import drive
# drive.mount('/content/drive')
# drive_path = '/content/drive/MyDrive/Scuola/Uni/Tesi'
# os.chdir(drive_path)

python -m spacy download en_core_web_lg
data = pd.read_excel("data/01_interim/data.xlsx")
eng_data = data[data["source"] == "Amazon"]

import spacy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# AMAZON ###########################
# 1. Load of the Spacy model and preparation of the text data
nlp = spacy.load("en_core_web_lg")
review_text = eng_data["review_text"].fillna("").astype(str)

# 2. Creation of Spacy docs and calculation of review lengths
docs = list(nlp.pipe(review_text, batch_size=50))
length_review = [len(doc) for doc in docs]

# 3. stopwords
stopword_extra = {
    "product", "bought", "purchased", "found", "used", "buy", "amazon", "review",
    "feel", "good", "like", "dry", "use", "love", "leave", "sensitive", "light",
    "oily", "look", "great", "soft", "greasy", "hydrate", "work", "perfect",
    "try", "super", "excellent", "hydrated", "heavy", "find", "little", "go",
    "make", "long", "quickly", "bit", "smooth", "calm", "nice", "apply", "need",
    "day", "time", "night", "morning", "evening", "month", "year", "week",
    "way", "thing", "stuff", "bit", "lot", "reason", "fact", "item",
    "people", "person", "daughter", "husband", "money", "price", "purchase",
    "moisturizing", "hydrating", "feeling", "overall", "tho", "one", "result",
    "lot", "hour"
}

words = []

# 4. Extraction of Nouns and Proper Nouns with filters
for doc in docs:
    for token in doc:
        # Filtriamo solo i Sostantivi (NOUN) e i Nomi Propri (PROPN)
        if token.pos_ in ["NOUN", "PROPN"]:

            lemma = token.lemma_.lower()

            # Applichiamo i filtri di pulizia
            if (not token.is_stop
                and not token.is_punct
                and token.is_alpha
                and len(token.text) > 2
                and lemma not in stopword_extra):

                words.append(lemma)

# 5. Calculatinf frequencies and creation of the WordCloud
counter = Counter(words)

wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="tab10",
    max_words=150
)
wc.generate_from_frequencies(dict(counter))

# 6. Drawing and saving
plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig("results/wordcloud_skincare_english_nouns.png", dpi=150, bbox_inches="tight")
plt.show()

# 7. Printing results
print("\nTop 50 Nouns:", counter.most_common(50))

# COUPANG ###########################
sudo apt-get install -y fonts-nanum
sudo fc-cache -fv
rm ~/.cache/matplotlib -rf
pip install konlpy

from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 0. Filter the data for Korean reviews
kor_data = data[data["source"] == "Coupang"]

# 1. Initialize the Okt tagger
okt = Okt()

# 2. Prepare the text data
reviews_kor_list = kor_data["review_text"].fillna("").astype(str).tolist()

# 3. K-Beauty Brands to Rescue (Translated to Korean)
brands_kor = [
    "에스트라", "조선미녀", "빌리프", "코스알엑스", "닥터지",
    "일리윤", "이즈앤트리", "클레어스", "라네즈", "믹순",
    "편강율", "라운드랩", "스킨천사", "스킨1004", "썸바이미",
    "순정", "설화수", "토리든", "아토팜"
]

# 4. Skincare Concepts to Rescue
rescue_words = [
    "세라마이드", "약산성", "가성비", "수딩젤", "마데카소사이드", "병풀추출물", "끈적거림",
    "레드블레미쉬", "속당김", "당김", "세럼", "붉은기", "안티에이징", "센텔라",
    "사용감", "수분기", "악건성", "유분기"
] + brands_kor

# 5. Your Ultimate Master Stopword List
stopword_extra_kor = {
    # --- General e-commerce and review noise ---
    "제품", "상품", "구매", "구입", "사용", "물건", "느낌", "도움",
    "쿠팡체험단", "이벤트", "작성", "부분", "정도", "리뷰", "후기", "제공",
    "무료", "조금", "완전", "한번", "장점", "진짜", "요즘", "처음",
    "분", "거", "것", "후", "날", "⭐", "=", "️", "♡", "✅", "수",
    "총평", "단점", "생각", "이유", "사실", "점", "때", "지금", "마지막", "사람", "속",
    "이", "저", "제", "더", "정말", "전", "번", "좀", "바로",
    "감", "형", "끈", "적임", "유", "단", "발라", "쓰기", "쿠팡", "체험",
    "추천", "효과", "부담", "유지", "용량", "가격", "타입", "케어", "화장품",
    "배송", "마음", "고민", "전혀", "보고", "편", "막", "주문", "포장", "선물",
    "개인", "돈", "시간", "기분", "상태", "테스트", "선택", "분위기", "단계",
    "통", "템", "의사", "만족도", "실제", "사서", "결론", "솔직", "대가", "신분", "판매", "정품",

    # --- Packaging and people noise ---
    "용기", "튜브", "뚜껑", "패키지", "디자인", "상자", "케이스",
    "가족", "엄마", "친구", "남편", "아들", "딸아이", "딸", "신랑", "주변", "집", "어른", "누구",

    # --- Adverbs pronouns filler and time words ---
    "안", "하나", "데", "거의", "편이", "게", "다음", "아주", "때문", "개", "용",
    "덕", "듬뿍", "살짝", "금방", "계속", "다른", "약간", "또", "중", "요", "걸",
    "다시", "예전", "뭐", "매우", "자주", "나", "꼭", "가장", "오히려", "평소",
    "모두", "이건", "일단", "이번", "무난", "전체", "꽤", "듯", "해", "위", "앞",
    "평", "부지", "항상", "그냥", "매일", "이제", "먼저", "시작", "벌써", "오늘",
    "며칠", "내내", "동안", "가끔", "무엇", "라면", "뭔가", "뒤", "늘", "두", "순간",
    "온", "바람", "경우", "수도", "원래", "더욱", "여러", "다만", "제일", "줄", "및",
    "만", "이상", "우선", "만큼", "몇", "일반", "크게", "보통", "내", "또한", "직접",
    "중간", "해도", "확", "별로", "싹", "그대로", "무조건", "다소", "막상", "아무", "대신",
    "가지", "종일", "양", "함유", "물", "세트", "거나", "대비", "추출", "워낙", "덕분", "일",

    # --- Single character grammatical dust ---
    "그", "건", "알", "도", "말", "과", "기", "덜", "니", "마", "난", "살", "시",
    "로", "쭉", "찬", "잔", "포", "면", "재", "애", "무", "세", "강", "열", "은",
    "팔", "구", "를", "보", "림", "움", "습", "링", "랩",

    # --- Broken fragments (from rescued words and brands) ---
    "이드", "라마", "일리", "아토", "독도", "딩", "가성", "베리", "닥터", "윤",
    "비", "카", "블", "라운드", "에스트", "거리", "약", "덧", "쉬", "플러스",
    "세라", "마이드", "산성", "수딩", "젤", "레드", "레미", "토리", "김", "럼",
    "은기", "이징", "센텔", "용감", "분기", "악", "천사", "조선", "미녀",
    "코스", "엑스", "이즈", "앤", "트리", "썸", "바이", "미", "편강"

    "아침", "저녁", "하루", "밤", "낮", "매일", "최고", "만족", "추천", "인생",
    "정착", "기대", "올리브영", "로켓", "광고", "할인", "손", "볼", "눈", "목",
    "코", "손등", "아이", "아기", "맘", "지인", "함", "임", "땐", "가성비",
    "사용감", "기존", "자체", "사진", "내용물", "기능", "가능", "구성"
}

nouns_kor = []

# 6. Extract nouns & apply rescue trick
print("Rescuing compound words and extracting nouns...")
for review in reviews_kor_list:

    # rescuing
    for safe_word in rescue_words:
        occurrences = review.count(safe_word)
        for _ in range(occurrences):
            nouns_kor.append(safe_word)

    # okt extraction
    for word in okt.nouns(review):
        if word not in stopword_extra_kor:
            nouns_kor.append(word)

# 7. Calculate frequencies
counter_nouns_kor = Counter(nouns_kor)

# 8. Generate WordCloud
font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

wc_nouns_kor = WordCloud(
    font_path=font_path,
    width=1200,
    height=600,
    background_color="white",
    colormap="tab10",
    max_words=150
)
wc_nouns_kor.generate_from_frequencies(dict(counter_nouns_kor))

# 9. Plot and save
plt.figure(figsize=(14, 7))
plt.imshow(wc_nouns_kor, interpolation="bilinear")
plt.axis("off")
plt.savefig("results/wordcloud_skincare_korean_NOUNS_konlpy.png", dpi=150, bbox_inches="tight")
plt.show()

# 10. Display stats
print("\nTop 50 Korean Nouns (KoNLPy):", counter_nouns_kor.most_common(50))

# AMAZON - ADJECTIVES ###########################

import spacy
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Loading the Spacy model and preparing the text data
nlp = spacy.load("en_core_web_lg")
review_text = eng_data["review_text"].fillna("").astype(str)

# 2. Creating Spacy docs and calculating review lengths
docs = list(nlp.pipe(review_text, batch_size=50))

# 3. stopwords
stopword_extra = {
    "good", "great", "nice", "perfect", "excellent", "amazing", "awesome", 
    "bad", "terrible", "super", "little", "much", "many", "more", "most", 
    "other", "such", "only", "few", "sure", "certain", "different", "real",
    "overall", "whole", "full", "same", "satisfied", "happy", "disappointed", 
    "glad", "scared", "surprised", "pleased", "delighted", "honest", "long", 
    "small", "high", "old", "new", "second", "huge", "big", "tiny", "large",
    "skeptical", "impressed", "worried", "entire", "multiple", "single", "double", "short", "past", 
    "immediate", "major", "well", "like", "non", "okay", "solid", "right", 
    "particular", "absolute", "valid", "similar", "able", "open", "wrong", 
    "true", "legit", "possible", "consistent", "fair", "social", "anti", 
    "bodied"
}

words = []

# 4. Extraction of adjectives with filters
for doc in docs:
    for token in doc:
        if token.pos_ == "ADJ":
            
            lemma = token.lemma_.lower()

            if (not token.is_stop
                and not token.is_punct
                and token.is_alpha
                and len(token.text) > 2
                and lemma not in stopword_extra):
                
                words.append(lemma)

# 5. Calculating frequencies and creating the WordCloud
counter = Counter(words)

wc = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    colormap="plasma",
    max_words=150
)
wc.generate_from_frequencies(dict(counter))

# 6. Drawing and saving
plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig("results/wordcloud_skincare_english_adjectives.png", dpi=150, bbox_inches="tight")
plt.show()

# 7. printing results
print("\nTop 50 Adjectives:", counter.most_common(50))

# COUPANG - ADJECTIVES ###########################
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Filter the data for Korean reviews
kor_data = data[data["source"] == "Coupang"]

# 2. Initialize the Okt tagger
okt = Okt()

# 3. Prepare the text data
reviews_kor_list = kor_data["review_text"].fillna("").astype(str).tolist()

# 4. Adjective-specific Stopwords
stopword_adj_kor = {
    "좋다", "괜찮다", "같다", "없다", "있다", "아니다", "그렇다", "이렇다", "저렇다", 
    "어떻다", "많다", "적다", "크다", "작다", "길다", "짧다", "다르다", "비슷하다",
    "아쉽다", "원하다", "필요하다", "이런", "그런", "저런", "어렵다", "쉽다",
    "만족하다", "이다", "만족스럽다", "확실하다", "꾸준하다", "빠르다", "심하다", 
    "충분하다", "유명하다", "편하다", "적당하다", "좋아하다", "저렴하다", 
    "넉넉하다", "가능하다", "강하다", "부족하다", "솔직하다", "딱이다", 
    "가깝다", "안되다", "중요하다", "기대하다", "싫다", "싫어하다", 
    "부탁드리다", "비싸다", "착하다", "뛰어나다", "신기하다", "아깝다", "속상하다", 
    "굉장하다", "특별하다", "예쁘다", "이쁘다", "바쁘다", "안녕하다", "급하다",
    "적합하다", "다양하다", "야하다", "단순하다", "적절하다", "밉다",
    "귀찮다", "힘들다", "고맙다", "놀랍다", "미치다"
}

adjectives_kor = []

# 5. Extract Adjectives with Stemming
print("Extracting and stemming adjectives...")
for review in reviews_kor_list:
    # okt.pos with stem=True converts conjugated words to their root form
    for word, pos in okt.pos(review, norm=True, stem=True):
        if pos == 'Adjective' and word not in stopword_adj_kor:
            adjectives_kor.append(word)

# 6. Calculate frequencies
counter_adj_kor = Counter(adjectives_kor)

# 7. Generate WordCloud
font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'

wc_adj_kor = WordCloud(
    font_path=font_path,
    width=1200,
    height=600,
    background_color="white",
    colormap="plasma",
    max_words=150
)
wc_adj_kor.generate_from_frequencies(dict(counter_adj_kor))

# 8. Plot and save
plt.figure(figsize=(14, 7))
plt.imshow(wc_adj_kor, interpolation="bilinear")
plt.axis("off")
plt.savefig("results/wordcloud_skincare_korean_ADJECTIVES_konlpy.png", dpi=150, bbox_inches="tight")
plt.show()

# 9. Display stats
print("\nTop 50 Korean Adjectives (KoNLPy Stemmed):", counter_adj_kor.most_common(50))
