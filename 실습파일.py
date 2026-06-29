# 실습파일.py
# 자연어 텍스트 데이터 전처리 및 워드 임베딩(Word Embedding) 실습
# 워드 임베딩 : 단어를 고정 크기의 실수 벡터로 표현하는 방법
# 비슷한 의미의 단어는 벡터 공간에서 가까운 위치에 놓이게 됨

import re
from konlpy.tag import Okt
from gensim.models import Word2Vec
from gensim.models import FastText
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ============================================================
# 0. 한글 폰트 설정 (matplotlib 한글 깨짐 방지)
# ============================================================

import os
if os.path.exists("C:/Windows/Fonts/malgun.ttf"):
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# ============================================================
# 1. 샘플 코퍼스 준비
# ============================================================

# 코퍼스(corpus) : 자연어 처리 모델 학습에 사용하는 텍스트 데이터 모음
corpus = [
    "자연어 처리는 인공지능의 중요한 분야이다.",
    "딥러닝은 자연어 처리 분야에서 혁신을 가져왔다.",
    "워드 임베딩은 단어를 벡터로 표현하는 방법이다.",
    "Word2Vec 모델은 단어 간의 유사도를 벡터로 학습한다.",
    "FastText 는 단어를 부분 문자열(n-gram)로 분리해서 학습한다.",
    "형태소 분석은 한국어 자연어 처리의 핵심 전처리 단계이다.",
    "문장을 형태소 단위로 분리하면 단어 간의 관계를 더 잘 파악할 수 있다.",
    "인공지능 모델은 대용량 데이터로 학습할수록 성능이 높아진다.",
    "자연어 처리 모델은 텍스트 분류, 번역, 요약 등에 활용된다.",
    "딥러닝 기반 언어 모델은 문장의 문맥을 함께 고려하여 의미를 파악한다.",
    "기계 번역은 자연어 처리 기술 중 가장 오래된 연구 분야 중 하나이다.",
    "챗봇은 자연어 처리와 딥러닝 기술이 결합된 대표적인 응용 사례이다.",
    "데이터 전처리는 모델 학습 전 반드시 수행해야 하는 단계이다.",
    "토큰화는 문장을 단어나 형태소 단위로 나누는 전처리 과정이다.",
    "불용어 제거는 분석에 불필요한 단어를 제거하는 전처리 방법이다.",
]

print("=" * 60)
print("코퍼스 샘플 (총 {}개 문장)".format(len(corpus)))
print("=" * 60)
for i, sent in enumerate(corpus[:3]):
    print(f"  [{i+1}] {sent}")
print("  ...")


# ============================================================
# 2. 형태소 분석 기반 전처리
# ============================================================

print("\n" + "=" * 60)
print("전처리: 형태소 분석 + 불용어 제거")
print("=" * 60)

okt = Okt()

# 분석에 의미 없는 불용어 목록
stopwords = ['이', '그', '저', '것', '수', '등', '및', '더', '를', '에', '의',
             '은', '는', '이다', '있다', '하다', '되다', '않다', '없다', '위해',
             '통해', '중', '가', '과', '와', '도', '로', '으로', '에서', '까지']

def preprocess(sentence):
    """문장 -> 특수문자 제거 -> 형태소 분석 -> 불용어 제거 -> 토큰 리스트 반환"""
    # 한글과 공백만 남기기
    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', sentence)
    # 형태소 분석 (명사, 동사, 형용사만 추출)
    tokens = okt.pos(sentence, stem=True)
    result = [word for word, pos in tokens
              if pos in ('Noun', 'Verb', 'Adjective') and word not in stopwords and len(word) > 1]
    return result

# 전체 코퍼스 토큰화
tokenized_corpus = [preprocess(sent) for sent in corpus]

print("\n전처리 결과 예시:")
for i in range(3):
    print(f"  원문  : {corpus[i]}")
    print(f"  토큰  : {tokenized_corpus[i]}")
    print()


# ============================================================
# 3. Word2Vec 모델 학습
# ============================================================
# Word2Vec : 주변 단어를 이용해 현재 단어의 의미를 학습하는 임베딩 모델
# - CBOW  : 주변 단어들로 현재 단어를 예측
# - Skip-gram : 현재 단어로 주변 단어들을 예측

print("=" * 60)
print("Word2Vec 모델 학습")
print("=" * 60)

w2v_model = Word2Vec(
    sentences=tokenized_corpus,  # 토큰화된 문장 리스트
    vector_size=100,             # 임베딩 벡터 차원 수
    window=5,                    # 주변 단어 범위 (앞뒤 5단어)
    min_count=1,                 # 최소 등장 횟수 (1 이상인 단어만 학습)
    workers=4,                   # 학습에 사용할 CPU 코어 수
    sg=1,                        # 0: CBOW, 1: Skip-gram
    epochs=200,                  # 반복 학습 횟수
)

print(f"학습된 단어 수: {len(w2v_model.wv)}")
print(f"임베딩 벡터 차원: {w2v_model.vector_size}")

# 단어 벡터 조회
target_word = '자연어'
if target_word in w2v_model.wv:
    vec = w2v_model.wv[target_word]
    print(f"\n'{target_word}' 벡터 (앞 10차원): {vec[:10].round(4)}")

# 유사 단어 조회
print(f"\n'{target_word}'와 가장 유사한 단어 Top 5:")
try:
    similar = w2v_model.wv.most_similar(target_word, topn=5)
    for word, score in similar:
        print(f"  {word:10s} (유사도: {score:.4f})")
except KeyError:
    print(f"  '{target_word}' 단어가 사전에 없습니다.")


# ============================================================
# 4. FastText 모델 학습
# ============================================================
# FastText : Word2Vec 의 확장판
# 단어를 n-gram(부분 문자열)으로 분리해서 학습 → 미등록 단어(OOV)에도 벡터 생성 가능

print("\n" + "=" * 60)
print("FastText 모델 학습")
print("=" * 60)

ft_model = FastText(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=200,
)

print(f"학습된 단어 수: {len(ft_model.wv)}")

# FastText 는 학습 데이터에 없는 단어도 n-gram 기반으로 벡터 추정 가능
oov_word = '딥러닝모델'  # 학습 데이터에 없는 단어
oov_vec = ft_model.wv[oov_word]
print(f"\nOOV 단어 '{oov_word}' 벡터 (앞 5차원): {oov_vec[:5].round(4)}")
print("(Word2Vec 은 OOV 단어에서 KeyError 발생하지만 FastText 는 처리 가능)")


# ============================================================
# 5. 단어 유사도 계산
# ============================================================

print("\n" + "=" * 60)
print("단어 간 코사인 유사도 비교")
print("=" * 60)

word_pairs = [
    ('자연어', '딥러닝'),
    ('자연어', '형태소'),
    ('딥러닝', '인공지능'),
]

for w1, w2 in word_pairs:
    try:
        sim = w2v_model.wv.similarity(w1, w2)
        print(f"  {w1:8s} ↔ {w2:8s} : {sim:.4f}")
    except KeyError as e:
        print(f"  {e} 단어가 사전에 없습니다.")


# ============================================================
# 6. 임베딩 벡터 시각화 (2D PCA)
# ============================================================
# 고차원(100차원) 벡터를 2차원으로 축소해서 단어 분포를 시각화

print("\n" + "=" * 60)
print("임베딩 벡터 시각화 (PCA 2차원 축소)")
print("=" * 60)

from sklearn.decomposition import PCA

# 시각화할 단어 목록
display_words = [w for w in w2v_model.wv.index_to_key if len(w) > 1][:20]

# 단어 벡터 행렬 구성
vectors = np.array([w2v_model.wv[w] for w in display_words])

# PCA 로 2차원 축소
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

# 산점도 출력
plt.figure(figsize=(12, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, s=60)

for i, word in enumerate(display_words):
    plt.annotate(word, (reduced[i, 0], reduced[i, 1]),
                 fontsize=11, ha='center', va='bottom')

plt.title("Word2Vec 임베딩 벡터 2D 시각화 (PCA)", fontsize=14)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n실습 완료!")
