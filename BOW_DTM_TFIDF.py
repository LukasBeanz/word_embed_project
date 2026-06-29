# BOW_DTM_TFIDF.py

"""
BOW(Bag of Words)
: 단어의 순서는 고려하지 않고, 단어가 몇번 등장했는지에 집중하는 텍스트 수치화 방법임


DTM(Document-Term Matrix)
:여러 문서의 BOW 를 하나의 행렬로 모은 것

TF-IDF
: TF 는 특정 문서 안에서 단어가 얼마나 자주 등장하는지를 나타냄
- IDF 는 전체 문서에서 흔한 단어의 가중치는 낮추고, 흔하지 않은 단어의 가중치를 높임
"""
import sys
import subprocess
from math import log

try:
    from sklearn.feature_extraction.text import CountVectorizer

except ModuleNotFoundError as e:
    raise ModuleNotFoundError("scikit-learn is not installed. Please install it with 'pip install scikit-learn' !") from e


try:
    from konlpy.tag import Okt
    okt = Okt()
    KONLPY_AVAILABLE = True # KONLPY 사용 가능 상태
except Exception as e:
    okt = None  # okt 미사용 시 NameError 방지를 위해 None 으로 초기화
    KONLPY_AVAILABLE = False  # KONLPY 사용 불가 상태로 설정 → 이후 공백 기반 토큰화로 대체
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ModuleNotFoundError as e:
    nltk = None
    stopwords = None
    NLTK_AVAILABLE = False

# 공통 출력 함수 정의 =============================================
def print_section(title): # 실습 단계별 제목을 보기좋게 출력하기 위한 함수
    """ 콘솔에서 실습 구간을 구분하기 위한 제목 출력 함수 입니다."""
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)

# BOW 직접 구현 : 한국어 형태소 분석 기반 Bag Of Words 생성
print_section("1.BOW 구현: 한국어 문장을 단어 빈도 벡터로 변환")

if not KONLPY_AVAILABLE: # Konlpy 사용이 불가능하다면
    print('[안내] konlpy.Okt 실행에 실패했습니다.')
    print('[대체 실행] 현재 코드는 공백 기반 토큰화를 사용하여 계속 실행합니다.')

def tokenize_korean(document): # 한국어 문장을 토큰 리스트로 변환하는 함수
    """Okt 가 가능하면 형태소 분석을 사용하고, 불가능하면 공백 기준으로 토큰화합니다"""
    cleaned_document =  document.replace(".",'') # 마침표 제거 (불필요한 기호 제거 : 정제)
    cleaned_document = cleaned_document.replace(",",'') # 쉼표 제거
    if KONLPY_AVAILABLE:
        return okt.morphs(cleaned_document) # 형태소 단위로 분리해서 리턴
    return cleaned_document.split() # 공백 기준으로 단어들을 나눔 (토큰화)


def build_bag_of_words(document): # 하나의 문서에서 BOW 사전과 빈도 벡터를 만드는 함수
    """ 전달받은 문서에서 단어 인덱스 사전과 단어 빈도 벡터를 생성하는 함수입니다."""
    tokenize_document = tokenize_korean(document) # 문장을 토큰 단위로 나눈 결과 받음
    word_to_index = {} # 단어별 고유 인덱스를 저장할 빈 딕셔너리 생성
    bow = [] # 각 단어 인덱스 위치에 등장 회수 저장할 빈 리스트 생성
    for word in tokenize_document: # 토큰화된 단어를 하나씩 처리
        if word not in word_to_index: # 현재 사전에 등록되지 않은 단어라면
            word_to_index[word] = len(word_to_index) # word: index 로 저장
            bow.append(1) # 새 단어 등록에 대한 빈도 1을 추가 (index 순번을 맞춰서 저장함)

        else: # 현재 단어가 사전에 등록되어 있다면
            index = word_to_index[word] # 기존 등록된 단어의 순번을 조회함
            bow[index] += 1 # 해당 인덱스 위치의 빈도를 1 증가시킴
    return word_to_index, bow # 단어 인덱스 사전과 빈도 벡터 리턴함

# bow 사용 테스트 ------------------------------------------------------------------------

doc1 = '정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.'
doc2 = '소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.'

vocab1, bow1 = build_bag_of_words(doc1)
vocab2, bow2 = build_bag_of_words(doc2)

print_section("1-1. BOW 결과 출력")
print(f'문서1 단어 사전: {vocab1}')
print(f'문서1 BOW 벡터: {bow1}')
print(f'문서2 단어 사전: {vocab2}')
print(f'문서2 BOW 벡터: {bow2}')


# DTM (Document-Term Matrix) : scikit-learn CountVectorizer 이용 ========================
print_section("2. DTM 구현: CountVectorizer 로 여러 문서를 하나의 행렬로 표현")

corpus = [
    '먹고 싶은 사과',
    '먹고 싶은 바나나',
    '길고 노란 바나나 바나나',
    '저는 과일이 좋아요',
]

vector = CountVectorizer()
dtm = vector.fit_transform(corpus) # 문서를 DTM 행렬로 변환

print('단어 사전(feature_names):')
print(vector.get_feature_names_out()) # 학습된 단어 목록 출력

print('\nDTM 행렬 (밀집 행렬):')
print(dtm.toarray()) # 희소 행렬을 일반 배열로 변환해서 출력


# TF-IDF 직접 구현 ======================================================================
print_section("3. TF-IDF 직접 구현")

docs = [
    '먹고 싶은 사과',
    '먹고 싶은 바나나',
    '길고 노란 바나나 바나나',
    '저는 과일이 좋아요',
]

# TF(d,t) = 문서 d 에서 단어 t 가 등장하는 횟수
def tf(doc_tokens, term):
    return doc_tokens.count(term)

# IDF(t) = log( (문서 전체 수) / (단어 t 가 등장한 문서 수 + 1) ) + 1
def idf(docs_tokens, term):
    df = sum(1 for doc in docs_tokens if term in doc) # 단어가 등장한 문서 수
    return log(len(docs_tokens) / (df + 1)) + 1

# TF-IDF(d,t) = TF(d,t) × IDF(t)
def tfidf(doc_tokens, docs_tokens, term):
    return tf(doc_tokens, term) * idf(docs_tokens, term)

# 모든 문서 토큰화
docs_tokens = [tokenize_korean(doc) for doc in docs]

# 전체 어휘 집합 생성 (중복 제거)
vocab = sorted(set(word for doc in docs_tokens for word in doc))

print('전체 어휘:', vocab)
print()

# 각 문서별 TF-IDF 벡터 계산 및 출력
result = []
for i, doc_tokens in enumerate(docs_tokens):
    vec = [round(tfidf(doc_tokens, docs_tokens, term), 4) for term in vocab]
    result.append(vec)
    print(f'문서{i+1} TF-IDF: {vec}')


# scikit-learn TfidfVectorizer 로 검증 ==================================================
print_section("3-1. scikit-learn TfidfVectorizer 결과와 비교")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if KONLPY_AVAILABLE:
        # 형태소 분석 결과를 공백으로 합쳐서 TfidfVectorizer 에 전달
        tokenized_corpus = [' '.join(tokenize_korean(doc)) for doc in docs]
    else:
        tokenized_corpus = docs

    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(tokenized_corpus)

    print('단어 사전(feature_names):')
    print(tfidf_vec.get_feature_names_out())

    print('\nTF-IDF 행렬 (밀집 행렬):')
    print(tfidf_matrix.toarray().round(4))

except Exception as e:
    print(f'[오류] TfidfVectorizer 실행 실패: {e}')


# 불용어(Stopwords) 처리 ================================================================
print_section("4. 불용어(Stopwords) 처리: 분석에 불필요한 단어 제거")

# 불용어 : 분석에 도움이 되지 않는 조사, 접속사, 관사 등 자주 등장하지만 의미 없는 단어들
# 예) 영어: 'the', 'a', 'is', 'in' / 한국어: '이', '그', '저', '것', '수', '등'

# --- 4-1. 직접 정의한 한국어 불용어 사전 사용 ---
korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '및', '더', '를', '에', '의', '은', '는',
    '이다', '있다', '하다', '되다', '않다', '없다', '같다', '보다', '위해', '통해',
]

sample_doc = '자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리하는 기술이다.'

if KONLPY_AVAILABLE:
    tokens = okt.morphs(sample_doc, stem=True)
else:
    tokens = sample_doc.split()

filtered_tokens = [t for t in tokens if t not in korean_stopwords and len(t) > 1]

print(f'원본 토큰  : {tokens}')
print(f'불용어 제거: {filtered_tokens}')

# --- 4-2. NLTK 영어 불용어 사전 사용 ---
print()
if NLTK_AVAILABLE:
    try:
        nltk.download('stopwords', quiet=True)
        english_stopwords = set(stopwords.words('english'))

        english_doc = 'Natural language processing is the ability of a computer to understand human language.'
        eng_tokens = english_doc.lower().split()
        eng_filtered = [t for t in eng_tokens if t not in english_stopwords]

        print(f'[영어] 원본 토큰  : {eng_tokens}')
        print(f'[영어] 불용어 제거: {eng_filtered}')
    except Exception as e:
        print(f'[안내] NLTK 불용어 처리 실패: {e}')
else:
    print('[안내] NLTK 가 설치되어 있지 않아 영어 불용어 처리를 건너뜁니다.')
    print('       설치 명령: pip install nltk')
