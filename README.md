# 👍Personalized Recommendation System with python

> 이 레포지토리는 인프런의 ' python을 이용한 개인화 추천시스템' 강의를 정리한 내용이 담겨져 있습니다. 
>
> * 강의자 : 거친 코딩

## The people who studied with me🤝

* 🤝형수[@oony97](https://github.com/oony97)
* 🤝유진[@youjin99](https://github.com/youjin99)
* 🤝진영
* 🤝채영
* 🤝채원
* 🤝민희

## 📒outline

* **참고자료** : 연세대학교 경영대학 임일 교수님의  'python을 이용한 개인화 추천시스템'
* **강의 목적** : 주요 개인화 추천 알고리즘의 작동원리를 이해하는 것
* **강의에서 주로 다룰 내용**
  * 개인화 추천 기술의 전반적인 개념(연속값)
  * 협업 필터링
  * 행렬요인화
  * 딥러닝의 추천 알고리즘
  * 하이브리드 추천 알고리즘
* **강의 환경** : `Google Colab` 사용
* **선수 요구 지식** : `python`, `numpy`, `pandas`, `keras`, `SGD`
* **데이터 참고** : 구글 드라이브 [🔗](https://drive.google.com/drive/folders/19gkcIYjA3EjoNrMp9mn8KnZutKoPYLmg?usp=sharing)

## 📑contents<a id="contents"></a>

### Section 0. OT[⤴️](#contents)

* 2/8(화) 강의 소개

### Section 1. 추천 시스템 소개  [👉](./Section01.md)  |  [⤴️](#contents) 

* 주요 추천 알고리즘
* 추천 시스템 적용 사례

### Section 2. 기본적인 추천 시스템 [👉](./Section02.md) | 코드자료[⌨️](./Recosys/Section02.ipynb) [⤴️](#contents)

* 데이터 읽기
* 인기제품 방식
* 추천 시스템의 정확도 측정
* 사용자 집단별 추천

### Section 3. 협업 필터링 추천 시스템 [👉](./Section03.md) | 코드자료[⌨️](./Recosys/Section03.ipynb)  [⤴️](#contents)

* 협업 필터링의 원리
* 유사도지표
* 기본 CF 알고리즘
* 이웃을 고려한 CF
* 최적의 이웃 크기 결정
* 사용자의 평가경향을 고려한 CF
* 그 외의 CF 정확도 개선 방법
* 사용자 기반 CF와 아이템 기반 CF
* 추천 시스템의 성과측정지표

### Section 4. Matrix Factorization(MF) 기반 추천[👉](./Section04.md) | 코드자료[⌨️](./Recosys/Section04.ipynb) [⤴️](#contents)

* Matrix Factorization(MF) 방식의 원리
* SGD(Stochastic Gradient Decent)를 사용한 MF 알고리즘
* SGD를 사용한 MF 기본 알고리즘
* train/test 분리 MF 알고리즘
* MF의 최적 파라미터 찾기
* MF와 SVD

### Section 5. Surprise 패키지 사용[👉](./Section05.md) | 코드자료[⌨️](./Recosys/Section05.ipynb) [⤴️](#contents)

* Surprise 기본 활용 방법
* 알고리즘 비교
* 알고리즘 옵션 지정
* 다양한 조건의 비교
* 외부 데이터 사용

### Section 6. 딥러닝을 사용한 추천 시스템[👉](./Section06.md) | 코드자료[⌨️](./Recosys/Section06.ipynb) [⤴️](#contents)

* Matrix Factorization(MF)을 신경망으로 변환하기
* Keras로 MF 구현하기
* 딥러닝을 적용한 추천 시스템
* 딥러닝 모델에 변수 추가하기

### 섹션 7. 하이브리드 추천 시스템[👉](./Section07.md) | 코드자료[⌨️](./Recosys/Section07.ipynb) [⤴️](#contents)

* 하이브리드 추천 시스템의 장점
* 하이브리드 추천 시스템의 원리
* 하이브리드 추천 시스템(CF와 MF의 결합)

### 섹션 8. 대규모 데이터의 처리를 위한 Sparse Matrix 사용[👉](./Section08.md)[⤴️](#contents)

* Sparse Matrix의 개념과 Python에서의 사용
* Sparse Matrix를 추천 알고리즘에 적용하기

### 섹션 9. 추천 시스템 구축에서의 이슈[👉](./Section09.md)[⤴️](#contents)

* 신규 사용자와 아이템(Cold Start Problem)
* 확장성(Scalability)
* 추천의 활용(Presentation)
* 이진수 데이터(Binary Data)의 사용
* 사용자의 간접 평가 데이터(Indirect Evaluation Data) 확보
