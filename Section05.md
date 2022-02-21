# 👍Section 05_ Surprise 패키지 사용[↩](../../)

## contents📑<a id='contents'></a>

* 0_ 들어가기 전에[✏️](#0)
* 

## 0_ 들어가기 전에[📑](#contents)<a id='0'></a>

```python
!pip install scikit-surprise 					# 일반적인 경우
!conda install -c cond-forge scikit-surprise	# 아나콘다의 경우
```

## 1_ Surprise 기본 활용 방법

* ml-100k : MovieLens 100k 데이터(앞에서 계속 사용해온 데이터)
* ml-1m : MovieLens 1m데이터(100만개)
* jester : 조크사이트 게시물(650 만개)
* `Surprise()`에서 예측 알고리즘 패키지 [🔗](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html)

| 알고리즘                                                     | 설명                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`random_pred.NormalPredictor`](https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) | Training set의 분포가 정규분포라고 가정한 상태에서 평점을 무작위로 추출하는 알고리즘. 일반적으로 성능이 안좋음. |
| [`baseline_only.BaselineOnly`](https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly) | 사용자의 평점평균과 아이템의 평점평균을 모델화해서 예측하는 알고리즘 |
| [`knns.KNNBasic`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic) | 3.4번째 강의에서 소개한 집단을 고려한 기본적인 CF 알고리즘   |
| [`knns.KNNWithMeans`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans) | 3.6번째 강의에서 소개한 사용자의 평가경향을 고려한 CF알고리즘 |
| [`knns.KNNWithZScore`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithZScore) | 사용자의 평가경향을 표준(정규분포)화시킨 CF알고리즘          |
| [`knns.KNNBaseline`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline) | 사용자의 평점평균과 아이템의 평점평균을 모델화 시킨 것(Baseline rating)을 고려한 CF 알고리즘 |
| [`matrix_factorization.SVD`](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) | 4.4번재 강의에서 설명한 MF알고리즘                           |
| [`matrix_factorization.SVDpp`](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp) | MF를 기반으로 사용자의 특정 아이템에 대한 평가여부를 이진값으로 일종의 암묵적 평가(Implicit ratings)로 추가한 SVD++알고리즘 |
| [`matrix_factorization.NMF`](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF) | 행렬의 값이 전부 양수일때 사용가능한 MF 알고리즘             |
| [`slope_one.SlopeOne`](https://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne) | 간단하면서도 정확도가 높은 것이 특징인 SlopeOne알고리즘을 적용한 Item-Based CF 알고리즘 |
| [`co_clustering.CoClustering`](https://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering) | 사용자와 아이템을 동시에 클러스터링하는 기법을 적용한 CF알고리즘 |

* 패키지 설치

  ```python
  !pip install scikit-surprise
  ```

* 패키지 사용하여 분석

  ```python
  import numpy as np
  
  from surprise import BaselineOnly, KNNWithMeans, SVD, SVDpp, Dataset, accuracy, Reader
  from surprise.model_selection import cross_validate, train_test_split
  
  data = Dataset.load_builtin(name='ml-100k')
  
  # train test 분리
  trainset, testset = train_test_split(data, test_size=0.25)
  
  algo = KNNWithMeans()
  
  algo.fit(trainset)
  
  prediction = algo.test(testset)
  
  accuracy.rmse(prediction)
  ```

  
