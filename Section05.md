# ๐Section 05_ Surprise ํจํค์ง ์ฌ์ฉ[โฉ](../../)

## ๐contents<a id='contents'></a>

* 0_ ๋ค์ด๊ฐ๊ธฐ ์ ์[โ๏ธ](#0)
* 1_ Surprise ๊ธฐ๋ณธ ํ์ฉ ๋ฐฉ๋ฒ[โ๏ธ](#1)
* 2_ ์๊ณ ๋ฆฌ์ฆ ๋น๊ต[โ๏ธ](#2)
* 3_ ์๊ณ ๋ฆฌ์ฆ ์ต์ ์ง์ [โ๏ธ](#3)
* 4_ ๋ค์ํ ์กฐ๊ฑด์ ๋น๊ต[โ๏ธ](#4)
* 5_ ์ธ๋ถ ๋ฐ์ดํฐ ์ฌ์ฉ[โ๏ธ](#5)

## 0_ ๋ค์ด๊ฐ๊ธฐ ์ ์[๐](#contents)<a id='0'></a>

```python
!pip install scikit-surprise 					# ์ผ๋ฐ์ ์ธ ๊ฒฝ์ฐ
!conda install -c cond-forge scikit-surprise	# ์๋์ฝ๋ค์ ๊ฒฝ์ฐ
```

## 1_ Surprise ๊ธฐ๋ณธ ํ์ฉ ๋ฐฉ๋ฒ[๐](#contents)<a id='1'></a>

* ml-100k : MovieLens 100k ๋ฐ์ดํฐ(์์์ ๊ณ์ ์ฌ์ฉํด์จ ๋ฐ์ดํฐ)
* ml-1m : MovieLens 1m๋ฐ์ดํฐ(100๋ง๊ฐ)
* jester : ์กฐํฌ์ฌ์ดํธ ๊ฒ์๋ฌผ(650 ๋ง๊ฐ)
* `Surprise()`์์ ์์ธก ์๊ณ ๋ฆฌ์ฆ ํจํค์ง [๐](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html)

| ์๊ณ ๋ฆฌ์ฆ                                                     | ์ค๋ช                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`random_pred.NormalPredictor`](https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) | Training set์ ๋ถํฌ๊ฐ ์ ๊ท๋ถํฌ๋ผ๊ณ  ๊ฐ์ ํ ์ํ์์ ํ์ ์ ๋ฌด์์๋ก ์ถ์ถํ๋ ์๊ณ ๋ฆฌ์ฆ. ์ผ๋ฐ์ ์ผ๋ก ์ฑ๋ฅ์ด ์์ข์. |
| [`baseline_only.BaselineOnly`](https://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly) | ์ฌ์ฉ์์ ํ์ ํ๊ท ๊ณผ ์์ดํ์ ํ์ ํ๊ท ์ ๋ชจ๋ธํํด์ ์์ธกํ๋ ์๊ณ ๋ฆฌ์ฆ |
| [`knns.KNNBasic`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic) | 3.4๋ฒ์งธ ๊ฐ์์์ ์๊ฐํ ์ง๋จ์ ๊ณ ๋ คํ ๊ธฐ๋ณธ์ ์ธ CF ์๊ณ ๋ฆฌ์ฆ   |
| [`knns.KNNWithMeans`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans) | 3.6๋ฒ์งธ ๊ฐ์์์ ์๊ฐํ ์ฌ์ฉ์์ ํ๊ฐ๊ฒฝํฅ์ ๊ณ ๋ คํ CF์๊ณ ๋ฆฌ์ฆ |
| [`knns.KNNWithZScore`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithZScore) | ์ฌ์ฉ์์ ํ๊ฐ๊ฒฝํฅ์ ํ์ค(์ ๊ท๋ถํฌ)ํ์ํจ CF์๊ณ ๋ฆฌ์ฆ          |
| [`knns.KNNBaseline`](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline) | ์ฌ์ฉ์์ ํ์ ํ๊ท ๊ณผ ์์ดํ์ ํ์ ํ๊ท ์ ๋ชจ๋ธํ ์ํจ ๊ฒ(Baseline rating)์ ๊ณ ๋ คํ CF ์๊ณ ๋ฆฌ์ฆ |
| [`matrix_factorization.SVD`](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) | 4.4๋ฒ์ฌ ๊ฐ์์์ ์ค๋ชํ MF์๊ณ ๋ฆฌ์ฆ                           |
| [`matrix_factorization.SVDpp`](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp) | MF๋ฅผ ๊ธฐ๋ฐ์ผ๋ก ์ฌ์ฉ์์ ํน์  ์์ดํ์ ๋ํ ํ๊ฐ์ฌ๋ถ๋ฅผ ์ด์ง๊ฐ์ผ๋ก ์ผ์ข์ ์๋ฌต์  ํ๊ฐ(Implicit ratings)๋ก ์ถ๊ฐํ SVD++์๊ณ ๋ฆฌ์ฆ |
| [`matrix_factorization.NMF`](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF) | ํ๋ ฌ์ ๊ฐ์ด ์ ๋ถ ์์์ผ๋ ์ฌ์ฉ๊ฐ๋ฅํ MF ์๊ณ ๋ฆฌ์ฆ             |
| [`slope_one.SlopeOne`](https://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne) | ๊ฐ๋จํ๋ฉด์๋ ์ ํ๋๊ฐ ๋์ ๊ฒ์ด ํน์ง์ธ SlopeOne์๊ณ ๋ฆฌ์ฆ์ ์ ์ฉํ Item-Based CF ์๊ณ ๋ฆฌ์ฆ |
| [`co_clustering.CoClustering`](https://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering) | ์ฌ์ฉ์์ ์์ดํ์ ๋์์ ํด๋ฌ์คํฐ๋งํ๋ ๊ธฐ๋ฒ์ ์ ์ฉํ CF์๊ณ ๋ฆฌ์ฆ |

* ํจํค์ง ์ค์น

  ```python
  !pip install scikit-surprise
  ```

* ํจํค์ง ์ฌ์ฉํ์ฌ ๋ถ์

  ```python
  import numpy as np
  
  from surprise import BaselineOnly, KNNWithMeans, SVD, SVDpp, Dataset, accuracy, Reader
  from surprise.model_selection import cross_validate, train_test_split
  
  data = Dataset.load_builtin(name='ml-100k')
  
  # train test ๋ถ๋ฆฌ
  trainset, testset = train_test_split(data, test_size=0.25)
  
  algo = KNNWithMeans()
  
  algo.fit(trainset)
  
  prediction = algo.test(testset)
  
  accuracy.rmse(prediction)
  ```


## 2_ ์๊ณ ๋ฆฌ์ฆ ๋น๊ต[๐](#contents)<a id='2'></a>

| ์๊ณ ๋ฆฌ์ฆ       |                                                              |
| -------------- | ------------------------------------------------------------ |
| `BaselineOnly` | ์ฌ์ฉ์์ ํ์  ํ๊ท ๊ณผ ์์ดํ์ ํ์ ํ๊ท ์ ๋ชจ๋ธํ ํด์ ์์ธกํ๋ ์๊ณ ๋ฆฌ์ฆ |
| `KNNWithMeans` | ์ฌ์ฉ์์ ํ๊ฐ ๊ฒฝํฅ๊น์ง ๊ณ ๋ คํ CF                             |
| `SVD`          | mf ๊ธฐ๋ฐ ์๊ณ ๋ฆฌ์ฆ                                             |
| `SVDpp`        | mf  ๊ธฐ๋ฐ ์๊ณ ๋ฆฌ์ฆ์ ์ด์ง ๊ฐ์ผ๋ก ์ผ์ข์ ์๋ฌต์  ํ๊ฐ๊น์ง ์ถ๊ฐํด์ ๊ณ ๋ คํ ์๊ณ ๋ฆฌ์ฆ |

* ๋น๊ต ๊ตฌํ

  ```python
  #๋น๊ต์ ํ์ํ Surprise ์๊ณ ๋ฆฌ์ฆ
  from surprise import BaselineOnly
  from surprise import KNNWithMeans
  from surprise import SVD
  from surprise import SVDpp
  
  # ์ ํ๋ ์ธก์  ๊ด๋ จ ๋ชจ๋์ ๊ฐ์ ธ์จ๋ค.
  from surprise import accuracy
  
  # Dataset๊ด๋ จ ๋ชจ๋์ ๊ฐ์ ธ์จ๋ค.
  from surprise import Dataset
  
  # train/test set ๋ถ๋ฆฌ ๊ด๋ จ ๋ชจ๋์ ๊ฐ์ ธ์จ๋ค.
  from surprise.model_selection import train_test_split
  
  # ๊ฒฐ๊ณผ๋ฅผ ๊ทธ๋ํ๋ก ํ์ํ๊ธฐ ์ํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ
  import matplotlib.pyplot as plt
  
  # MovieLens 100k ๋ฐ์ดํฐ ๋ถ๋ฌ์ค๊ธฐ
  data = Dataset.load_builtin(name=u'ml-100k')
  
  # train/test 0.75 : 0.25๋ก ๋ถ๋ฆฌ
  
  trainset, testset = train_test_split(data, test_size=0.25)
  
  algorithms = [BaselineOnly, KNNWithMeans, SVD, SVDpp]
  
  names = []
  results = []
  
  for option in algorithms:
      algo = option()
      names.append(option.__name__)
      algo.fit(trainset)
      predictions = algo.test(testset)
      results.append(accuracy.rmse(predictions))
  names = np.array(names)
  results = np.array(results)
  
  index = np.argsort(results)
  plt.ylim(0.8, 1)
  plt.plot(names[index], results[index])
  results[index]
  
  # ์คํ ๊ฒฐ๊ณผ
  Estimating biases using als...
  RMSE: 0.9420
  Computing the msd similarity matrix...
  Done computing similarity matrix.
  RMSE: 0.9512
  RMSE: 0.9345
  RMSE: 0.9208
  array([0.92080759, 0.9345095 , 0.941986  , 0.95123484])
  ```

  ![](./image/5_2-1.png)

## 3_ ์๊ณ ๋ฆฌ์ฆ ์ต์ ์ง์ [๐](#contents)<a id='3'></a>

```python
# ์๊ณ ๋ฆฌ์ฆ ์ต์์ ๋ํด ๋์๋๋ฆฌ ํํ๋ก ์ ์ฅ
sim_options = {'name': 'pearson_baseline', # name์๋ค๊ฐ ์ ์ฌ๋ ์งํ์ ์ข๋ฅ๋ฅผ ์ค์ 
               'user_based': True}          # True๋ ์ ์  ๋ฒ ์ด์ค CF
algo = KNNWithMeans(k=30, sim_options=sim_options)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# ์คํ ๊ฒฐ๊ณผ
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
RMSE: 0.9399
0.9399320564069582
```



![](./image/5_3-1.png)

| ์ ์ฌ๋ ์งํ            | ์ค๋ช                                                         |
| ---------------------- | ------------------------------------------------------------ |
| `msd`                  | ๋ ์ฌ์ฉ์๊ฐ ๊ณตํต์ผ๋ก ํ๊ฐํ๋ ์์ดํ์ ์ด ๊ฐฏ์์ ๊ณตํต ํ๊ฐ ์ํ์ ๋ํ ์์ดํ์ ํ์  ์ฐจ์ด๋ฅผ ๊ณ์ฐํจ. |
| `cosine_sim`           | ์ฌ์ฉ์์๋ ์์ดํ์ ์ฐพ์๋, ๊ต์งํฉ ์์์๋ง ๋น๊ต             |
| `pearson_sim`          | ๋ ๋ฒกํฐ์ ์๊ด๊ณ์<br />์๋์ ์ธ ๋ฐฉ์์ ์ ์ฌ๋               |
| `pearson_baseline_sim` | ๋ฒ ์ด์ค๋ผ์ธ์์ ์์ธกํ ๋ฒ ์ด์ค ๊ฐ์ ๋นผ์ค.                      |

## 4_ ๋ค์ํ ์กฐ๊ฑด์ ๋น๊ต[๐](#contents)<a id='4'></a>

```python
# ์ง๋จ๊ณผ ์ฌ์ฉ์์ ํ๊ฐ๊ฒฝํฅ์ ํจ๊ป ๊ณ ๋ คํ CF ์๊ณ ๋ฆฌ์ฆ
from surprise import KNNWithMeans

# Dataset๊ด๋ จ ๋ชจ๋์ ๊ฐ์ ธ์จ๋ค.
from surprise import Dataset

# ์ ํ๋ ์ธก์  ๊ด๋ จ ๋ชจ๋์ ๊ฐ์ ธ์จ๋ค.
from surprise import accuracy

# train/test set ๋ถ๋ฆฌ ๊ด๋ จ ๋ชจ๋์ ๊ฐ์ ธ์จ๋ค.
from surprise.model_selection import train_test_split

data = Dataset.load_builtin(name=u'ml-100k')

# train/test 0.75 : 0.25๋ก ๋ถ๋ฆฌ
trainset, testset = train_test_split(data, test_size=0.25)

result = []

for neighbor_size in (10, 20, 30, 40, 50, 60):
    algo = KNNWithMeans(k=neighbor_size, 
                        sim_options={'name':'pearson_baseline', 
                        'user_based': True})
    algo.fit(trainset)
    predictions = algo.test(testset)
    result.append([neighbor_size, accuracy.rmse(predictions)])

result

# ์คํ ๊ฒฐ๊ณผ
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
RMSE: 0.9613
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
RMSE: 0.9481
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
RMSE: 0.9456
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
RMSE: 0.9454
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
RMSE: 0.9455
Estimating biases using als...
Computing the pearson_baseline similarity matrix...
Done computing similarity matrix.
RMSE: 0.9458
[[10, 0.9613315707363571],
 [20, 0.9480987237220185],
 [30, 0.9456218907663596],
 [40, 0.9453857146472053],
 [50, 0.9455253980578735],
 [60, 0.9457836179502214]]
```

* ์ผ์ผ์ด ํ์ดํผํ๋ผ๋ฏธํฐ๋ฅผ ์ฐพ๋ ๊ฒ์ด ์ฐพ๋ ๊ฒ์ด ์ด๋ ค์...

* ์ด ์ด๋ ค์์ ํด๊ฒฐํ๊ธฐ ์ํ `GridSearchCV`

* KNN ๋ค์ํ ํ๋ผ๋ฏธํฐ ๋น๊ต ๊ตฌํ

  ```python
  # Grid Search๋ฅผ ์ํ ๋ชจ๋ ๊ฐ์ ธ์ค๊ธฐ
  from surprise.model_selection import GridSearchCV
  param_grid = {
      'k': [5, 10, 15, 25],
      'sim_options': {'name': ['pearson_baseline', 'cosine'],     # ๋ฆฌ์คํธ ํํ๋ก ๋ง๋ฆ.
                      'user_based': [True, False]
      }
  }
  
  gs = GridSearchCV(KNNWithMeans,
                  param_grid, 
                  measures=['rmse'], 
                  cv=4)                   # cv๋?๋ฐ์ดํฐ๋ฅผ ๋ช๊ฐ์ ์ธํธ๋ก ๋๋์ด ํฌ๋ก์ค ๋ฐธ๋ฅ์์ด์ ํ ๊ฒ์ธ์ง? ์ ํ๋๋ฅผ 4๋ฒ ๊ณ์ฐํ์ฌ ํ๊ท ์ ๊ณ์ฐํจ.
  
  gs.fit(data)
  
  # ์คํ ๊ฒฐ๊ณผ
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Estimating biases using als...
  Computing the pearson_baseline similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  Computing the cosine similarity matrix...
  Done computing similarity matrix.
  ```

* SVD ๋ค์ํ ํ๋ผ๋ฏธํฐ ๋น๊ต

  ```python
  from surprise import SVD
  from surprise.model_selection import GridSearchCV
  
  param_grid = {
      'n_epochs': [70, 80, 90],
      'lr_all': [0.005, 0.006, 0.007],
      'reg_all': [0.05, 0.07, 0.1]
  }
  gs = GridSearchCV(algo_class = SVD,
                  param_grid = param_grid,
                  measures=['rmse'],
                  cv=4)
  gs.fit(data)
  
  print(gs.best_score('rmse'))
  
  print(gs.best_params['rmse'])
  ```

## 5_ ์ธ๋ถ ๋ฐ์ดํฐ ์ฌ์ฉ[๐](#contents)<a id='5'></a>

```python
# csv ํ์ผ์์ ๋ถ๋ฌ์ค๊ธฐ
import pandas as pd
# ๋ฐ์ดํฐ ์ฝ๊ธฐ ๊ด๋ จ๋ ๋ชจ๋์ ๊ฐ์ ธ์จ๋ค.
from surprise import Reader
# Dataset ๊ด๋ จ ๋ชจ๋์ ๊ฐ์ ธ์จ๋ค.
from surprise import Dataset

# DataFrame ํํ๋ก ๋ฐ์ดํฐ๋ฅผ ์ฝ์ด์จ๋ค.
r_cols = ['user_id', 'movie_id', 'rating','timestamp']
ratings = pd.read_csv('./Data/u.data',
                        names=r_cols,
                        sep='\t',
                        encoding='latin-1')

reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)
```

