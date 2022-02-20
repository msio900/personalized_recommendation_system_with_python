# 👍Section 04_ Matrix Factorization(MF) 기반 추천[↩](../../)

## contents📑<a id='contents'></a>

* 0_ 들어가기 전에[✏️](#0)
* 1_ Matrix Factorization(MF)방식의 원리[✏️](#1)
* 2_ SQD(Stichastic Gradient Decent)를 사용한 MF 알고리즘[✏️](#2)

## 0_ 들어가기 전에[📑](#contents)<a id='0'></a>

|      | 메모리 기반 알고리즘                          | 모델 기반 알고리즘                                   |
| ---- | --------------------------------------------- | ---------------------------------------------------- |
| 설명 | 메모리에 있는 데이터를 계산해서 추천하는 방식 | 데이터로부터 미리 모델을 구성후 필요시 추천하는 방식 |
| 특징 | 개별 사용자 데이터 집중                       | 전체 사용자 패턴 집중                                |
| 장점 | 원래 데이터에 충실하게 사용                   | 대규모 데이터에 빠르게 반응                          |
| 단점 | 대규모 데이터에 느리게 반응                   | 모델 생성 과정 오래 걸림                             |

## 1_ Matrix Factorization(MF)방식의 원리[📑](#contents)<a id='1'></a>

![](./image/4_1-1.png)

* R = N *M

* P = M * K

* Q<sup>T</sup> = N *K

* 사용자와 아이템의 특성을 k개의 잠재 요인을 사용해서 분석하는 모델

* 예시

  |        | 액션 - 드라마<br />(-1~1) | 판타지-사실주의<br />(-1~1) |
  | :----: | :-----------------------: | :-------------------------: |
  | User 1 |           -0.43           |            0.21             |
  | User 2 |           0.31            |            0.92             |
  | User 3 |           0.69            |            -0.03            |
  | User 4 |           0.49            |            -0.3             |

* 아이템 요인에 대해서 해석 가능

  |         | 액션 - 드라마<br />(-1~1) | 판타지-사실주의<br />(-1~1) |
  | ------- | ------------------------- | --------------------------- |
  | Movie 1 | 0.31                      | 0.6                         |
  | Movie 2 | 0.61                      | -0.82                       |
  | Movie 3 | -0.38                     | -0.61                       |
  | Movie 4 | -0.79                     | 0.08                        |

* 2차원 공간에 배치하면

  ![](./image/4_1-2.png)

* 사용자의 영화별 예측 평점

  * ![](./image/4_1-3.png)

  |        | Movie 1 | Movie 2 | Movie 3 | Movie 4 |
  | ------ | ------- | ------- | ------- | ------- |
  | User 1 | -0.0073 | -0.4345 | 0.0353  | 0.3565  |
  | User 2 | 0.6481  | -0.5653 | -0.679  | -0.1713 |
  | User 3 | 0.1959  | 0.4455  | -0.2439 | -0.5475 |
  | User 4 | -0.0374 | 0.5266  | 0.0082  | -0.3874 |

## 2_ SQD(Stichastic Gradient Decent)를 사용한 MF 알고리즘[📑](#contents)<a id='2'></a>

![](./image/4_2-1.png)

![](./image/4_2-2.png)

![](./image/4_2-3.png)

* 예상값

* 예측 오차
* 예측오차의 제곱을 하고 p와 q에 대해서 편미분을 함.
* 다시 활용을 해서 p, q를 업데이트함. 
* 알파는 얼만큼의 가중치를 둘 것인지?

![](./image/4_2-4.png)

![](./image/4_2-5.png)

* 오버피팅을 어떻게 줄이는가?
  * 정규화 행을 어떻게 줄이는가?
  * weight만 주게 되면 너무 잘 

## 3_ SGD를 사용한 MF 기본 알고리즘[📑](#contents)<a id='3'></a>

* 세팅

  ```python
  import os
  import numpy as np
  import pandas as pd
  
  base_src =  './Data'
  u_data_src = os.path.join(base_src, 'u.data')
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv(u_data_src,
                          sep='\t',
                          names=r_cols,
                          encoding='latin-1')
  
  # timestamp 제거
  ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)
  ```

* 구현

  ```python
  class MF():
      def __init__(self, ratings, hyper_params):
          self.R = np.array(ratings)
          self.num_users, self.num_items = np.shape(self.R)
          self.K = hyper_params['K']
          self.alpha = hyper_params['alpha']
          self.beta = hyper_params['beta']
          self.iterations = hyper_params['iterations']
          self.verbose = hyper_params['verbose']
  
      def rmse(self):
          xs, ys = self.R.nonzero()
          self.predictions = []
          self.errors = []
  
          for x, y in zip(xs, ys):
              prediction = self.get_prediction(x, y)
              self.predictions.append(prediction)
              self.errors.append(self.R[x, y] - prediction)
          self.predictions = np.array(self.predictions)
          self.errors = np.array(self.errors)
  
          return np.sqrt(np.mean(self.errors**2))
  
      def train(self):
          self.P = np.random.normal(scale=1./self.K,                  # scale = 표준편차를 이야기함
                                      size=(self.num_users, self.K))  # size 실제 유저수와 잠재요인의 갯수 = 크기 값
          self.Q = np.random.normal(scale=1./self.K,
                                      size = (self.num_items, self.K))
  
          self.b_u = np.zeros(self.num_users)
          self.b_d = np.zeros(self.num_items)
          self.b = np.mean(self.R[self.R.nonzero()])
  
          rows, columns = self.R.nonzero()
          self.samples = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]
  
          training_process = []
          for i in range(self.iterations):
              np.random.shuffle(self.samples)
              self.sgd()
              rmse = self.rmse()
              training_process.append((i+1, rmse)) # 몇번 째의 RMSE인지?
              if self.verbose:
                  if (i+1) % 10 == 0:
                      print('Iteration : %d ; train RMSE = %.4f'%(i + 1, rmse))
          return training_process
  
      def get_prediction(self, i, j):
          prediction = self.b + self.b_u[i] +self.b_d[j] + self.P[i, :].dot(self.Q[j,].T)
          return prediction
      def sgd(self):
          for i, j, r in self.samples:
              prediction = self.get_prediction(i, j)
              e = (r-prediction)
  
              self.b_u[i] += self.alpha * (e - (self.beta * self.b_u[i]))
              self.b_d[j] += self.alpha * (e - (self.beta * self.b_d[j]))
  
              self.P[i,:] += self.alpha * ((e * self.Q[j,:] - (self.beta * self.P[i, :])))
              self.Q[j,:] += self.alpha * ((e * self.Q[i,:] - (self.beta * self.Q[j, :])))
  
  R_temp = ratings.pivot(index='user_id',\
                          columns='movie_id',
                          values='rating').fillna(0)
  
  hyper_params = {
      'K' : 30,
      'alpha' : 0.001,
      'beta' : 0.02,
      'iterations' :100,
      'verbose' : True
  }
  
  mf = MF(R_temp, hyper_params)
  
  train_process = mf.train()
  
  # 실행 결과
  Iteration : 10 ; train RMSE = 0.9588
  Iteration : 20 ; train RMSE = 0.9380
  Iteration : 30 ; train RMSE = 0.9291
  Iteration : 40 ; train RMSE = 0.9241
  Iteration : 50 ; train RMSE = 0.9208
  Iteration : 60 ; train RMSE = 0.9185
  Iteration : 70 ; train RMSE = 0.9166
  Iteration : 80 ; train RMSE = 0.9150
  Iteration : 90 ; train RMSE = 0.9135
  Iteration : 100 ; train RMSE = 0.9120
  ```

  
