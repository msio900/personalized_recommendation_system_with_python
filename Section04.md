# ๐Section 04_ Matrix Factorization(MF) ๊ธฐ๋ฐ ์ถ์ฒ[โฉ](../../)

## contents๐<a id='contents'></a>

* 0_ ๋ค์ด๊ฐ๊ธฐ ์ ์[โ๏ธ](#0)
* 1_ Matrix Factorization(MF)๋ฐฉ์์ ์๋ฆฌ[โ๏ธ](#1)
* 2_ SGD(Stichastic Gradient Decent)๋ฅผ ์ฌ์ฉํ MF ์๊ณ ๋ฆฌ์ฆ[โ๏ธ](#2)
* 3_ SGD๋ฅผ ์ฌ์ฉํ MF ๊ธฐ๋ณธ ์๊ณ ๋ฆฌ์ฆ[โ๏ธ](#3)
* 4_ train/test ๋ถ๋ฆฌ MF ์๊ณ ๋ฆฌ์ฆ[โ๏ธ](#4)
* 5_ MF์ ์ต์  ํ๋ผ๋ฏธํฐ ์ฐพ๊ธฐ[โ๏ธ](#5)
* 6_ MF์ SVD[โ๏ธ](#6)

## 0_ ๋ค์ด๊ฐ๊ธฐ ์ ์[๐](#contents)<a id='0'></a>

|      | ๋ฉ๋ชจ๋ฆฌ ๊ธฐ๋ฐ ์๊ณ ๋ฆฌ์ฆ                          | ๋ชจ๋ธ ๊ธฐ๋ฐ ์๊ณ ๋ฆฌ์ฆ                                   |
| ---- | --------------------------------------------- | ---------------------------------------------------- |
| ์ค๋ช | ๋ฉ๋ชจ๋ฆฌ์ ์๋ ๋ฐ์ดํฐ๋ฅผ ๊ณ์ฐํด์ ์ถ์ฒํ๋ ๋ฐฉ์ | ๋ฐ์ดํฐ๋ก๋ถํฐ ๋ฏธ๋ฆฌ ๋ชจ๋ธ์ ๊ตฌ์ฑํ ํ์์ ์ถ์ฒํ๋ ๋ฐฉ์ |
| ํน์ง | ๊ฐ๋ณ ์ฌ์ฉ์ ๋ฐ์ดํฐ ์ง์ค                       | ์ ์ฒด ์ฌ์ฉ์ ํจํด ์ง์ค                                |
| ์ฅ์  | ์๋ ๋ฐ์ดํฐ์ ์ถฉ์คํ๊ฒ ์ฌ์ฉ                   | ๋๊ท๋ชจ ๋ฐ์ดํฐ์ ๋น ๋ฅด๊ฒ ๋ฐ์                          |
| ๋จ์  | ๋๊ท๋ชจ ๋ฐ์ดํฐ์ ๋๋ฆฌ๊ฒ ๋ฐ์                   | ๋ชจ๋ธ ์์ฑ ๊ณผ์  ์ค๋ ๊ฑธ๋ฆผ                             |

## 1_ Matrix Factorization(MF)๋ฐฉ์์ ์๋ฆฌ[๐](#contents)<a id='1'></a>

![](./image/4_1-1.png)

* R = N *M

* P = M * K

* Q<sup>T</sup> = N *K

* ์ฌ์ฉ์์ ์์ดํ์ ํน์ฑ์ k๊ฐ์ ์ ์ฌ ์์ธ์ ์ฌ์ฉํด์ ๋ถ์ํ๋ ๋ชจ๋ธ

* ์์

  |        | ์ก์ - ๋๋ผ๋ง<br />(-1~1) | ํํ์ง-์ฌ์ค์ฃผ์<br />(-1~1) |
  | :----: | :-----------------------: | :-------------------------: |
  | User 1 |           -0.43           |            0.21             |
  | User 2 |           0.31            |            0.92             |
  | User 3 |           0.69            |            -0.03            |
  | User 4 |           0.49            |            -0.3             |

* ์์ดํ ์์ธ์ ๋ํด์ ํด์ ๊ฐ๋ฅ

  |         | ์ก์ - ๋๋ผ๋ง<br />(-1~1) | ํํ์ง-์ฌ์ค์ฃผ์<br />(-1~1) |
  | ------- | ------------------------- | --------------------------- |
  | Movie 1 | 0.31                      | 0.6                         |
  | Movie 2 | 0.61                      | -0.82                       |
  | Movie 3 | -0.38                     | -0.61                       |
  | Movie 4 | -0.79                     | 0.08                        |

* 2์ฐจ์ ๊ณต๊ฐ์ ๋ฐฐ์นํ๋ฉด

  ![](./image/4_1-2.png)

* ์ฌ์ฉ์์ ์ํ๋ณ ์์ธก ํ์ 

  * ![](./image/4_1-3.png)

  |        | Movie 1 | Movie 2 | Movie 3 | Movie 4 |
  | ------ | ------- | ------- | ------- | ------- |
  | User 1 | -0.0073 | -0.4345 | 0.0353  | 0.3565  |
  | User 2 | 0.6481  | -0.5653 | -0.679  | -0.1713 |
  | User 3 | 0.1959  | 0.4455  | -0.2439 | -0.5475 |
  | User 4 | -0.0374 | 0.5266  | 0.0082  | -0.3874 |

## 2_ SQD(Stichastic Gradient Decent)๋ฅผ ์ฌ์ฉํ MF ์๊ณ ๋ฆฌ์ฆ[๐](#contents)<a id='2'></a>

![](./image/4_2-1.png)

![](./image/4_2-2.png)

![](./image/4_2-3.png)

* ์์๊ฐ

* ์์ธก ์ค์ฐจ
* ์์ธก์ค์ฐจ์ ์ ๊ณฑ์ ํ๊ณ  p์ q์ ๋ํด์ ํธ๋ฏธ๋ถ์ ํจ.
* ๋ค์ ํ์ฉ์ ํด์ p, q๋ฅผ ์๋ฐ์ดํธํจ. 
* ์ํ๋ ์ผ๋งํผ์ ๊ฐ์ค์น๋ฅผ ๋ ๊ฒ์ธ์ง?

![](./image/4_2-4.png)

![](./image/4_2-5.png)

* ์ค๋ฒํผํ์ ์ด๋ป๊ฒ ์ค์ด๋๊ฐ?
  * ์ ๊ทํ ํ์ ์ด๋ป๊ฒ ์ค์ด๋๊ฐ?
  * weight๋ง ์ฃผ๊ฒ ๋๋ฉด ๋๋ฌด ์ ๋ง๊ฒ ๋์ด ์ค์ฐจ๋ฅผ ์ถ๊ฐํจ.
  * ๋ฒ ํ : ์ ๊ทํ ์ ๋ ๊ณ ๋ คํ์ฌ ์ต์ ์ ๋ฒ ํ๋ฅผ ๊ตฌํจ.
* ๊ฐ ์ฌ์ฉ์์ ๊ฐ ์์ดํ์ ๊ฒฝํฅ์ฑ ๊ณ ๋ คํ๊ธฐ
  * ๊ฒฝํฅ์ฑ ์ ๊ฑฐํ๊ธฐ

* ์ต์ข์ ์ผ๋ก M, P, Q๋ฅผ ๊ตฌํจ.

## 3_ SGD๋ฅผ ์ฌ์ฉํ MF ๊ธฐ๋ณธ ์๊ณ ๋ฆฌ์ฆ[๐](#contents)<a id='3'></a>

* ์ธํ

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
  
  # timestamp ์ ๊ฑฐ
  ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)
  ```

* ๊ตฌํ

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
          self.P = np.random.normal(scale=1./self.K,                  # scale = ํ์คํธ์ฐจ๋ฅผ ์ด์ผ๊ธฐํจ
                                      size=(self.num_users, self.K))  # size ์ค์  ์ ์ ์์ ์ ์ฌ์์ธ์ ๊ฐฏ์ = ํฌ๊ธฐ ๊ฐ
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
              training_process.append((i+1, rmse)) # ๋ช๋ฒ ์งธ์ RMSE์ธ์ง?
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
  
  # ์คํ ๊ฒฐ๊ณผ
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

  

## 4_ train/test ๋ถ๋ฆฌ MF ์๊ณ ๋ฆฌ์ฆ[๐](#contents)<a id='4'></a>

* ๊ตฌํ

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
  
  # timestamp ์ ๊ฑฐ
  ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)
  
  # train / test set ๋ถ๋ฆฌ
  from sklearn.utils import shuffle
  TRAIN_SIZE = 0.75
  
  # (์ฌ์ฉ์ - ์ํ - ํ์ )
  ratings = shuffle(ratings, random_state=2021)   # ๋ชจ๋  ์ฌ๋์ด ๋๊ฐ์ raitings๊ฐ ๋์ค๊ฒ ๋จ. 
  cutoff = int(TRAIN_SIZE * len(ratings))
  ratings_train = ratings.iloc[:cutoff]           # ์ด๋๊น์ง 
  ratings_test = ratings.iloc[cutoff:]            # ์ด๋ ์ดํ๋ก ๋ฐ์ดํฐ๋ฅผ ๋ฝ์ผ๋ฉด ๋ ๊ฒ ์ธ์ง?
  
  class NEW_MF():
      def __init__(self, ratings, hyper_params):
          self.R = np.array(ratings)
          # ์ฌ์ฉ์ ์ (num_users)์ ์์ดํ ์ (num_items)๋ฅผ ๋ฐ์์จ๋ค.
          self.num_users, self.num_items = np.shape(self.R)
          # ์๋๋ MF weight ์กฐ์ ์ ์ํ ํ์ดํผ ํ๋ผ๋ฏธ๋๋ค.
          # K : ์ ์ฌ์์ธ(latent factor)์ ์
          self.K = hyper_params['K']
          # alpha : ํ์ต๋ฅ 
          self.alpha = hyper_params['alpha']
          # beta : ์ ๊ทํ ๊ฐ์
          self.beta = hyper_params['beta']
          # iterations : SGD์ ๊ณ์ฐ์ ํ  ๋ ๋ฐ๋ณต ํ์
          self.iterations = hyper_params['iterations']
          # verbose : SGD์ ํ์ต ๊ณผ์ ์ ์ค๊ฐ์ค๊ฐ์ ์ถ๋ ฅํ  ๊ฒ์ธ์ง์ ๋ํ ์ฌ๋ถ
          self.verbose = hyper_params['verbose']
  
          # ์ง๋ ์๊ฐ๊ณผ ์กฐ๊ธ ๋ค๋ฅธ ๋ถ๋ถ
          # movie_lens ๋ฐ์ดํฐ๋ ๊ต์ฅํ ์ ์ ๋ฆฌ๋ ๋ฐ์ดํฐ ์ด์ง๋ง, ํ์์ ๋ฐ์ดํฐ๋ ์ฐ์๊ฐ์ด ์๋ ์ ์์. 
          # self.R์ numpy๋ก ๋ณํ ์ํฌ ๊ฒฝ์ฐ ์ค๊ฐ์ ๋น์ด์๋ ์ค์  id ๋ self.R์ ๊ฐ๊ณผ ๋งค์นญ์ด ์๋จ.
          ### Item id์ ๊ดํ ###
          item_id_index = []
          index_item_id = []
          for i, one_id in enumerate(ratings):
              item_id_index.append([one_id, i])
              index_item_id.append([i, one_id])
          self.item_id_index = dict(item_id_index)    # ์ด๋ค id ๊ฐ์ด ๋ค์ด์ค๋๋ผ๋ id ๊ฐ๊ณผ numpy array์ ๋งคํ ์์ผ์ค.
          self.index_item_id = dict(index_item_id)
  
          user_id_index = []
          index_user_id = []
          for i, one_id in enumerate(ratings.T):
              user_id_index.append([one_id, i])
              index_user_id.append([i, one_id])
          self.user_id_index = dict(user_id_index)
          self.index_user_id = dict(index_user_id)
  
  
      def rmse(self):
          # self.R์์ ํ์ ์ด ์๋ (0์ด ์๋) ์์์ ์ธ๋ฑ์ค๋ฅผ ๊ฐ์ ธ์จ๋ค.
          xs, ys = self.R.nonzero()
          # prediction๊ณผ error๋ฅผ ๋ด์ ๋ฆฌ์คํธ ๋ณ์ ์ด๊ธฐํ
          self.predictions = []
          self.errors = []
          # ํ์ ์ด ์๋ ์์(์ฌ์ฉ์ x, ์์ดํ y) ๊ฐ๊ฐ์ ๋ํด์ ์๋์ ์ฝ๋๋ฅผ ์คํํ๋ค.
          for x, y in zip(xs, ys):
              # ์ฌ์ฉ์ x, ์์ดํ y์ ๋ํด์ ํ์  ์์ธก์น๋ฅผ get_prediction()ํจ์๋ฅผ ์ฌ์ฉํด์ ๊ณ์ฐํ๋ค.
              prediction = self.get_prediction(x, y)
              # ์์ธก๊ฐ์ ์์ธก๊ฐ ๋ฆฌ์คํธ์ ์ถ๊ฐํ๋ค.
              self.predictions.append(prediction)
              # ์ค์ ๊ฐ(R)๊ณผ ์์ธก๊ฐ์ ์ฐจ์ด(errors) ๊ณ์ฐํด์ ์ค์ฐจ๊ฐ ๋ฆฌ์คํธ์ ์ถ๊ฐํ๋ค.
              self.errors.append(self.R[x, y] - prediction)
          # ์์ธก๊ฐ ๋ฆฌ์คํธ์ ์ค์ฐจ๊ฐ ๋ฆฌ์คํธ๋ฅผ numpy arrayํํ๋ก ๋ณํํ๋ค.
          self.predictions = np.array(self.predictions)
          self.errors = np.array(self.errors)
          # error๋ฅผ ํ์ฉํด์ RMSE๋ฅผ ๋์ถ
          return np.sqrt(np.mean(self.errors**2))
  
      def sgd(self):
          for i, j, r in self.samples:
              # ์ฌ์ฉ์ i : ์์ดํ j์ ๋ํ ํ์  ์์ธก์น ๊ณ์ฐ
              prediction = self.get_prediction(i, j)
              # ์ค์  ํ์ ๊ณผ ๋น๊ตํ ์ค์ฐจ ๊ณ์ฐ
              e = (r - prediction)
              # ์ฌ์ฉ์ ํ๊ฐ ๊ฒฝํฅ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
              self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
              # ์์ดํ ํ๊ฐ ๊ฒฝํฅ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
              self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])
              # P ํ๋ ฌ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
              self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
              # Q ํ๋ ฌ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
              self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])
      def get_prediction(self, i, j):
          # ์ฌ์ฉ์ i, ์์ดํ j์ ๋ํ ํ์  ์์ธก์น๋ฅผ ์์์ ๋ฐฐ์ ๋ ์์ ์ด์ฉํด์ ๊ตฌํ๋ค.
          prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
          return prediction
  
      # Test set ์ ์ 
      def set_test(self, ratings_test):
          test_set = []
          for i in range(len(ratings_test)):      # test ๋ฐ์ดํฐ์ ์๋ ๊ฐ ๋ฐ์ดํฐ์ ๋ํด์
              x = self.user_id_index[ratings_test.iloc[i, 0]]
              y = self.item_id_index[ratings_test.iloc[i, 1]]
              z = ratings_test.iloc[i, 2]
              test_set.append([x, y, z])
              self.R[x, y] = 0                    # test set์ผ๋ก ์ง์ ํ ๊ฒ๋ค์ ๋ชจ๋ 0์ผ๋ก
          self.test_set = test_set
          return test_set                   
  
  
      # Test set RMSE ๊ณ์ฐ
      def test_rmse(self):
          error = 0   # 0์ผ๋ก ์ด๊ธฐํ
          for one_set in self.test_set:
              predicted = self.get_prediction(one_set[0], one_set[1])
              error += pow(one_set[2] - predicted, 2)         # pow : e => e^2 ์ฐจ์น
          return np.sqrt(error/len(self.test_set))
  
      def test(self):
          # Initializing user-feature and item-feature matrix
          self.P = np.random.normal(scale=1./self.K,
                                      size=(self.num_users, self.K))
          self.Q = np.random.normal(scale=1./self.K, 
                                      size=(self.num_items, self.K))
  
          # ์ ์  ๊ฒฝํฅ
          self.b_u = np.zeros(self.num_users)
          self.b_d = np.zeros(self.num_items)
          self.b = np.mean(self.R[self.R.nonzero()]) # ์จ์ ํ ๊ฑธ๋ฆฐ๊ฒ๋ค๋ง ๊ณ์ฐํ๊ฒ ํจ. 
  
          # List of training samples
          rows, columns = self.R.nonzero()            # non zero์ธ ๊ฒ๋ง ์ธ๋ฑ์ค๋ฅผ ๊ฐ์ ธ์ด.
          self.samples = [(i, j, self.R[i,j]) for i, j in zip(rows, columns)]
  
          # Stochastic gradient descent for given number of iterations
          training_process = []
          for i in range(self.iterations):
              np.random.shuffle(self.samples)
              self.sgd()
              rmse1 = self.rmse()
              rmse2 = self.test_rmse()
              training_process.append((i+1, rmse1, rmse2))
              if self.verbose:
                  if (i+1) % 10 == 0:
                      print("Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %.4f" % (i+1, rmse1, rmse2))
          return training_process
  
      def get_one_prediction(self, user_id, item_id):
          return self.get_prediction(self.user_id_index[user_id],
                                      self.item_id_index[item_id])    # ์์ธก์น๋ฅผ ๊ณ์ฐํด์ค.
      
      # Full user-movie rating matrix
      def full_prediction(self):
          return self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis,:] + self.P.dot(self.Q.T) # ์ ์ฒด๋ฅผ ๊ณ์ฐํด์ค.
  
  R_temp = ratings.pivot(index='user_id',
                      columns = 'movie_id',
                      values = 'rating').fillna(0)
  
  hyper_params = {
      'K' : 30,
      'alpha' : 0.001,
      'beta' : 0.02,
      'iterations' : 100,
      'verbose' : True
  }
  
  mf = NEW_MF(R_temp, hyper_params)
  
  test_set = mf.set_test(ratings_test)
  results = mf.test()
  
  # ์คํ ๊ฒฐ๊ณผ
  Iteration: 10 ; Train RMSE = 0.9666 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9412 ; Test RMSE = 0.9623
  Iteration: 30 ; Train RMSE = 0.9297 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9228 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9179 ; Test RMSE = 0.9493
  Iteration: 60 ; Train RMSE = 0.9139 ; Test RMSE = 0.9478
  Iteration: 70 ; Train RMSE = 0.9101 ; Test RMSE = 0.9466
  Iteration: 80 ; Train RMSE = 0.9059 ; Test RMSE = 0.9455
  Iteration: 90 ; Train RMSE = 0.9008 ; Test RMSE = 0.9442
  Iteration: 100 ; Train RMSE = 0.8941 ; Test RMSE = 0.9425
  ```

* ์ ์ฒด ์์ธก ๊ฒฐ๊ณผ

  ```python
  print(mf.full_prediction())
  
  # ์คํ ๊ฒฐ๊ณผ
  [[3.91454543 3.39333233 2.98759373 ... 3.38110623 3.46510384 3.41074121]
   [3.80528174 3.24899288 2.90001427 ... 3.26739582 3.36065561 3.33977482]
   [3.40858422 2.91780662 2.49550962 ... 2.88229575 2.9774935  2.94997816]
   ...
   [4.14704511 3.60480179 3.24708529 ... 3.58197039 3.70818832 3.69413186]
   [4.32293351 3.78026638 3.40714565 ... 3.75817252 3.89115358 3.86689988]
   [3.83928592 3.36202847 2.94700558 ... 3.29387556 3.42047262 3.39708125]]
  ```

* ํ๋์ ์์ธก๊ฐ ๊ฐ์ ธ์ค๊ธฐ

  ```python
  print(mf.get_one_prediction(1, 2))
  
  # ์คํ ๊ฒฐ๊ณผ
  3.3933323287310477
  ```

## 5_ MF์ ์ต์  ํ๋ผ๋ฏธํฐ ์ฐพ๊ธฐ[๐](#contents)<a id='5'></a>

* ๊ณผ์ ํฉ`overfitting`์ ๋ง๊ธฐ ์ํด

| ๋๋ต์ ์ธ ์ต์ ์ K ์์น ์ฐพ๊ธฐ | โ    | ๋๋ต์  K ์ฃผ๋ณ ํ์์ผ๋ก, ์ต์  K ์ฐพ๊ธฐ | โ    | ์ฃผ์ด์ง K ํตํด ์ต์ ์ `iterations` ์ ํ |
| --------------------------- | ---- | ----------------------------------- | ---- | -------------------------------------- |
| 50 ~ 260, k = 10            |      | 50~70, k = 60                       |      | fix, k=63 iteration =123               |

* ์ต์ ์ ํ๋ฆฌ๋ฏธํฐ ์ฐพ๋ ํ์ต

  ```python
  # ์ต์ ์ K ์ฐพ๊ธฐ
  results = []
  index = []
  
  R_temp = ratings.pivot(index='user_id',
                          columns='movie_id',
                          values='rating').fillna(0)
  for K in range(50, 261, 10):
      print(f'K : {K}')
      hyper_params = {
          'K' : K,
          'alpha' : 0.001,
          'beta' : 0.02,
          'iterations' : 100,
          'verbose' : True
      }
      mf = NEW_MF(R_temp,
                  hyper_params)
      test_set = mf.set_test(ratings_test)
      result = mf.test()
      index.append(K)
      results.append(result)
      
   # ์คํ ๊ฒฐ๊ณผ
  K : 50
  Iteration: 10 ; Train RMSE = 0.9669 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9417 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9305 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9239 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9195 ; Test RMSE = 0.9493
  Iteration: 60 ; Train RMSE = 0.9160 ; Test RMSE = 0.9479
  Iteration: 70 ; Train RMSE = 0.9129 ; Test RMSE = 0.9467
  Iteration: 80 ; Train RMSE = 0.9097 ; Test RMSE = 0.9458
  Iteration: 90 ; Train RMSE = 0.9060 ; Test RMSE = 0.9447
  Iteration: 100 ; Train RMSE = 0.9012 ; Test RMSE = 0.9432
  K : 60
  Iteration: 10 ; Train RMSE = 0.9669 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9418 ; Test RMSE = 0.9623
  Iteration: 30 ; Train RMSE = 0.9307 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9242 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9198 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9165 ; Test RMSE = 0.9479
  Iteration: 70 ; Train RMSE = 0.9137 ; Test RMSE = 0.9469
  Iteration: 80 ; Train RMSE = 0.9108 ; Test RMSE = 0.9461
  Iteration: 90 ; Train RMSE = 0.9076 ; Test RMSE = 0.9451
  Iteration: 100 ; Train RMSE = 0.9035 ; Test RMSE = 0.9440
  K : 70
  Iteration: 10 ; Train RMSE = 0.9670 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9419 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9308 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9244 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9201 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9169 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9143 ; Test RMSE = 0.9470
  Iteration: 80 ; Train RMSE = 0.9117 ; Test RMSE = 0.9463
  Iteration: 90 ; Train RMSE = 0.9089 ; Test RMSE = 0.9455
  Iteration: 100 ; Train RMSE = 0.9055 ; Test RMSE = 0.9446
  K : 80
  Iteration: 10 ; Train RMSE = 0.9670 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9420 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9309 ; Test RMSE = 0.9551
  Iteration: 40 ; Train RMSE = 0.9245 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9203 ; Test RMSE = 0.9493
  Iteration: 60 ; Train RMSE = 0.9171 ; Test RMSE = 0.9479
  Iteration: 70 ; Train RMSE = 0.9145 ; Test RMSE = 0.9470
  Iteration: 80 ; Train RMSE = 0.9120 ; Test RMSE = 0.9461
  Iteration: 90 ; Train RMSE = 0.9093 ; Test RMSE = 0.9453
  Iteration: 100 ; Train RMSE = 0.9059 ; Test RMSE = 0.9443
  K : 90
  Iteration: 10 ; Train RMSE = 0.9670 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9420 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9310 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9247 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9205 ; Test RMSE = 0.9493
  Iteration: 60 ; Train RMSE = 0.9174 ; Test RMSE = 0.9479
  Iteration: 70 ; Train RMSE = 0.9149 ; Test RMSE = 0.9470
  Iteration: 80 ; Train RMSE = 0.9126 ; Test RMSE = 0.9462
  Iteration: 90 ; Train RMSE = 0.9101 ; Test RMSE = 0.9454
  Iteration: 100 ; Train RMSE = 0.9071 ; Test RMSE = 0.9445
  K : 100
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9421 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9311 ; Test RMSE = 0.9551
  Iteration: 40 ; Train RMSE = 0.9247 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9206 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9175 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9151 ; Test RMSE = 0.9470
  Iteration: 80 ; Train RMSE = 0.9129 ; Test RMSE = 0.9462
  Iteration: 90 ; Train RMSE = 0.9105 ; Test RMSE = 0.9455
  Iteration: 100 ; Train RMSE = 0.9077 ; Test RMSE = 0.9447
  K : 110
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9421 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9311 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9248 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9207 ; Test RMSE = 0.9493
  Iteration: 60 ; Train RMSE = 0.9177 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9153 ; Test RMSE = 0.9470
  Iteration: 80 ; Train RMSE = 0.9131 ; Test RMSE = 0.9463
  Iteration: 90 ; Train RMSE = 0.9108 ; Test RMSE = 0.9455
  Iteration: 100 ; Train RMSE = 0.9081 ; Test RMSE = 0.9447
  K : 120
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9421 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9312 ; Test RMSE = 0.9551
  Iteration: 40 ; Train RMSE = 0.9249 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9207 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9178 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9154 ; Test RMSE = 0.9470
  Iteration: 80 ; Train RMSE = 0.9132 ; Test RMSE = 0.9463
  Iteration: 90 ; Train RMSE = 0.9110 ; Test RMSE = 0.9456
  Iteration: 100 ; Train RMSE = 0.9083 ; Test RMSE = 0.9447
  K : 130
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9422 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9312 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9249 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9208 ; Test RMSE = 0.9493
  Iteration: 60 ; Train RMSE = 0.9179 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9155 ; Test RMSE = 0.9470
  Iteration: 80 ; Train RMSE = 0.9134 ; Test RMSE = 0.9463
  Iteration: 90 ; Train RMSE = 0.9113 ; Test RMSE = 0.9456
  Iteration: 100 ; Train RMSE = 0.9088 ; Test RMSE = 0.9448
  K : 140
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9422 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9312 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9250 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9209 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9180 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9157 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9136 ; Test RMSE = 0.9463
  Iteration: 90 ; Train RMSE = 0.9116 ; Test RMSE = 0.9457
  Iteration: 100 ; Train RMSE = 0.9092 ; Test RMSE = 0.9449
  K : 150
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9422 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9312 ; Test RMSE = 0.9551
  Iteration: 40 ; Train RMSE = 0.9250 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9209 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9180 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9158 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9138 ; Test RMSE = 0.9464
  Iteration: 90 ; Train RMSE = 0.9118 ; Test RMSE = 0.9458
  Iteration: 100 ; Train RMSE = 0.9096 ; Test RMSE = 0.9451
  K : 160
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9422 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9313 ; Test RMSE = 0.9551
  Iteration: 40 ; Train RMSE = 0.9250 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9210 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9181 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9159 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9139 ; Test RMSE = 0.9464
  Iteration: 90 ; Train RMSE = 0.9120 ; Test RMSE = 0.9458
  Iteration: 100 ; Train RMSE = 0.9099 ; Test RMSE = 0.9451
  K : 170
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9422 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9313 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9251 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9210 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9182 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9160 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9141 ; Test RMSE = 0.9464
  Iteration: 90 ; Train RMSE = 0.9122 ; Test RMSE = 0.9458
  Iteration: 100 ; Train RMSE = 0.9101 ; Test RMSE = 0.9451
  K : 180
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9422 ; Test RMSE = 0.9623
  Iteration: 30 ; Train RMSE = 0.9313 ; Test RMSE = 0.9551
  Iteration: 40 ; Train RMSE = 0.9251 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9211 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9182 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9160 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9141 ; Test RMSE = 0.9464
  Iteration: 90 ; Train RMSE = 0.9123 ; Test RMSE = 0.9458
  Iteration: 100 ; Train RMSE = 0.9103 ; Test RMSE = 0.9452
  K : 190
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9423 ; Test RMSE = 0.9623
  Iteration: 30 ; Train RMSE = 0.9313 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9251 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9211 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9183 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9161 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9143 ; Test RMSE = 0.9465
  Iteration: 90 ; Train RMSE = 0.9125 ; Test RMSE = 0.9459
  Iteration: 100 ; Train RMSE = 0.9107 ; Test RMSE = 0.9453
  K : 200
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9423 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9313 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9251 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9211 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9183 ; Test RMSE = 0.9481
  Iteration: 70 ; Train RMSE = 0.9162 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9144 ; Test RMSE = 0.9465
  Iteration: 90 ; Train RMSE = 0.9127 ; Test RMSE = 0.9459
  Iteration: 100 ; Train RMSE = 0.9109 ; Test RMSE = 0.9454
  K : 210
  Iteration: 10 ; Train RMSE = 0.9671 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9423 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9314 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9252 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9212 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9184 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9162 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9144 ; Test RMSE = 0.9464
  Iteration: 90 ; Train RMSE = 0.9127 ; Test RMSE = 0.9459
  Iteration: 100 ; Train RMSE = 0.9109 ; Test RMSE = 0.9454
  K : 220
  Iteration: 10 ; Train RMSE = 0.9672 ; Test RMSE = 0.9806
  Iteration: 20 ; Train RMSE = 0.9423 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9314 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9252 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9212 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9184 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9163 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9145 ; Test RMSE = 0.9465
  Iteration: 90 ; Train RMSE = 0.9128 ; Test RMSE = 0.9459
  Iteration: 100 ; Train RMSE = 0.9111 ; Test RMSE = 0.9454
  K : 230
  Iteration: 10 ; Train RMSE = 0.9672 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9423 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9314 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9252 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9212 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9184 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9163 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9145 ; Test RMSE = 0.9465
  Iteration: 90 ; Train RMSE = 0.9129 ; Test RMSE = 0.9459
  Iteration: 100 ; Train RMSE = 0.9111 ; Test RMSE = 0.9454
  K : 240
  Iteration: 10 ; Train RMSE = 0.9672 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9423 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9314 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9252 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9212 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9184 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9163 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9146 ; Test RMSE = 0.9465
  Iteration: 90 ; Train RMSE = 0.9130 ; Test RMSE = 0.9460
  Iteration: 100 ; Train RMSE = 0.9113 ; Test RMSE = 0.9455
  K : 250
  Iteration: 10 ; Train RMSE = 0.9672 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9423 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9314 ; Test RMSE = 0.9551
  Iteration: 40 ; Train RMSE = 0.9252 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9213 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9185 ; Test RMSE = 0.9481
  Iteration: 70 ; Train RMSE = 0.9164 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9147 ; Test RMSE = 0.9465
  Iteration: 90 ; Train RMSE = 0.9131 ; Test RMSE = 0.9460
  Iteration: 100 ; Train RMSE = 0.9114 ; Test RMSE = 0.9454
  K : 260
  Iteration: 10 ; Train RMSE = 0.9672 ; Test RMSE = 0.9807
  Iteration: 20 ; Train RMSE = 0.9423 ; Test RMSE = 0.9622
  Iteration: 30 ; Train RMSE = 0.9314 ; Test RMSE = 0.9552
  Iteration: 40 ; Train RMSE = 0.9252 ; Test RMSE = 0.9515
  Iteration: 50 ; Train RMSE = 0.9213 ; Test RMSE = 0.9494
  Iteration: 60 ; Train RMSE = 0.9185 ; Test RMSE = 0.9480
  Iteration: 70 ; Train RMSE = 0.9164 ; Test RMSE = 0.9471
  Iteration: 80 ; Train RMSE = 0.9147 ; Test RMSE = 0.9465
  Iteration: 90 ; Train RMSE = 0.9131 ; Test RMSE = 0.9460
  Iteration: 100 ; Train RMSE = 0.9115 ; Test RMSE = 0.9455
  ```

* `summary`ํ์ธ

  ```python
  summary = []
  for i in range(len(results)):
      RMSE = []
      for result in results[i]:
          RMSE.append(result[2])
      min = np.min(RMSE)
      j = RMSE.index(min)
      summary.append([index[i], j+1, RMSE[j]])
  ```

  

![](./image/4_5-1.png)

| ์ต์  ํ๋ผ๋ฏธํฐ 1 ๊ตฌํ๊ธฐ | โ    | ์ต์  ํ๋ผ๋ฏธํฐ 2 ๊ตฌํ๊ธฐ<br />(ํ๋ผ๋ฏธํฐ 1 ๊ณ ์ ) | โ    | ์ต์  ํ๋ผ๋ฏธํฐ 3 ๊ตฌํ๊ธฐ<br />(ํ๋ผ๋ฏธํฐ1, 2 ๊ณ ์ ) | โ    | ์ต์  ํ๋ฆฌ๋ฏธํฐ 4 ๊ตฌํ๊ธฐ<br />(ํ๋ฆฌ๋ฏธํฐ 1, 2, 3 ๊ณ ์ ) |
| ---------------------- | ---- | --------------------------------------------- | ---- | ----------------------------------------------- | ---- | --------------------------------------------------- |

## 6_ MF์ SVD[๐](#contents)<a id='6'></a>

![](./image/4_6-1.png)

* `MF`์ `SVD`(Singular Value Decomposition, ํน์ด๊ฐ ๋ถํด)
  * `SVD`
    * 3๊ฐ์ ํ๋ ฌ๋ก ๋๋ ์ค. 
    * ์๋ ํ๋ ฌ์ ๋ถํดํด์ 3๊ฐ์ ํ๋ ฌ๋ก ๋๋ ์ค๋ค์ ๋ค์ ํ๊ฐ์ ํ๋ ฌ๋ก ๋ง๋ฆ.
    * null์ ํ์ฉํ์ง ์์.
    * 0์ผ๋ก ๋์ฒดํ๊ฒ๋๋ฉด...๊ทธ๋ฅ 0์ผ๋ก ๊ฐ๊น์ด ๊ฐ์ผ๋ก ๊ฒฐ๊ณผ๊ฐ ๋์ด.
    * ์ฐจ์ ์ถ์ -> 10000๊ฐ โ 500๊ฐ
  * `MF`
    * 2๊ฐ์ ํ๋ ฌ๋ก ๋๋ ์ค.
    * null์ 0์ผ๋ก ๋์ฒด ๊ฐ๋ฅ
* ์ถ์ฒ ์์คํ ๋ถ์ผ์์๋ `SVD`๊ฐ ๊ฑฐ์ ์ฌ์ฉ๋์ง ์์
* `SVD++`  : `MF`๋ฅผ ๊ฐ๋ํด์ ๋ง๋  ๊ฐ๋
