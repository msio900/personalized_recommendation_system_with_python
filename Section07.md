# 👍Section 07_ 하이브리드 추천 시스템[↩](../../)

## contents📑<a id='contents'></a>

* 0_ 들어가기 전에[✏️](#0)
* 1_ 하이브리드 추천 시스템의 장점
* 2_ 하이브리드 추천 시스템의 원리
* 3_ 하이브리드 추천 시스템(CF와 MF의 결합)

## 0_ 들어가기 전에[📑](#contents)<a id='0'></a>

![](./image/7_0-1.png)

## 1_ 하이브리드 추천 시스템의 장점[📑](#contents)<a id='1'></a>

![](./image/7_1-1.png)

## 2_ 하이브리드 추천 시스템의 원리[📑](#contents)<a id='2'></a>

```python
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd

r_cols = ["user_id","movie_id",'rating','timestamp']
ratings = pd.read_csv('./Data/u.data',
                    sep = '\t',
                    names = r_cols,
                    encoding='latin-1')

ratings_train, ratings_test = train_test_split(ratings, 
                                               test_size = 0.2,
                                               shuffle = True,
                                               random_state = 2021)

def RMSE2(y_true, y_pred):
  return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))

#2개의 더미 추천 엔진 
def recommender_1(recom_list):      #추천해야할 리스트를 받아오고 
  recommendations = [] #예측치 저장 
  for pair in recom_list: 
    recommendations.append(random.random() * 4 + 1) # 1~5사이의 난수 발생 
  return np.array(recommendations)

def recommender_2(recom_list): 
  recommendations = [] #예측치 저장 
  for pair in recom_list: 
    recommendations.append(random.random() * 4 + 1) #1~5사이의 난수 발생 
  return np.array(recommendations)

weight = [0.8, 0.2] #결합 비중 
recom_list = np.array(ratings_test)
predictions_1 = recommender_1(recom_list)
predictions_2 = recommender_2(recom_list)

predictions = predictions_1 * weight[0] + predictions_2 * weight[1] # 두 추천엔진의 예측값을 가중치 처리함.
RMSE2(recom_list[:,2], predictions)

# 실행 결과
1.5707050630820683
```

## 3_ 하이브리드 추천 시스템(CF와 MF의 결합)[📑](#contents)<a id='3'></a>

* 실제 하이브리드 추천 시스템 구현

  ```python
  from sklearn.model_selection import train_test_split
  import random
  import numpy as np
  import pandas as pd
  import os 
  
  #MF방식 
  class NEW_MF(): 
    def __init__(self, ratings, hyper_params): 
      self.R = np.array(ratings)
      #사용자 수(num_users)와 아이템 수(num_iterms)를 받아온다.
      self.num_users, self.num_items = np.shape(self.R)
      #아래는 MF weight 조절을 위한 하이퍼파라미터이다. 
      #K : 잠재요인의 수 
      self.K = hyper_params['K'] #key값 
      self.alpha = hyper_params['alpha'] #학습률
      self.beta = hyper_params['beta'] #정규화 계수 
      self.iterations = hyper_params['iterations'] #반복 횟수
      self.verbose = hyper_params['verbose'] #학습과정 출력 여부 결정 
    
      # 매핑 : index를 맞춰줌 
      # item id 
      item_id_index = []
      index_item_id = []
      for i, one_id in enumerate(ratings): #i : index, one_id : movie_id
        item_id_index.append([one_id, i])
        index_item_id.append([i, one_id])
      self.item_id_index = dict(item_id_index)
      self.index_item_id = dict(index_item_id)
      
      #user id 
      user_id_index = []
      index_user_id  = []
      for i, one_id in enumerate(ratings.T): #i : index, one_id : user_id
        user_id_index.append([one_id, i])    
        index_user_id.append([i, one_id])
      self.user_id_index = dict(user_id_index)
      self.index_user_id = dict(index_user_id)
    
    def rmse(self):
      #rating data에서 0이 아닌 요소의 인덱스
      xs, ys = self.R.nonzero() 
      #prediction과 error를 담을 리스트 변수 초기화 
      self.predictions = []
      self.errors = [] 
      #평점이 있는 요소(사용자 x, 아이템 y) 각각에 대해서 아래의 코드를 실행한다. 
      for x,y in zip(xs, ys): 
        #사용자 x, 아이템 y에 대해서 평점 예측치를 get_predition() 함수를 사용해서 계산한다.
        prediction = self.get_prediction(x,y)
        #예측값을 예측값 리스트에 추가한다.
        self.predictions.append(prediction)
        #실제값(R)과 예측값의 차이(errors) 계산해서 오차값 리스트에 추가한다.
        self.errors.append(self.R[x,y]-prediction)
      #예측값 리스트와 오차값 리스트를 numpy array형태로 변환한다.
      self.predictions = np.array(self.predictions)
      #error를 활용해서 RMSE 도출 
      self.errors = np.array(self.errors)
      return np.sqrt(np.mean(self.errors**2))
  
    def sgd(self): 
      for i,j,r in self.samples:  #i,j : 인덱스, r : 평점 
        #사용자 i, 아이템 j에 대한 평점 예측치 계산 
        prediction = self.get_prediction(i,j)
        #실제 평점과 비교한 오차 계산 
        e = (r-prediction)
        # 가중치에 대해서 계속해서 업데이트 함. 편미분
        #사용자 평가 경향 계산 및 업데이트
        self.b_u[i] += self.alpha * (e - (self.beta * self.b_u[i]))
        #아이템 평가 경향 계산 및 업데이트
        self.b_d[j] += self.alpha * (e- (self.beta * self.b_d[j]))
  
        #P 행렬 계산 및 업데이트
        self.P[i,:] += self.alpha * ((e * self.Q[j,:] - self.beta * self.P[i,:]))
        #Q 행렬 계산 및 업데이트
        self.Q[j,:] += self.alpha * ((e * self.P[i,:])- (self.beta * self.Q[j, :]))
  
    def get_prediction(self, i, j): 
      #사용자 i, 아이템 j에 대한 평점 예측치를 앞에서 배웠던 식을 이용해서 구한다.
      prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i,:].dot(self.Q[j,].T) #전체 평점 + 유저에 대한 평가 경항 + 아이템에 대한 평가 경향 + 사용자 요인값*아이템 요인값
      return prediction 
  
    #Test set 선정 
    def set_test(self, ratings_test):
      test_set = []
      for i in range(len(ratings_test)):
        x = self.user_id_index[ratings_test.iloc[i,0]] #user index
        y = self.item_id_index[ratings_test.iloc[i,1]] #item index
        z = ratings_test.iloc[i,2] #실제 평점 
        test_set.append([x,y,z])
        self.R[x,y] = 0 #평점 0으로 만들기 
      self.test_set = test_set 
      return test_set 
  
    #Test set RMSE 계산 
    def test_rmse(self):
      error = 0
      for one_set in self.test_set: 
        predicted = self.get_prediction(one_set[0],one_set[1])
        #pow : 차승 
        error += pow(one_set[2]-predicted, 2)
      return np.sqrt(error/len(self.test_set))
  
    def test(self): #학습 
      self.P = np.random.normal(scale=1./self.K,
                                size = (self.num_users, self.K))
      self.Q = np.random.normal(scale=1./self.K,
                                size = (self.num_items, self.K))
      self.b_u = np.zeros(self.num_users)
      self.b_d = np.zeros(self.num_items)
      self.b = np.mean(self.R[self.R.nonzero()])
  
      rows, columns = self.R.nonzero() 
      self.samples = [(i,j,self.R[i,j]) for i,j in zip(rows, columns)]
  
      training_process = []
      for i in range(self.iterations):
        np.random.shuffle(self.samples)
        self.sgd() #weight 값 업데이트 
        rmse1 = self.rmse() #training set
        rmse2 = self.test_rmse() #test set
        training_process.append((i+1, rmse1, rmse2))
        if self.verbose == True:
          if (i+1) % 10 == 0:
            print("Iteration : %d ; train RMSE = %.4f; test RMSE %.4f"%(i+1, rmse1, rmse2))
      return training_process 
  
    def get_one_prediction(self, user_id, item_id): #하나 예측 
      return self.get_prediction(self.user_id_index[user_id],
                                 self.item_id_index[item_id])
    
    def full_prediction(self): #전체 예측 
      return self.b + self.b_u[:, np.newaxis] + self.b_d[np.newaxis, :] + self.P.dot(self.Q.T)
  
  base_src = './Data'
  u_data_src = os.path.join(base_src,'u.data')
  r_cols = ["user_id","movie_id",'rating','timestamp']
  ratings = pd.read_csv(u_data_src,
                      sep = '\t',
                      names = r_cols,
                      encoding='latin-1')
  
  ratings_train, ratings_test = train_test_split(ratings, 
                                                 test_size = 0.2,
                                                 shuffle = True,
                                                 random_state = 2021)
  
  #full matrix
  R_temp = ratings.pivot(index = 'user_id',
                         columns = 'movie_id',
                         values = 'rating').fillna(0)
        
  hyper_params = {
      'K' : 30,
      'alpha':0.001,
      'beta':0.02,
      'iterations':100,
      'verbose':True
  }
  
  mf = NEW_MF(R_temp, hyper_params)
  test_set = mf.set_test(ratings_test) #일부분은 test로 지정 
  result = mf.test()
  
  
  
  ###################################################################
  
  from sklearn.metrics.pairwise import cosine_similarity
  
  rating_matrix = ratings_train.pivot(index ="user_id", columns = "movie_id", values = "rating")
  
  rating_mean = rating_matrix.mean(axis=1) 
  rating_bias = (rating_matrix.T - rating_mean).T #사용자 평가 경향 고려
  
  matrix_dummy = rating_matrix.copy().fillna(0)
  user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
  user_similarity = pd.DataFrame(user_similarity,
                                 index = rating_matrix.index,
                                 columns = rating_matrix.index)
  
  #CF 방식 
  def CF_knn_bias(user_id, movie_id, neighbor_size=0):
    if movie_id in rating_bias.columns: 
      sim_scores = user_similarity[user_id].copy()
      movie_ratings = rating_bias[movie_id].copy()
      none_rating_idx = movie_ratings[movie_ratings.isnull()].index
      movie_ratings = movie_ratings.drop(none_rating_idx)
      sim_scores = sim_scores.drop(none_rating_idx)
  
      if neighbor_size == 0:
        prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
        prediction = prediction + rating_mean[user_id]
  
      else:
        if len(sim_scores) > 1:
          neighbor_size = min(neighbor_size, len(sim_scores))
          sim_scores = np.array(sim_scores)
          movie_ratings = np.array(movie_ratings)
          user_idx = np.argsort(sim_scores)
          sim_scores = sim_scores[user_idx][-neighbor_size:]
          movie_ratings = movie_ratings[user_idx][-neighbor_size:]
          prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
          prediction = prediction + rating_mean[user_id]
        else: 
          prediction = rating_mean[user_id] #사용자의 평가경향 고려 
    else: 
      prediction = rating_mean[user_id]
    return prediction
  
  
  def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))
  
  #Hybrid 추천 알고리즘 
  def recommender_1(recom_list, mf):
    recommendations = np.array([
                                mf.get_one_prediction(user, movie) for (user, movie) in recom_list
    ])
    return recommendations
  
  def recommender_2(recom_list, neighbor_size=0):
    recommendations = np.array([
                                CF_knn_bias(user, movie, neighbor_size) for (user, movie) in recom_list
    ])
    return recommendations
  
  
  recom_list = np.array(ratings_test.iloc[:,[0,1]]) #전체에서 user_id, movie_id
  predictions_1 = recommender_1(recom_list, mf)
  predictions_2 = recommender_2(recom_list, 37)
  
  print('reco 1 :', RMSE2(ratings_test.iloc[:,2], predictions_1))
  print('reco 2 :', RMSE2(ratings_test.iloc[:,2], predictions_2))
  
  
  weight = [0.8, 0.2]
  predictions = predictions_1 * weight[0] + predictions_2 * weight[1]
  
  print('reco 1+2 : ', RMSE2(ratings_test.iloc[:,2], predictions))
  ```

* 최적의 예측 모델 찾기

  ```python
  result = []
  weight_rate = []
  for i in np.arange(0.1, 0.01):
      weight = [i, 1.0-i]
      predictions = predictions_1 * weight[0] + predictions_2 * weight[1]
      print("weight - %.2f RMSE = %.7f"%(weight[0], weight[1], RMSE2(ratings_test.iloc[:,2], predictions)))
      result.append(RMSE2(ratings_test.iloc[:,2], predictions))
      weight_rate.append(weight)
  print(min(result))
  index_min = result.index(min(result))
  print(weight_rate[index_min])
  ```

  
