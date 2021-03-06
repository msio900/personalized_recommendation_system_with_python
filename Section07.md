# ๐Section 07_ ํ์ด๋ธ๋ฆฌ๋ ์ถ์ฒ ์์คํ[โฉ](../../)

## contents๐<a id='contents'></a>

* 0_ ๋ค์ด๊ฐ๊ธฐ ์ ์[โ๏ธ](#0)
* 1_ ํ์ด๋ธ๋ฆฌ๋ ์ถ์ฒ ์์คํ์ ์ฅ์ 
* 2_ ํ์ด๋ธ๋ฆฌ๋ ์ถ์ฒ ์์คํ์ ์๋ฆฌ
* 3_ ํ์ด๋ธ๋ฆฌ๋ ์ถ์ฒ ์์คํ(CF์ MF์ ๊ฒฐํฉ)

## 0_ ๋ค์ด๊ฐ๊ธฐ ์ ์[๐](#contents)<a id='0'></a>

![](./image/7_0-1.png)

## 1_ ํ์ด๋ธ๋ฆฌ๋ ์ถ์ฒ ์์คํ์ ์ฅ์ [๐](#contents)<a id='1'></a>

![](./image/7_1-1.png)

## 2_ ํ์ด๋ธ๋ฆฌ๋ ์ถ์ฒ ์์คํ์ ์๋ฆฌ[๐](#contents)<a id='2'></a>

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

#2๊ฐ์ ๋๋ฏธ ์ถ์ฒ ์์ง 
def recommender_1(recom_list):      #์ถ์ฒํด์ผํ  ๋ฆฌ์คํธ๋ฅผ ๋ฐ์์ค๊ณ  
  recommendations = [] #์์ธก์น ์ ์ฅ 
  for pair in recom_list: 
    recommendations.append(random.random() * 4 + 1) # 1~5์ฌ์ด์ ๋์ ๋ฐ์ 
  return np.array(recommendations)

def recommender_2(recom_list): 
  recommendations = [] #์์ธก์น ์ ์ฅ 
  for pair in recom_list: 
    recommendations.append(random.random() * 4 + 1) #1~5์ฌ์ด์ ๋์ ๋ฐ์ 
  return np.array(recommendations)

weight = [0.8, 0.2] #๊ฒฐํฉ ๋น์ค 
recom_list = np.array(ratings_test)
predictions_1 = recommender_1(recom_list)
predictions_2 = recommender_2(recom_list)

predictions = predictions_1 * weight[0] + predictions_2 * weight[1] # ๋ ์ถ์ฒ์์ง์ ์์ธก๊ฐ์ ๊ฐ์ค์น ์ฒ๋ฆฌํจ.
RMSE2(recom_list[:,2], predictions)

# ์คํ ๊ฒฐ๊ณผ
1.5707050630820683
```

## 3_ ํ์ด๋ธ๋ฆฌ๋ ์ถ์ฒ ์์คํ(CF์ MF์ ๊ฒฐํฉ)[๐](#contents)<a id='3'></a>

* ์ค์  ํ์ด๋ธ๋ฆฌ๋ ์ถ์ฒ ์์คํ ๊ตฌํ

  ```python
  from sklearn.model_selection import train_test_split
  import random
  import numpy as np
  import pandas as pd
  import os 
  
  #MF๋ฐฉ์ 
  class NEW_MF(): 
    def __init__(self, ratings, hyper_params): 
      self.R = np.array(ratings)
      #์ฌ์ฉ์ ์(num_users)์ ์์ดํ ์(num_iterms)๋ฅผ ๋ฐ์์จ๋ค.
      self.num_users, self.num_items = np.shape(self.R)
      #์๋๋ MF weight ์กฐ์ ์ ์ํ ํ์ดํผํ๋ผ๋ฏธํฐ์ด๋ค. 
      #K : ์ ์ฌ์์ธ์ ์ 
      self.K = hyper_params['K'] #key๊ฐ 
      self.alpha = hyper_params['alpha'] #ํ์ต๋ฅ 
      self.beta = hyper_params['beta'] #์ ๊ทํ ๊ณ์ 
      self.iterations = hyper_params['iterations'] #๋ฐ๋ณต ํ์
      self.verbose = hyper_params['verbose'] #ํ์ต๊ณผ์  ์ถ๋ ฅ ์ฌ๋ถ ๊ฒฐ์  
    
      # ๋งคํ : index๋ฅผ ๋ง์ถฐ์ค 
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
      #rating data์์ 0์ด ์๋ ์์์ ์ธ๋ฑ์ค
      xs, ys = self.R.nonzero() 
      #prediction๊ณผ error๋ฅผ ๋ด์ ๋ฆฌ์คํธ ๋ณ์ ์ด๊ธฐํ 
      self.predictions = []
      self.errors = [] 
      #ํ์ ์ด ์๋ ์์(์ฌ์ฉ์ x, ์์ดํ y) ๊ฐ๊ฐ์ ๋ํด์ ์๋์ ์ฝ๋๋ฅผ ์คํํ๋ค. 
      for x,y in zip(xs, ys): 
        #์ฌ์ฉ์ x, ์์ดํ y์ ๋ํด์ ํ์  ์์ธก์น๋ฅผ get_predition() ํจ์๋ฅผ ์ฌ์ฉํด์ ๊ณ์ฐํ๋ค.
        prediction = self.get_prediction(x,y)
        #์์ธก๊ฐ์ ์์ธก๊ฐ ๋ฆฌ์คํธ์ ์ถ๊ฐํ๋ค.
        self.predictions.append(prediction)
        #์ค์ ๊ฐ(R)๊ณผ ์์ธก๊ฐ์ ์ฐจ์ด(errors) ๊ณ์ฐํด์ ์ค์ฐจ๊ฐ ๋ฆฌ์คํธ์ ์ถ๊ฐํ๋ค.
        self.errors.append(self.R[x,y]-prediction)
      #์์ธก๊ฐ ๋ฆฌ์คํธ์ ์ค์ฐจ๊ฐ ๋ฆฌ์คํธ๋ฅผ numpy arrayํํ๋ก ๋ณํํ๋ค.
      self.predictions = np.array(self.predictions)
      #error๋ฅผ ํ์ฉํด์ RMSE ๋์ถ 
      self.errors = np.array(self.errors)
      return np.sqrt(np.mean(self.errors**2))
  
    def sgd(self): 
      for i,j,r in self.samples:  #i,j : ์ธ๋ฑ์ค, r : ํ์  
        #์ฌ์ฉ์ i, ์์ดํ j์ ๋ํ ํ์  ์์ธก์น ๊ณ์ฐ 
        prediction = self.get_prediction(i,j)
        #์ค์  ํ์ ๊ณผ ๋น๊ตํ ์ค์ฐจ ๊ณ์ฐ 
        e = (r-prediction)
        # ๊ฐ์ค์น์ ๋ํด์ ๊ณ์ํด์ ์๋ฐ์ดํธ ํจ. ํธ๋ฏธ๋ถ
        #์ฌ์ฉ์ ํ๊ฐ ๊ฒฝํฅ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
        self.b_u[i] += self.alpha * (e - (self.beta * self.b_u[i]))
        #์์ดํ ํ๊ฐ ๊ฒฝํฅ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
        self.b_d[j] += self.alpha * (e- (self.beta * self.b_d[j]))
  
        #P ํ๋ ฌ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
        self.P[i,:] += self.alpha * ((e * self.Q[j,:] - self.beta * self.P[i,:]))
        #Q ํ๋ ฌ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
        self.Q[j,:] += self.alpha * ((e * self.P[i,:])- (self.beta * self.Q[j, :]))
  
    def get_prediction(self, i, j): 
      #์ฌ์ฉ์ i, ์์ดํ j์ ๋ํ ํ์  ์์ธก์น๋ฅผ ์์์ ๋ฐฐ์ ๋ ์์ ์ด์ฉํด์ ๊ตฌํ๋ค.
      prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i,:].dot(self.Q[j,].T) #์ ์ฒด ํ์  + ์ ์ ์ ๋ํ ํ๊ฐ ๊ฒฝํญ + ์์ดํ์ ๋ํ ํ๊ฐ ๊ฒฝํฅ + ์ฌ์ฉ์ ์์ธ๊ฐ*์์ดํ ์์ธ๊ฐ
      return prediction 
  
    #Test set ์ ์  
    def set_test(self, ratings_test):
      test_set = []
      for i in range(len(ratings_test)):
        x = self.user_id_index[ratings_test.iloc[i,0]] #user index
        y = self.item_id_index[ratings_test.iloc[i,1]] #item index
        z = ratings_test.iloc[i,2] #์ค์  ํ์  
        test_set.append([x,y,z])
        self.R[x,y] = 0 #ํ์  0์ผ๋ก ๋ง๋ค๊ธฐ 
      self.test_set = test_set 
      return test_set 
  
    #Test set RMSE ๊ณ์ฐ 
    def test_rmse(self):
      error = 0
      for one_set in self.test_set: 
        predicted = self.get_prediction(one_set[0],one_set[1])
        #pow : ์ฐจ์น 
        error += pow(one_set[2]-predicted, 2)
      return np.sqrt(error/len(self.test_set))
  
    def test(self): #ํ์ต 
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
        self.sgd() #weight ๊ฐ ์๋ฐ์ดํธ 
        rmse1 = self.rmse() #training set
        rmse2 = self.test_rmse() #test set
        training_process.append((i+1, rmse1, rmse2))
        if self.verbose == True:
          if (i+1) % 10 == 0:
            print("Iteration : %d ; train RMSE = %.4f; test RMSE %.4f"%(i+1, rmse1, rmse2))
      return training_process 
  
    def get_one_prediction(self, user_id, item_id): #ํ๋ ์์ธก 
      return self.get_prediction(self.user_id_index[user_id],
                                 self.item_id_index[item_id])
    
    def full_prediction(self): #์ ์ฒด ์์ธก 
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
  test_set = mf.set_test(ratings_test) #์ผ๋ถ๋ถ์ test๋ก ์ง์  
  result = mf.test()
  
  
  
  ###################################################################
  
  from sklearn.metrics.pairwise import cosine_similarity
  
  rating_matrix = ratings_train.pivot(index ="user_id", columns = "movie_id", values = "rating")
  
  rating_mean = rating_matrix.mean(axis=1) 
  rating_bias = (rating_matrix.T - rating_mean).T #์ฌ์ฉ์ ํ๊ฐ ๊ฒฝํฅ ๊ณ ๋ ค
  
  matrix_dummy = rating_matrix.copy().fillna(0)
  user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
  user_similarity = pd.DataFrame(user_similarity,
                                 index = rating_matrix.index,
                                 columns = rating_matrix.index)
  
  #CF ๋ฐฉ์ 
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
          prediction = rating_mean[user_id] #์ฌ์ฉ์์ ํ๊ฐ๊ฒฝํฅ ๊ณ ๋ ค 
    else: 
      prediction = rating_mean[user_id]
    return prediction
  
  
  def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))
  
  #Hybrid ์ถ์ฒ ์๊ณ ๋ฆฌ์ฆ 
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
  
  
  recom_list = np.array(ratings_test.iloc[:,[0,1]]) #์ ์ฒด์์ user_id, movie_id
  predictions_1 = recommender_1(recom_list, mf)
  predictions_2 = recommender_2(recom_list, 37)
  
  print('reco 1 :', RMSE2(ratings_test.iloc[:,2], predictions_1))
  print('reco 2 :', RMSE2(ratings_test.iloc[:,2], predictions_2))
  
  
  weight = [0.8, 0.2]
  predictions = predictions_1 * weight[0] + predictions_2 * weight[1]
  
  print('reco 1+2 : ', RMSE2(ratings_test.iloc[:,2], predictions))
  ```

* ์ต์ ์ ์์ธก ๋ชจ๋ธ ์ฐพ๊ธฐ

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

  
