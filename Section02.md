# 👍Section 02_ 기본적인 추천시스템

## contents📑<a id='contents'></a>

* 0_ 참고사항[✏️](#0)
* 1_ 데이터 읽기[✏️](#1)
* 2_ 인기제품 방식[✏️](#2)
* 3_ 추천 시스템의 정확도 측정[✏️](#3)
* 4_ 사용자 집단별 추천[✏️](#4)

## 0_ 참고사항[📑](#contents)<a id='0'></a>

* 실습을 위해 `movieLens`데이터 사용
  * 영화 1점-5점 평가
  * MovieLens 100K와 20M 사용
  * 데이터 첨부[🔗](https://drive.google.com/drive/folders/19gkcIYjA3EjoNrMp9mn8KnZutKoPYLmg)

## 1_ 데이터 읽기[📑](#contents)<a id='1'></a>

* 공통 데이터 폴더 경로
  * drive/MyDrive/Recosys/Data
* MovieLens 100K 데이터는 3가지 파일로 구성
  * 사용자 데이터 : u.user
  * 영화에 대한 데이터 : u.item
  * 영화 평가 데이터 : u.data

* 사용자 u.user 파일을 DataFrame으로 읽기

  ```python
  import os
  import pandas as pd
  
  base_src = './Data'
  u_user_src = os.path.join(base_src, 'u.user')
  u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
  users = pd.read_csv(u_user_src,
                      sep = '|',
                      names = u_cols,
                      encoding='latin-1')
  users = users.set_index('user_id')
  users.head()
  # 실행 결과
  
  	age	sex	occupation	zip_code
  user_id				
  1	24	M	technician	85711
  2	53	F	other	94043
  3	23	M	writer	32067
  4	24	M	technician	43537
  5	33	F	other	15213
  ```

* u.item 파일을 DataFrame으로 읽기

  ```python
  u_item_src = os.path.join(base_src, 'u.item')
  i_cols = ['movie_id', 'title','release date', 'video release date',
              'IMDB URL', 'unknown', 'Action','Adventure','Animation',
              'Children\'s', 'Comedy', 'Crime','Documentary','Drama','Fantasy',
              'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
  movies = pd.read_csv(u_item_src,
                      sep='|',
                      names=i_cols,
                      encoding='latin-1')
  movies = movies.set_index('movie_id')
  movies.head()
  
  # 실행 결과
  
  	title	release date	video release date	IMDB URL	unknown	Action	Adventure	Animation	Children's	Comedy	...	Fantasy	Film-Noir	Horror	Musical	Mystery	Romance	Sci-Fi	Thriller	War	Western
  movie_id																					
  1	Toy Story (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Toy%20Story%2...	0	0	0	1	1	1	...	0	0	0	0	0	0	0	0	0	0
  2	GoldenEye (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?GoldenEye%20(...	0	1	1	0	0	0	...	0	0	0	0	0	0	0	1	0	0
  3	Four Rooms (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Four%20Rooms%...	0	0	0	0	0	0	...	0	0	0	0	0	0	0	1	0	0
  4	Get Shorty (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Get%20Shorty%...	0	1	0	0	0	1	...	0	0	0	0	0	0	0	0	0	0
  5	Copycat (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Copycat%20(1995)	0	0	0	0	0	0	...	0	0	0	0	0	0	0	1	0	0
  5 rows × 23 columns
  ```

* u.data 파일을 DataFrame으로 읽기

  ```python
  u_data_src = os.path.join(base_src, 'u.data')
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv(u_data_src,
                          sep='\t',
                          names=r_cols,
                          encoding='latin-1')
  ratings = ratings.set_index('user_id')
  ratings.head()
  
  # 실행 결과
  	movie_id	rating	timestamp
  user_id			
  196	242	3	881250949
  186	302	3	891717742
  22	377	1	878887116
  244	51	2	880606923
  166	346	1	886397596
  ```

## 2_ 인기제품 방식[📑](#contents)<a id='2'></a>

* 개별 사용자 정보가 아닌 **Best-Seller** 제품을 추천함. 

  * 가장 간단한 추천을 제공함.

* 인기제품 방식 추천 방식 function

  ```python
  # 인기 제품 방식 추천 function
  def recom_movie(n_items):
      movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
      movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
      recom_movies = movies.loc[movie_sort.index]
      recommendations = recom_movies['title']
      return recommendations
  
  recom_movie(5)
  
  # 실행 결과
  # 인기 제품 방식 추천 function
  def recom_movie(n_items):
      movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
      movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
      recom_movies = movies.loc[movie_sort.index]
      recommendations = recom_movies['title']
      return recommendations
  
  recom_movie(5)
  ```

## 3_ 추천 시스템의 정확도 측정[📑](#contents)<a id='3'></a>

* 추천시스템의 성능 = `정확성`

* `데이터`를 `훈련용 데이터`와 `테스트용 데이터`로 나눔.
  
  ![](./image/2_3-1.png)
  $$
  RMSE = \sqrt{{1 \over n} \sum_{i=1}^n({y_{i}-\hat y})^2}
  $$
  
* best seller 방식으로 구한 예측값을 `RMSE`로 정확도를 구함. 

  ```python
  # 100K의 영화 평점에 대해서 실제값과 best-seller 방식으로 구한 예측값의 RSME를 계산하는 코드
  
  import numpy as np
  
  def RMSE(Y_true, Y_pred):
      return np.sqrt(np.mean((np.array(Y_true)-np.array(Y_pred))**2))
  
  # 정확도 계산
  rmse = []
  movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
  
  for user in set(ratings.index):
      Y_true = ratings.loc[user]['rating']
      # best-seller 방식으로
      Y_pred = movie_mean[ratings.loc[user]['movie_id']]  # movie_id에 대한 예측값.
      accuracy = RMSE(Y_true, Y_pred)
      rmse.append(accuracy)
  
  # RSME 계산
  print(np.mean(rmse))
  
  # 실행 결과
  0.996007224010567
  ```

  

## 4_ 사용자 집단별 추천[📑](#contents)<a id='4'></a>

* 집단을 나누기 위한 변수 설정

  * 예를 들면 `남자`, `여자` → 집단을 나눠서 best-seller 방식을 도입

* 먼저, `train_set`, `test_set`으로 우선적으로 나눠 봄.

* 데이터 불러오기

  ```python
  base_src = './Data'
  u_user_src = os.path.join(base_src, 'u.user')
  u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
  users = pd.read_csv(u_user_src,
                      sep = '|',
                      names = u_cols,
                      encoding='latin-1')
  
  u_item_src = os.path.join(base_src, 'u.item')
  i_cols = ['movie_id', 'title','release date', 'video release date',
              'IMDB URL', 'unknown', 'Action','Adventure','Animation',
              'Children\'s', 'Comedy', 'Crime','Documentary','Drama','Fantasy',
              'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
  movies = pd.read_csv(u_item_src,
                      sep='|',
                      names=i_cols,
                      encoding='latin-1')
  
  u_data_src = os.path.join(base_src, 'u.data')
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv(u_data_src,
                          sep='\t',
                          names=r_cols,
                          encoding='latin-1')
  
  
  # ratings DataFrame에서 timestamp 제거
  ratings = ratings.drop('timestamp', axis=1)
  movies = movies[['movie_id','title']]
  ```

* 예측 및 정확도 계산

  ```python
  # 데이터 train, test set 분리
  from sklearn.model_selection import train_test_split
  x = ratings.copy()
  y = ratings['user_id']
  
  x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                      test_size=0.25, # 훈련용 데이터셋을 75% 테스트용 데이터셋을 25%로 설정
                                                      stratify=y)     # 층화추출 - 데이터가 마다 숫자가 다른데,,,뭉쳐있는 경우를 대비해줌.
  # 정확도(RMSE)를 계산하는 함수
  def RMSE(y_true, y_pred):
      return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))
  
  # 모델별 RMSE를 계산하는 함수
  def score(model):
      id_pairs = zip(x_test['user_id'], x_test['movie_id'])
      y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
      y_true = np. array(x_test['rating'])
      return RMSE(y_true, y_pred)
  
  # best_seller 함수를 이용한 정확도 계산
  train_mean = x_train.groupby(['movie_id'])['rating'].mean()
  def best_seller(user_id, movie_id):
      # train_set에는 데이터가 있는데...test_set에는 없는 경우
      try:
          rating = train_mean[movie_id]
      except:
          rating = 3.0        # 존재하지 않으면 기본값 3.0을 주겠다는 것!
      return rating
  
  score(best_seller)
  
  # 실행 결과
  1.0286518613154672
  ```

* 성별에 따른 예측값 계산

  ```python
  # 성별에 따른 예측값 계산
  merged_ratings = pd.merge(x_train, users)
  
  users = users.set_index('user_id')
  
  g_mean = merged_ratings[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()
  g_mean
  # 실행 결과
  movie_id  sex
  1         F      3.752809
            M      3.931624
  2         F      3.470588
            M      3.175000
  3         F      2.500000
                     ...   
  1674      M      4.000000
  1678      M      1.000000
  1680      M      2.000000
  1681      M      3.000000
  1682      M      3.000000
  Name: rating, Length: 3028, dtype: float64
  ```

* 피벗 테이블 이용

  ```python
  rating_matrix = x_train.pivot(index='user_id',
                                  columns='movie_id',
                                  values='rating')
  
  rating_matrix
  
  # 실행 결과
  movie_id	1	2	3	4	5	6	7	8	9	10	...	1668	1669	1670	1672	1673	1674	1678	1680	1681	1682
  user_id																					
  1	5.0	3.0	4.0	3.0	3.0	5.0	4.0	1.0	NaN	3.0	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  2	4.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	2.0	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  3	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  4	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  5	4.0	3.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
  939	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	5.0	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  940	NaN	NaN	NaN	NaN	NaN	NaN	4.0	NaN	3.0	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  941	5.0	NaN	NaN	NaN	NaN	NaN	4.0	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  942	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  943	NaN	5.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  ```

* Gender 기준으로 나눠서 추천해봄.

  ```python
  # Gender 기준 추천
  def cf_gender(user_id, movie_id):
      if movie_id in rating_matrix.columns:
          gender = users.loc[user_id]['sex']
          if gender in g_mean[movie_id].index:
              gender_rating = g_mean[movie_id][gender]
          else:
              gender_rating = 3.0
      else:
          gender_rating = 3.0
      return gender_rating
  score(cf_gender)
  
  # 실행 결과
  1.0434230218773983
  ```

  
