# Section 02_ 기본적인 추천시스템

## contents📑<a id='contents'></a>

* 0_ 참고사항
* 1_ 데이터 읽기
* 2_ 인기제품 방식

## 0_ 참고사항

* 실습을 위해 `movieLens`데이터 사용
  * 영화 1점-5점 평가
  * MovieLens 100K와 20M 사용
  * 데이터 첨부[🔗](https://drive.google.com/drive/folders/19gkcIYjA3EjoNrMp9mn8KnZutKoPYLmg)

## 1_ 데이터 읽기

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

## 2_ 인기제품 방식

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

## 3_ 추천 시스템의 정확도 측정

* 추천시스템의 성능 = `정확성`

* `데이터`를 `훈련용 데이터`와 `테스트용 데이터`로 나눔.
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

  
