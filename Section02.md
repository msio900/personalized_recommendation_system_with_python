# ๐Section 02_ ๊ธฐ๋ณธ์ ์ธ ์ถ์ฒ์์คํ

## contents๐<a id='contents'></a>

* 0_ ์ฐธ๊ณ ์ฌํญ[โ๏ธ](#0)
* 1_ ๋ฐ์ดํฐ ์ฝ๊ธฐ[โ๏ธ](#1)
* 2_ ์ธ๊ธฐ์ ํ ๋ฐฉ์[โ๏ธ](#2)
* 3_ ์ถ์ฒ ์์คํ์ ์ ํ๋ ์ธก์ [โ๏ธ](#3)
* 4_ ์ฌ์ฉ์ ์ง๋จ๋ณ ์ถ์ฒ[โ๏ธ](#4)

## 0_ ์ฐธ๊ณ ์ฌํญ[๐](#contents)<a id='0'></a>

* ์ค์ต์ ์ํด `movieLens`๋ฐ์ดํฐ ์ฌ์ฉ
  * ์ํ 1์ -5์  ํ๊ฐ
  * MovieLens 100K์ 20M ์ฌ์ฉ
  * ๋ฐ์ดํฐ ์ฒจ๋ถ[๐](https://drive.google.com/drive/folders/19gkcIYjA3EjoNrMp9mn8KnZutKoPYLmg)

## 1_ ๋ฐ์ดํฐ ์ฝ๊ธฐ[๐](#contents)<a id='1'></a>

* ๊ณตํต ๋ฐ์ดํฐ ํด๋ ๊ฒฝ๋ก
  * drive/MyDrive/Recosys/Data
* MovieLens 100K ๋ฐ์ดํฐ๋ 3๊ฐ์ง ํ์ผ๋ก ๊ตฌ์ฑ
  * ์ฌ์ฉ์ ๋ฐ์ดํฐ : u.user
  * ์ํ์ ๋ํ ๋ฐ์ดํฐ : u.item
  * ์ํ ํ๊ฐ ๋ฐ์ดํฐ : u.data

* ์ฌ์ฉ์ u.user ํ์ผ์ DataFrame์ผ๋ก ์ฝ๊ธฐ

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
  # ์คํ ๊ฒฐ๊ณผ
  
  	age	sex	occupation	zip_code
  user_id				
  1	24	M	technician	85711
  2	53	F	other	94043
  3	23	M	writer	32067
  4	24	M	technician	43537
  5	33	F	other	15213
  ```

* u.item ํ์ผ์ DataFrame์ผ๋ก ์ฝ๊ธฐ

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
  
  # ์คํ ๊ฒฐ๊ณผ
  
  	title	release date	video release date	IMDB URL	unknown	Action	Adventure	Animation	Children's	Comedy	...	Fantasy	Film-Noir	Horror	Musical	Mystery	Romance	Sci-Fi	Thriller	War	Western
  movie_id																					
  1	Toy Story (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Toy%20Story%2...	0	0	0	1	1	1	...	0	0	0	0	0	0	0	0	0	0
  2	GoldenEye (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?GoldenEye%20(...	0	1	1	0	0	0	...	0	0	0	0	0	0	0	1	0	0
  3	Four Rooms (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Four%20Rooms%...	0	0	0	0	0	0	...	0	0	0	0	0	0	0	1	0	0
  4	Get Shorty (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Get%20Shorty%...	0	1	0	0	0	1	...	0	0	0	0	0	0	0	0	0	0
  5	Copycat (1995)	01-Jan-1995	NaN	http://us.imdb.com/M/title-exact?Copycat%20(1995)	0	0	0	0	0	0	...	0	0	0	0	0	0	0	1	0	0
  5 rows ร 23 columns
  ```

* u.data ํ์ผ์ DataFrame์ผ๋ก ์ฝ๊ธฐ

  ```python
  u_data_src = os.path.join(base_src, 'u.data')
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv(u_data_src,
                          sep='\t',
                          names=r_cols,
                          encoding='latin-1')
  ratings = ratings.set_index('user_id')
  ratings.head()
  
  # ์คํ ๊ฒฐ๊ณผ
  	movie_id	rating	timestamp
  user_id			
  196	242	3	881250949
  186	302	3	891717742
  22	377	1	878887116
  244	51	2	880606923
  166	346	1	886397596
  ```

## 2_ ์ธ๊ธฐ์ ํ ๋ฐฉ์[๐](#contents)<a id='2'></a>

* ๊ฐ๋ณ ์ฌ์ฉ์ ์ ๋ณด๊ฐ ์๋ **Best-Seller** ์ ํ์ ์ถ์ฒํจ. 

  * ๊ฐ์ฅ ๊ฐ๋จํ ์ถ์ฒ์ ์ ๊ณตํจ.

* ์ธ๊ธฐ์ ํ ๋ฐฉ์ ์ถ์ฒ ๋ฐฉ์ function

  ```python
  # ์ธ๊ธฐ ์ ํ ๋ฐฉ์ ์ถ์ฒ function
  def recom_movie(n_items):
      movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
      movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
      recom_movies = movies.loc[movie_sort.index]
      recommendations = recom_movies['title']
      return recommendations
  
  recom_movie(5)
  
  # ์คํ ๊ฒฐ๊ณผ
  # ์ธ๊ธฐ ์ ํ ๋ฐฉ์ ์ถ์ฒ function
  def recom_movie(n_items):
      movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
      movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
      recom_movies = movies.loc[movie_sort.index]
      recommendations = recom_movies['title']
      return recommendations
  
  recom_movie(5)
  ```

## 3_ ์ถ์ฒ ์์คํ์ ์ ํ๋ ์ธก์ [๐](#contents)<a id='3'></a>

* ์ถ์ฒ์์คํ์ ์ฑ๋ฅ = `์ ํ์ฑ`

* `๋ฐ์ดํฐ`๋ฅผ `ํ๋ จ์ฉ ๋ฐ์ดํฐ`์ `ํ์คํธ์ฉ ๋ฐ์ดํฐ`๋ก ๋๋.
  
  ![](./image/2_3-1.png)
  $$
  RMSE = \sqrt{{1 \over n} \sum_{i=1}^n({y_{i}-\hat y})^2}
  $$
  
* best seller ๋ฐฉ์์ผ๋ก ๊ตฌํ ์์ธก๊ฐ์ `RMSE`๋ก ์ ํ๋๋ฅผ ๊ตฌํจ. 

  ```python
  # 100K์ ์ํ ํ์ ์ ๋ํด์ ์ค์ ๊ฐ๊ณผ best-seller ๋ฐฉ์์ผ๋ก ๊ตฌํ ์์ธก๊ฐ์ RSME๋ฅผ ๊ณ์ฐํ๋ ์ฝ๋
  
  import numpy as np
  
  def RMSE(Y_true, Y_pred):
      return np.sqrt(np.mean((np.array(Y_true)-np.array(Y_pred))**2))
  
  # ์ ํ๋ ๊ณ์ฐ
  rmse = []
  movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
  
  for user in set(ratings.index):
      Y_true = ratings.loc[user]['rating']
      # best-seller ๋ฐฉ์์ผ๋ก
      Y_pred = movie_mean[ratings.loc[user]['movie_id']]  # movie_id์ ๋ํ ์์ธก๊ฐ.
      accuracy = RMSE(Y_true, Y_pred)
      rmse.append(accuracy)
  
  # RSME ๊ณ์ฐ
  print(np.mean(rmse))
  
  # ์คํ ๊ฒฐ๊ณผ
  0.996007224010567
  ```

  

## 4_ ์ฌ์ฉ์ ์ง๋จ๋ณ ์ถ์ฒ[๐](#contents)<a id='4'></a>

* ์ง๋จ์ ๋๋๊ธฐ ์ํ ๋ณ์ ์ค์ 

  * ์๋ฅผ ๋ค๋ฉด `๋จ์`, `์ฌ์` โ ์ง๋จ์ ๋๋ ์ best-seller ๋ฐฉ์์ ๋์

* ๋จผ์ , `train_set`, `test_set`์ผ๋ก ์ฐ์ ์ ์ผ๋ก ๋๋  ๋ด.

* ๋ฐ์ดํฐ ๋ถ๋ฌ์ค๊ธฐ

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
  
  
  # ratings DataFrame์์ timestamp ์ ๊ฑฐ
  ratings = ratings.drop('timestamp', axis=1)
  movies = movies[['movie_id','title']]
  ```

* ์์ธก ๋ฐ ์ ํ๋ ๊ณ์ฐ

  ```python
  # ๋ฐ์ดํฐ train, test set ๋ถ๋ฆฌ
  from sklearn.model_selection import train_test_split
  x = ratings.copy()
  y = ratings['user_id']
  
  x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                      test_size=0.25, # ํ๋ จ์ฉ ๋ฐ์ดํฐ์์ 75% ํ์คํธ์ฉ ๋ฐ์ดํฐ์์ 25%๋ก ์ค์ 
                                                      stratify=y)     # ์ธตํ์ถ์ถ - ๋ฐ์ดํฐ๊ฐ ๋ง๋ค ์ซ์๊ฐ ๋ค๋ฅธ๋ฐ,,,๋ญ์ณ์๋ ๊ฒฝ์ฐ๋ฅผ ๋๋นํด์ค.
  # ์ ํ๋(RMSE)๋ฅผ ๊ณ์ฐํ๋ ํจ์
  def RMSE(y_true, y_pred):
      return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))
  
  # ๋ชจ๋ธ๋ณ RMSE๋ฅผ ๊ณ์ฐํ๋ ํจ์
  def score(model):
      id_pairs = zip(x_test['user_id'], x_test['movie_id'])
      y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
      y_true = np. array(x_test['rating'])
      return RMSE(y_true, y_pred)
  
  # best_seller ํจ์๋ฅผ ์ด์ฉํ ์ ํ๋ ๊ณ์ฐ
  train_mean = x_train.groupby(['movie_id'])['rating'].mean()
  def best_seller(user_id, movie_id):
      # train_set์๋ ๋ฐ์ดํฐ๊ฐ ์๋๋ฐ...test_set์๋ ์๋ ๊ฒฝ์ฐ
      try:
          rating = train_mean[movie_id]
      except:
          rating = 3.0        # ์กด์ฌํ์ง ์์ผ๋ฉด ๊ธฐ๋ณธ๊ฐ 3.0์ ์ฃผ๊ฒ ๋ค๋ ๊ฒ!
      return rating
  
  score(best_seller)
  
  # ์คํ ๊ฒฐ๊ณผ
  1.0286518613154672
  ```

* ์ฑ๋ณ์ ๋ฐ๋ฅธ ์์ธก๊ฐ ๊ณ์ฐ

  ```python
  # ์ฑ๋ณ์ ๋ฐ๋ฅธ ์์ธก๊ฐ ๊ณ์ฐ
  merged_ratings = pd.merge(x_train, users)
  
  users = users.set_index('user_id')
  
  g_mean = merged_ratings[['movie_id', 'sex', 'rating']].groupby(['movie_id', 'sex'])['rating'].mean()
  g_mean
  # ์คํ ๊ฒฐ๊ณผ
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

* ํผ๋ฒ ํ์ด๋ธ ์ด์ฉ

  ```python
  rating_matrix = x_train.pivot(index='user_id',
                                  columns='movie_id',
                                  values='rating')
  
  rating_matrix
  
  # ์คํ ๊ฒฐ๊ณผ
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

* Gender ๊ธฐ์ค์ผ๋ก ๋๋ ์ ์ถ์ฒํด๋ด.

  ```python
  # Gender ๊ธฐ์ค ์ถ์ฒ
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
  
  # ์คํ ๊ฒฐ๊ณผ
  1.0434230218773983
  ```

  
