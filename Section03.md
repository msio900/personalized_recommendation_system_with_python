# πSection 03_ νμ νν°λ§ μΆμ² μμ€ν[β©](../../)

## contentsπ<a id='contents'></a>

* 0_ λ€μ΄κ°κΈ° μ μ[βοΈ](#0)
* 1_ νμ νν°λ§μ μλ¦¬[βοΈ](#1)
* 2_ μ μ¬λ μ§ν[βοΈ](#2)
* 3_ κΈ°λ³Έ CF μκ³ λ¦¬μ¦[βοΈ](#3)
* 4_ μ΄μμ κ³ λ €ν CF[βοΈ](#4)
* 5_ μ΅μ μ μ΄μ ν¬κΈ° κ²°μ [βοΈ](#5)
* 6_ μ¬μ©μμ νκ° κ²½ν₯μ κ³ λ €ν CF[βοΈ](#6)
* 7_ κ·Έ μΈμ CF μ νλ κ°μ  λ°©λ²[βοΈ](#7)
* 8_ μ¬μ©μ κΈ°λ° CFμ μμ΄ν κΈ°λ° CF[βοΈ](#8)
* 9_ μΆμ² μμ€νμ μ±κ³ΌμΈ‘μ μ§ν[βοΈ](#9)

## 0_ λ€μ΄κ°κΈ° μ μ[π](#contents)<a id='0'></a>

* νμ νν°λ§(Collaborative Filtering: CF) : μ΄λ€ μμ΄νμ λν΄ **λΉμ·ν μ·¨ν₯** μ κ°μ§ μ¬λλ€μ **λ€λ₯Έ μμ΄ν λν** λΉμ·ν μ·¨ν₯μ κ°μ§ κ²μ΄λ€. 

## 1_ νμ νν°λ§μ μλ¦¬[π](#contents)<a id='1'></a>

* νμ νν°λ§μ μ·¨ν₯μ΄ λΉμ·ν μ¬λλ€μ μ§λ¨ μ‘΄μ¬ κ°μ 
  * `μ μ¬μ§λ¨`μ μ·¨ν₯μ `μΆμ² λμ`μκ² μΆμ²

|                            | Movie 1 | Movie 2 | Movie 3 | Movie 4 | Movie 5 | Correlation with User 1 |
| -------------------------- | ------- | ------- | ------- | ------- | ------- | ----------------------- |
| User 1                     | 2       | 5       | 3       |         |         | -                       |
| User 2                     | 4       | 4       | 3       | 5       | 1       | 0.19                    |
| User 3                     | 1       | 5       | 4       |         | 5       | 0.89                    |
| User 4                     | 3       | 5       | 3       | 2       | 5       | 0.94                    |
| User5                      | 4       | 5       | 3       | 4       |         | 0.65                    |
| Average of User 3 & User 4 | 2       | 5       | 3.5     | 2       | 5       |                         |

> 1. κ° μ¬μ©μμ μ μ¬μ±μ κ³μ°ν¨. 
> 2. User 1κ³Ό κ°μ₯ μ μ¬ν User 3, User 4
> 3. User 1μ΄ λ³΄μ§ μμ movie 4, 5μ λν΄μ νκ°λ₯Ό λ§€κ²¨μΌ ν¨.
> 4. νμ  νκ· μ΄ λμ movie5 λ₯Ό User1μκ² μΆμ²

## 2_ μ μ¬λ μ§ν[π](#contents)<a id='2'></a>

1. μκ΄κ³μ

   ![](./image/3_2-1.png)

   * κ°μ₯ μ΄ν΄νκΈ° μ¬μ΄ μ μ¬λ
   * -1 ~ 1 μ¬μ΄ κ°

2. μ½μ¬μΈ μ μ¬λ

   ![](./image/3_2-2.png)

   ![](./image/3_2-3.png)

   * νμ νν°λ§μμ κ°μ₯ λλ¦¬ μ°μ΄λ μ μ¬λ
   * κ° μμ΄ν β νλμ μ°¨μ, μ¬μ©μμ νκ°κ° β μ’νκ°
   * λ μ¬μ©μμ νκ°κ° μ μ¬  β thetaλ μμμ§κ³ , μ½μ¬μΈ κ°μ μ»€μ§
   * -1 ~ 1 μ¬μ΄μ κ°
   * λ°μ΄ν° μ΄μ§κ°(binary)  β νλλͺ¨ν  κ³μ(tanimoto coefficient) μ¬μ© κΆμ₯

   > ![](./image/3_2-4.png)
   >
   > * νλλͺ¨ν  κ³μ(tanimoto coefficient)

3. μμΉ΄λ κ³μ

   ![](./image/3_2-5.png)

   * νλλͺ¨ν  κ³μμ λ³ν β μμΉ΄λ κ³μ
   * μ΄μ§μ λ°μ΄ν° β μ’μ κ²°κ³Ό

## 3_ κΈ°λ³Έ CF μκ³ λ¦¬μ¦[π](#contents)<a id='3'></a>

|               1λ¨κ³               |                 2λ¨κ³                 |                            3λ¨κ³                             |                    4λ¨κ³                     |
| :-------------------------------: | :-----------------------------------: | :----------------------------------------------------------: | :------------------------------------------: |
| λͺ¨λ  μ¬μ©μ κ° νκ°μ μ μ¬λ κ³μ° | μΆμ² λμκ³Ό λ€λ₯Έ μ¬μ©μκ° μ μ¬λ μΆμΆ | μΆμ² λμμ΄ νκ°νμ§ μλ μμ΄νμ λν μμ νκ°κ° κ³μ°<br />νκ°κ° = λ€λ₯Έ μ¬μ©μ νκ° * λ€λ₯Έ μ¬μ©μ μ μ¬λ | μμ΄ν μ€μμ μμ νκ°κ° κ°μ₯ λμ Nκ° μΆμ² |

* μ½μ¬μΈ μ μ¬λ ` user similarity` νμΈ

  ```python
  from sklearn.metrics.pairwise import cosine_similarity
  matrix_dummy = ratings_matrix.copy().fillna(0)
  user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
  pd.DataFrame(user_similarity)
  
  # μ€ν κ²°κ³Ό
  	0	1	2	3	4	5	6	7	8	9	...	933	934	935	936	937	938	939	940	941	942
  0	1.000000	0.155444	0.018091	0.037327	0.312866	0.284093	0.314349	0.222513	0.061672	0.259640	...	0.277484	0.100635	0.217765	0.158706	0.191710	0.122333	0.268500	0.090656	0.117199	0.300281
  1	0.155444	1.000000	0.092771	0.155143	0.073775	0.164682	0.089138	0.105211	0.039402	0.129152	...	0.120233	0.257595	0.294916	0.263524	0.254662	0.223836	0.155516	0.162992	0.146779	0.084631
  2	0.018091	0.092771	1.000000	0.286608	0.000000	0.075458	0.075167	0.063237	0.023599	0.034582	...	0.024609	0.000000	0.127619	0.033076	0.115424	0.036505	0.089818	0.026124	0.075625	0.016051
  3	0.037327	0.155143	0.286608	1.000000	0.040745	0.078150	0.091273	0.197772	0.057661	0.080271	...	0.068545	0.047433	0.126053	0.080815	0.088510	0.000000	0.156057	0.063831	0.091585	0.078438
  4	0.312866	0.073775	0.000000	0.040745	1.000000	0.207036	0.303716	0.177732	0.034706	0.143567	...	0.285905	0.038067	0.075206	0.075666	0.112294	0.028831	0.187536	0.137671	0.084622	0.264256
  ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
  938	0.122333	0.223836	0.036505	0.000000	0.028831	0.085675	0.072986	0.065599	0.051001	0.067263	...	0.057012	0.342350	0.210314	0.240177	0.302099	1.000000	0.023006	0.182925	0.018191	0.066604
  939	0.268500	0.155516	0.089818	0.156057	0.187536	0.270205	0.218335	0.163793	0.095369	0.261729	...	0.237978	0.050925	0.102415	0.068787	0.096689	0.023006	1.000000	0.098165	0.151992	0.145666
  940	0.090656	0.162992	0.026124	0.063831	0.137671	0.115379	0.053265	0.124405	0.082122	0.063780	...	0.021922	0.151324	0.190489	0.174950	0.207626	0.182925	0.098165	1.000000	0.102520	0.070007
  941	0.117199	0.146779	0.075625	0.091585	0.084622	0.245607	0.217934	0.121717	0.019845	0.125410	...	0.176588	0.096589	0.054800	0.086533	0.070681	0.018191	0.151992	0.102520	1.000000	0.131230
  942	0.300281	0.084631	0.016051	0.078438	0.264256	0.226032	0.304070	0.193257	0.026911	0.187314	...	0.186892	0.136145	0.107752	0.025145	0.170094	0.066604	0.145666	0.070007	0.131230	1.000000
  943 rows Γ 943 columns
  ```

* μ’λ λͺννκ² λ³΄λ μ½λ

  ```python
  ##### μ½μ¬μΈ μ μ¬λ κ³μ° #####
  from sklearn.metrics.pairwise import cosine_similarity
  matrix_dummy = ratings_matrix.copy().fillna(0)
  user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
  # pd.DataFrame(user_similarity)
  user_similarity = pd.DataFrame(user_similarity,
                                  index=ratings_matrix.index,
                                  columns=ratings_matrix.index)
  
  user_similarity
  # μ€ν κ²°κ³Ό
  user_id	1	2	3	4	5	6	7	8	9	10	...	934	935	936	937	938	939	940	941	942	943
  user_id																					
  1	1.000000	0.167475	0.020191	0.039798	0.297028	0.316073	0.351824	0.186010	0.072262	0.266069	...	0.270688	0.110942	0.221324	0.200087	0.182838	0.084611	0.305987	0.125474	0.142787	0.309173
  2	0.167475	1.000000	0.049921	0.136059	0.054034	0.199332	0.098998	0.097644	0.149016	0.142020	...	0.134538	0.287440	0.278580	0.414659	0.245380	0.226573	0.228027	0.140147	0.120413	0.095338
  3	0.020191	0.049921	1.000000	0.331186	0.000000	0.084230	0.040335	0.073655	0.079531	0.044186	...	0.009252	0.000000	0.080859	0.077540	0.079100	0.000000	0.085910	0.046812	0.153801	0.000000
  4	0.039798	0.136059	0.331186	1.000000	0.013009	0.027840	0.063617	0.129836	0.000000	0.000000	...	0.014678	0.000000	0.104229	0.177691	0.117125	0.000000	0.069387	0.136157	0.130458	0.036406
  5	0.297028	0.054034	0.000000	0.013009	1.000000	0.163537	0.304846	0.205443	0.044580	0.130982	...	0.282285	0.012708	0.050599	0.033572	0.082193	0.045358	0.212185	0.069671	0.177189	0.244168
  ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
  939	0.084611	0.226573	0.000000	0.000000	0.045358	0.081536	0.068369	0.097502	0.000000	0.048143	...	0.037531	0.349827	0.090386	0.209692	0.341866	1.000000	0.069122	0.153208	0.039309	0.092480
  940	0.305987	0.228027	0.085910	0.069387	0.212185	0.293066	0.264376	0.197424	0.133528	0.246606	...	0.295447	0.039216	0.125821	0.198247	0.132105	0.069122	1.000000	0.039815	0.206281	0.161205
  941	0.125474	0.140147	0.046812	0.136157	0.069671	0.126866	0.075428	0.164109	0.000000	0.070974	...	0.040198	0.171394	0.218108	0.155721	0.331758	0.153208	0.039815	1.000000	0.035286	0.085458
  942	0.142787	0.120413	0.153801	0.130458	0.177189	0.274914	0.266272	0.185634	0.090780	0.223458	...	0.184115	0.000000	0.067045	0.126248	0.079181	0.039309	0.206281	0.035286	1.000000	0.122316
  943	0.309173	0.095338	0.000000	0.036406	0.244168	0.185030	0.306245	0.227087	0.038717	0.149595	...	0.193110	0.171463	0.109478	0.057879	0.153257	0.092480	0.161205	0.085458	0.122316	1.000000
  943 rows Γ 943 columns
  ```

* `CF_simple` ν¨μ μ μ

  ```python
  #### μ£Όμ΄μ§ μνμ (movie_id) κ°μ€ νκ·  ratingμ κ³μ°νλ ν¨μ ####
  def CF_simple(user_id, movie_id):
      if movie_id in ratings_matrix.columns:
          sim_scores = user_similarity[user_id].copy()
          movie_ratings = ratings_matrix[movie_id].copy()
          none_ratings_idx = movie_ratings[movie_ratings.isnull()].index
          movie_ratings = movie_ratings.dropna()
          sim_scores = sim_scores.drop(none_ratings_idx)
          mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
  
      else:
          mean_rating = 3.0
      return mean_rating
  ```

* μ νλ κ³μ°

  ```python
  #### μ νλ κ³μ° ####
  score(CF_simple)
  
  # μ€ν κ²°κ³Ό
  1.0169396412312266
  ```

  

## 4_ μ΄μμ κ³ λ €ν CF[π](#contents)<a id='4'></a>

* λ¨μ CF μκ³ λ¦¬μ¦ κ°μ  λ°©λ² β `KNN λ°©λ²` , `Thresholding λ°©λ²`

  * KNN λ°©λ² : λμ μ μ¬λλ₯Ό κ°μ§ κ²λ€λΌλ¦¬ μμKλ₯Ό μ ν¨.
  * Thresholding λ°©λ² : λ―Έλ¦¬ λͺνΌμΌνΈλ₯Ό μ ν΄λκ³  κ·Έ κΈ°μ€μ μΆ©μ‘±μν€λ κ²μ μ§λ¨μΌλ‘ λ¬Άμ.

* CF_knn

  ```python
  def CF_knn(user_id, movie_id, neighbor_size=0):
      if movie_id in ratings_matrix.columns:
          sim_scores = user_similarity[user_id].copy()
          movie_ratings = ratings_matrix[movie_id].copy()
          none_rating_idx = movie_ratings[movie_ratings.isnull()].index
          movie_ratings = movie_ratings.dropna()
          sim_scores = sim_scores.drop(none_rating_idx)
  
          if neighbor_size == 0:
              mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
          else:
              if len(sim_scores) > 1:
                  neighbor_size = min(neighbor_size, len(sim_scores))
                  sim_scores = np.array(sim_scores)
                  movie_ratings = np.array(movie_ratings)
                  user_idx = np.argsort(sim_scores)
                  sim_scores = sim_scores[user_idx][-neighbor_size:]
                  movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                  mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
              else:
                  mean_rating = 3.0
      else:
          mean_rating = 3.0
      return mean_rating
  
  # μ νλ κ³μ°
  score(CF_knn, neighbor_size=30)
  
  # μ€ν κ²°κ³Ό
  1.0108871337570486
  ```

  * μ½κ° κ°μ λ¨μ μμ μμ.

* μνλ₯Ό μΆμ²

  ```python
  #### μ€μ  μ£Όμ΄μ§ μ¬μ©μμ λν΄ μΆμ²μ λ°λ κΈ°λ₯ κ΅¬ν ####
  ratings_matrix = x_train.pivot(index='user_id',
                                   columns='movie_id',
                                   values='rating')
  
  matrix_dummy = ratings_matrix.copy().fillna(0)
  user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
  user_similarity = pd.DataFrame(user_similarity,
                                  index=ratings_matrix.index,
                                  columns=ratings_matrix.index)
  def recom_movie(user_id, n_items, neighbor_size=30):
      user_movie = ratings_matrix.loc[user_id].copy()
      for movie in ratings_matrix.columns:
          if pd.notnull(user_movie.loc[movie]):
              user_movie.loc[movie] = 0
  
          else:
              user_movie.loc[movie] = CF_knn(user_id, movie, neighbor_size)
  
  
      movie_sort = user_movie.sort_values(ascending=False)[:n_items]
      recom_movies = movies.loc[movie_sort.index]
      recommendations = recom_movies['title']
      return recommendations
  
  recom_movie(user_id=729, n_items=5, neighbor_size=30)
  
  # μ€ν κ²°κ³Ό
  <ipython-input-3-c5813698dafe>:92: RuntimeWarning: invalid value encountered in double_scalars
    mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
  movie_id
  1293                         Star Kid (1997)
  1467    Saint of Fort Washington, The (1993)
  1189                      Prefontaine (1997)
  1491                 Tough and Deadly (1995)
  1466                Margaret's Museum (1995)
  Name: title, dtype: object
  ```

## 5_ μ΅μ μ μ΄μ ν¬κΈ° κ²°μ [π](#contents)<a id='5'></a>

![](./image/3_5-1.png)

* `overfitting`μ λ§κΈ° μν΄ μ΅μ μ μ‘°κ±΄μ μ°ΎμμΌ ν¨.

```python
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

##### λ°μ΄ν° λΆλ¬μ€κΈ° λ° λ°μ΄ν°μ λ§λ€κΈ° #####
base_src = './Data'
u_user_src = os.path.join(base_src, 'u.user')
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv(u_user_src,
                    sep = '|',
                    names = u_cols,
                    encoding='latin-1')
users = users.set_index('user_id')

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

u_data_src = os.path.join(base_src, 'u.data')
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(u_data_src,
                        sep='\t',
                        names=r_cols,
                        encoding='latin-1')

# RMSE ν¨μ
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))

# score(RMSE) κ³μ°
def score(model, neighbor_size=0):      # neighbor_size μ§μ  
    # νμ€νΈ λ°μ΄ν°μ user_idμ movie_idκ° pairλ₯Ό λ§μΆ° ννν μμ λ¦¬μ€νΈλ°μ΄ν°λ₯Ό λ§λ¦.
    id_pairs = zip(x_test['user_id'], x_test['movie_id'])
    # λͺ¨λ  μ¬μ©μ-μν μ§μμ λν΄μ μ£Όμ΄μ§ μμΈ‘ λͺ¨λΈμ μν΄ μμΈ‘κ° κ³μ° λ° λ¦¬μ€νΈν λ°μ΄ν° μμ±
    y_pred = np.array([model(user, movie, neighbor_size) for (user, movie) in id_pairs])
    # μ€μ  νμ κ°
    y_true = np.array(x_test['rating'])
    return RMSE(y_true, y_pred)

x = ratings.copy()
y = ratings['user_id']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)

########################################################################################

ratings_matrix = x_train.pivot(index='user_id', columns='movie_id', values='rating')

from sklearn.metrics.pairwise import cosine_similarity
matrix_dummy = ratings_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity,
                                index=ratings_matrix.index,
                                columns=ratings_matrix.index)

def CF_knn(user_id, movie_id, neighbor_size=0):
    # train/test setμ λΆν μ λ°λΌ rating_matrixμ μνκ° μλμ§ νμΈ
    if movie_id in ratings_matrix.columns:
        # μ£Όμ΄μ§ μ¬μ©μ(user_id)μ λ€λ₯Έ μ¬μ©μμ μ μ¬λ μΆμΆ
        sim_scores = user_similarity[user_id].copy()
        # μ£Όμ΄μ§ μν(movie_id)μ λ€λ₯Έ μ¬μ©μμ μ μ¬λ μΆμΆ
        movie_ratings = ratings_matrix[movie_id].copy()
        # μ£Όμ΄μ§ μνμ λν΄μ νκ°λ₯Ό νμ§ μμ μ¬μ©μλ₯Ό κ°μ€νκ· κ³μμμ μ μΈνκΈ° μν΄ μΈλ±μ€ μΆμΆ
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        # μ£Όμ΄μ§ μνμ λν΄μ νκ°λ₯Ό νμ§ μμ μ¬λ μ μΈ
        movie_ratings = movie_ratings.dropna()
        # μ£Όμ΄μ§ μνλ₯Ό νκ°νμ§ μμ μ¬μ©μμμ μ μ¬λλ₯Ό μ κ±°, κ°μ€ νκ·  κ³μ°ν λ νμκ° μκΈ° λλ¬Έ
        sim_scores = sim_scores.drop(none_rating_idx)


        #### Neighbot sizeκ° μ§μ λμ§ μμ κ²½μ° ####
        if neighbor_size == 0:
            mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
        #### Neighbot sizeκ° μ§μ λ κ²½μ° ####
        else:
            if len(sim_scores) > 1:
                neighbor_size = min(neighbor_size, len(sim_scores))
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                user_idx = np.argsort(sim_scores)
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            else:
                mean_rating = 3.0
    # train/test setμ λΆν μ λ°λΌ rating_matrixμ ν΄λΉ μνκ° μμΌλ©΄ κΈ°λ³Έκ° 3.0μμΈ‘μΉλ‘ κ°μ£Ό
    else:
        mean_rating = 3.0
    return mean_rating
# neighbor_sizeκ° 10, 20, 30, 40, 50, 60μΈ κ²½μ°μ λν΄μ RMSEλ₯Ό κ³μ°νκ³  μ΄λ₯Ό μΆλ ₯νλ€.
for neighbor_size in [10,20,30,40,50,60]:
    print('Neighbor Size = %d : RMSE = %.4f'%(neighbor_size, score(CF_knn, neighbor_size)))

# μ€ν κ²°κ³Ό
Neighbor Size = 10 : RMSE = 1.0314
Neighbor Size = 20 : RMSE = 1.0165
Neighbor Size = 30 : RMSE = 1.0128
Neighbor Size = 40 : RMSE = 1.0124
Neighbor Size = 50 : RMSE = 1.0128
Neighbor Size = 60 : RMSE = 1.0137
```

## 6_ μ¬μ©μμ νκ° κ²½ν₯μ κ³ λ €ν CF[π](#contents)<a id='6'></a>

![](./image/3_6-1.png)

* λ¨μμ μ¬μλ νκ° κ²½ν₯μ κ³ λ €νμλ, λ¨μμ κ²½μ° νμ μ λκ², μ¬μμ νμ μ λ?κ² νκ°ν΄μΌν¨. 
  * μλ₯Ό λ€λ©΄ 
  * κ°μΈ νμ  = 3.0 
  * μ§λ¨ νμ  = 4.0
  * μ§λ¨ νκ·  = 3.5
  * 3.5 - a (μ¬κΈ°μ aλ μ§λ¨κ³Ό μ¬μ©μ μ¬μ΄μ νμ  νκ· μ μ°¨μ΄λ₯Ό λ§ν¨.)
  * 1-3.5 = 2.5 λΌκ³  μμν¨.

|          1λ¨κ³          |                            2λ¨κ³                             |                            3λ¨κ³                             |                   4λ¨κ³                   |
| :---------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---------------------------------------: |
| κ° μ¬μ©μ νμ νκ·  κ³μ° | νμ  β κ° μ¬μ©μμ νκ· μμμ μ°¨μ΄λ‘ λ³ν<br />νμ  - ν΄λΉ μ¬μ©μμ νμ  νκ·  | νμ  νΈμ°¨μ μμΈ‘κ° κ³μ°<br />νκ°κ° = νμ νΈμ°¨ * λ€λ₯Έ μ¬μ©μ μ μ¬λ | μ€μ  μμΈ‘κ°  = νμ νΈμ°¨ μμΈ‘κ° - νμ νκ·  |

* `rating_mean` νμΈ

  ```python
  rating_mean = ratings_matrix.mean(axis=1)
  rating_mean
  
  # μ€ν κ²°κ³Ό
  user_id
  1      3.612745
  2      3.673913
  3      2.804878
  4      4.277778
  5      2.809160
           ...   
  939    4.243243
  940    3.425000
  941    4.058824
  942    4.254237
  943    3.412698
  Length: 943, dtype: float64
          
  ratings_matrix
  # μ€ν κ²°κ³Ό
  movie_id	1	2	3	4	5	6	7	8	9	10	...	1670	1672	1673	1674	1675	1677	1679	1680	1681	1682
  user_id																					
  1	NaN	3.0	NaN	3.0	3.0	NaN	4.0	1.0	NaN	3.0	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  2	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	2.0	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  3	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  4	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  5	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
  939	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	5.0	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  940	NaN	NaN	NaN	NaN	NaN	NaN	4.0	NaN	3.0	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  941	5.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  942	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  943	NaN	5.0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  943 rows Γ 1646 columns
  ```

* κ° μ¬μ©μμ νμ  νκ· μ λΉΌμ€ : `rating_bias`

  * νμ  νκ· μ νΈμ°¨

  ```python
  rating_bias = (ratings_matrix.T - rating_mean).T # Tλ transforms
  rating_bias
  
  # μ€ν κ²°κ³Ό
  movie_id	1	2	3	4	5	6	7	8	9	10	...	1669	1671	1672	1674	1675	1677	1678	1679	1681	1682
  user_id																					
  1	1.372549	-0.627451	0.372549	-0.627451	-0.627451	NaN	NaN	NaN	1.372549	-0.627451	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  2	0.255319	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	-1.744681	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  3	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  4	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  5	1.099237	0.099237	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
  939	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	0.729730	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  940	NaN	NaN	NaN	-1.525000	NaN	NaN	0.475	1.475	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  941	1.000000	NaN	NaN	NaN	NaN	NaN	0.000	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  942	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  943	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
  943 rows Γ 1643 columns
  ```

* μ¬μ©μ νκ° κ²½ν₯μ κ³ λ €ν ν¨μ μ€μ 

  ```python
  # μ¬μ©μ νκ° κ²½ν₯μ κ³ λ €ν ν¨μ
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
                  prediction = rating_mean[user_id]
      else:
          prediction = rating_mean[user_id]
      return prediction
  
  score(CF_knn_bias, 30)
  
  # μ€ν κ²°κ³Ό
  0.9476824815031243
  ```

## 7_ κ·Έ μΈμ CF μ νλ κ°μ  λ°©λ²[π](#contents)<a id='7'></a>

 ![](./image/3_7-1.png)

* μ λ’°λ κ°μ€μΉλ₯Ό λκΈ°!

  * μμΈ‘κ°μ λ―Όκ°νλ€. κ·Έλμ μ§μ  ν μ μμ.

* counts νμΈ

  ```python
  rating_binary_1 = np.array(rating_matrix > 0).astype(float)
  rating_binary_2 = rating_binary_1.T
  
  counts = np.dot(rating_binary_1, rating_binary_2)
  counts = pd.DataFrame(counts,
                          index = rating_matrix.index,
                          columns=rating_matrix.index).fillna(0)
  counts
  
  # μ€ν κ²°κ³Ό
  user_id	1	2	3	4	5	6	7	8	9	10	...	934	935	936	937	938	939	940	941	942	943
  user_id																					
  1	204.0	11.0	4.0	1.0	40.0	59.0	78.0	17.0	2.0	42.0	...	44.0	8.0	30.0	11.0	20.0	9.0	28.0	7.0	14.0	44.0
  2	11.0	47.0	6.0	2.0	1.0	14.0	9.0	4.0	1.0	7.0	...	12.0	10.0	15.0	10.0	13.0	8.0	9.0	3.0	8.0	6.0
  3	4.0	6.0	40.0	7.0	0.0	7.0	9.0	3.0	0.0	5.0	...	1.0	1.0	8.0	5.0	8.0	1.0	9.0	1.0	7.0	1.0
  4	1.0	2.0	7.0	18.0	1.0	2.0	5.0	5.0	0.0	1.0	...	2.0	0.0	4.0	1.0	4.0	0.0	5.0	1.0	4.0	1.0
  5	40.0	1.0	0.0	1.0	131.0	21.0	56.0	12.0	1.0	21.0	...	34.0	3.0	9.0	2.0	12.0	4.0	14.0	3.0	11.0	36.0
  ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
  939	9.0	8.0	1.0	0.0	4.0	6.0	9.0	4.0	0.0	5.0	...	4.0	9.0	12.0	8.0	17.0	37.0	3.0	3.0	1.0	6.0
  940	28.0	9.0	9.0	5.0	14.0	32.0	42.0	11.0	3.0	33.0	...	26.0	4.0	13.0	8.0	8.0	3.0	80.0	5.0	11.0	12.0
  941	7.0	3.0	1.0	1.0	3.0	7.0	5.0	3.0	2.0	4.0	...	2.0	5.0	11.0	4.0	7.0	3.0	5.0	16.0	3.0	3.0
  942	14.0	8.0	7.0	4.0	11.0	23.0	30.0	6.0	2.0	8.0	...	17.0	2.0	7.0	5.0	6.0	1.0	11.0	3.0	59.0	15.0
  943	44.0	6.0	1.0	1.0	36.0	29.0	64.0	15.0	0.0	20.0	...	26.0	8.0	15.0	5.0	17.0	6.0	12.0	3.0	15.0	126.0
  943 rows Γ 943 columns
  ```

* κ³΅ν΅νκ° μν μ λ°μ

  ```python
  rating_binary_1 = np.array(rating_matrix > 0).astype(float)
  rating_binary_2 = rating_binary_1.T
  
  counts = np.dot(rating_binary_1, rating_binary_2)
  counts = pd.DataFrame(counts,
                          index = rating_matrix.index,
                          columns=rating_matrix.index).fillna(0)
  
  def CF_knn_bias_sig(user_id, movie_id, neighbor_size=0):
      if movie_id in rating_bias.columns:
          sim_scores = user_similarity[user_id].copy()
          movie_ratings = rating_bias[movie_id].copy()
  
          no_rating = movie_ratings.isnull()              # nullκ°μΈ κ²λ€μ Trueλ‘ μ€μ 
          common_counts = counts[user_id]                 # κ³΅ν΅νκ° μνμ
          low_significance = common_counts < SIG_LEVEL    # κ³΅ν΅νκ° μνμκ° λ―Έλ¦¬ μ ν΄μ§ μ«μλ³΄λ€ μμ μ¬μ©μλ₯Ό TRUEλ‘ νμ
          none_rating_idx = movie_ratings[no_rating | low_significance].index
  
          movie_ratings = movie_ratings.drop(none_rating_idx)
          sim_scores = sim_scores.drop(none_rating_idx)
  
          if neighbor_size == 0:
              prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
              prediction = prediction + rating_mean[user_id]
          else:
              if len(sim_scores) > MIN_RATINGS:
                  neighbor_size = min(neighbor_size, len(sim_scores))
                  sim_scores = np.array(sim_scores)
                  movie_ratings = np.array(movie_ratings)
                  user_idx = np.argsort(sim_scores)
                  sim_scores = sim_scores[user_idx][-neighbor_size:]
                  movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                  prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                  prediction = prediction +rating_mean[user_id]
              else:
                  prediction = rating_mean[user_id]
      else:
          prediction = rating_mean[user_id]
      return prediction
  
  SIG_LEVEL = 3
  MIN_RATINGS = 3
  
  score(CF_knn_bias_sig, 30)
  
  # μ€ν κ²°κ³Ό
  0.9390548282662213
  ```

* κ°μ  μ

  ```python
      # μμΈ‘ κ°μ΄ 0.6μ΄λ€. 
      if prediction <= 1:
          prediction = 1
      elif prediction >= 5:
          prediction = 5
      # 5.2 >= 5 κ³μ°μ μ’λ λ¨μνκ².
  ```

## 8_ μ¬μ©μ κΈ°λ° CFμ μμ΄ν κΈ°λ° CF[π](#contents)<a id='8'></a>

|        | MOVIE 1 | MOVIE 2 | MOVIE 3 | MOVIE 4 |
| :----: | :-----: | :-----: | :-----: | :-----: |
| User 1 |    4    |    3    |    5    |         |
| User 2 |         |    2    |    1    |    2    |
| User 3 |    1    |    5    |         |    3    |
| User 4 |         |         |    4    |    5    |

* μ¬μ©μ κΈ°λ° CFμ μμ΄ν κΈ°λ° CFμ λΉκ΅

  | μ¬μ©μ κΈ°λ° CF                                               | μμ΄ν κΈ°λ° CF                                    |
  | ------------------------------------------------------------ | ------------------------------------------------- |
  | λ°μ΄ν°κ° νλΆν κ²½μ° μ νν μΆμ²<br />κ²°κ³Όμ λν μνμ± μ‘΄μ¬ | κ³μ°μ΄ λΉ λ¦<br />μλ°μ΄ν°μ λν κ²°κ³Ό μν₯μ΄ μ μ |

  * λ°μ΄ν° ν¬κΈ° μ κ³ , μ¬μ©μμ λν μ λ³΄κ° μλ κ²½μ° **μ¬μ©μ κΈ°λ° CF** μ μ 
  * λ°μ΄ν° ν¬κΈ° ν¬κ³ , μΆ©λΆν μ λ³΄κ° μλ κ²½μ° **μμ΄ν κΈ°λ° CF** μ μ 

* `item_similarity` μΆλ ₯

  ```python
  rating_matrix_t = np.transpose(rating_matrix)
  
  matrix_dummy = rating_matrix_t.copy().fillna(0)
  
  item_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
  item_similarity = pd.DataFrame(item_similarity,
                                  index = rating_matrix_t.index,
                                  columns = rating_matrix_t.index)
  item_similarity
  # μ€ν κ²°κ³Ό
  movie_id	1	2	3	4	5	6	7	8	9	10	...	1670	1671	1672	1673	1674	1676	1677	1679	1680	1681
  movie_id																					
  1	1.000000	0.291130	0.260815	0.387949	0.188804	0.120631	0.455862	0.342607	0.361525	0.155987	...	0.0	0.0	0.055995	0.041996	0.0	0.00000	0.000000	0.0	0.0	0.055995
  2	0.291130	1.000000	0.220749	0.393555	0.277135	0.045543	0.253821	0.192784	0.183228	0.119749	...	0.0	0.0	0.089642	0.000000	0.0	0.00000	0.000000	0.0	0.0	0.089642
  3	0.260815	0.220749	1.000000	0.240068	0.182427	0.080277	0.263677	0.136441	0.209873	0.112720	...	0.0	0.0	0.000000	0.000000	0.0	0.00000	0.036037	0.0	0.0	0.000000
  4	0.387949	0.393555	0.240068	1.000000	0.228883	0.058704	0.384913	0.333897	0.313956	0.169524	...	0.0	0.0	0.065310	0.000000	0.0	0.10885	0.043540	0.0	0.0	0.065310
  5	0.188804	0.277135	0.182427	0.228883	1.000000	0.047729	0.233474	0.256426	0.190654	0.020761	...	0.0	0.0	0.000000	0.000000	0.0	0.00000	0.000000	0.0	0.0	0.000000
  ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
  1676	0.000000	0.000000	0.000000	0.108850	0.000000	0.000000	0.000000	0.077267	0.064889	0.097745	...	0.0	0.0	0.000000	0.000000	0.0	1.00000	0.000000	0.0	0.0	0.000000
  1677	0.000000	0.000000	0.036037	0.043540	0.000000	0.000000	0.058887	0.096583	0.081111	0.000000	...	0.0	0.0	0.000000	0.000000	0.0	0.00000	1.000000	0.0	0.0	0.000000
  1679	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	...	0.0	0.0	0.000000	0.000000	0.0	0.00000	0.000000	1.0	1.0	0.000000
  1680	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	...	0.0	0.0	0.000000	0.000000	0.0	0.00000	0.000000	1.0	1.0	0.000000
  1681	0.055995	0.089642	0.000000	0.065310	0.000000	0.000000	0.058887	0.000000	0.064889	0.000000	...	0.0	0.0	1.000000	0.000000	0.0	0.00000	0.000000	0.0	0.0	1.000000
  1638 rows Γ 1638 columns
  ```

* μμ΄ν κΈ°λ° CF

  ```python
  import os
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  
  ##### λ°μ΄ν° λΆλ¬μ€κΈ° λ° λ°μ΄ν°μ λ§λ€κΈ° #####
  base_src = './Data'
  u_user_src = os.path.join(base_src, 'u.user')
  u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
  users = pd.read_csv(u_user_src,
                      sep = '|',
                      names = u_cols,
                      encoding='latin-1')
  users = users.set_index('user_id')
  
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
  
  u_data_src = os.path.join(base_src, 'u.data')
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv(u_data_src,
                          sep='\t',
                          names=r_cols,
                          encoding='latin-1')
  
  def RMSE(y_true, y_pred):
      return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))
  
  def score(model):      # neighbor_size μ§μ  
      # νμ€νΈ λ°μ΄ν°μ user_idμ movie_idκ° pairλ₯Ό λ§μΆ° ννν μμ λ¦¬μ€νΈλ°μ΄ν°λ₯Ό λ§λ¦.
      id_pairs = zip(x_test['user_id'], x_test['movie_id'])
      # λͺ¨λ  μ¬μ©μ-μν μ§μμ λν΄μ μ£Όμ΄μ§ μμΈ‘ λͺ¨λΈμ μν΄ μμΈ‘κ° κ³μ° λ° λ¦¬μ€νΈν λ°μ΄ν° μμ±
      y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
      # μ€μ  νμ κ°
      y_true = np.array(x_test['rating'])
      return RMSE(y_true, y_pred)
  
  x = ratings.copy()
  y = ratings['user_id']
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y)
  rating_matrix = x_train.pivot(index='user_id', columns='movie_id', values='rating')
  
  ########################################################################################
  rating_matrix_t = np.transpose(rating_matrix)
  
  matrix_dummy = rating_matrix_t.copy().fillna(0)
  
  item_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
  item_similarity = pd.DataFrame(item_similarity,
                                  index = rating_matrix_t.index,
                                  columns = rating_matrix_t.index)
  
  def CF_IBCF(user_id, movie_id):
      if movie_id in item_similarity.columns:
          sim_scores = item_similarity[movie_id]
          user_rating = rating_matrix_t[user_id]
          none_rating_idx = user_rating[user_rating.isnull()].index
          user_rating = user_rating.dropna()
          sim_scores = sim_scores.drop(none_rating_idx)
          mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
      else:
          mean_rating = 3.0
      
      return mean_rating
  score(CF_IBCF)
  
  # μ€ν κ²°κ³Ό
  1.0196572278928668
  ```

  

  ## 9_ μΆμ² μμ€νμ μ±κ³ΌμΈ‘μ μ§ν[π](#contents)<a id='9'></a>
  
  |                  1λ¨κ³                  |                      2λ¨κ³                       |                      3λ¨κ³                       |
  | :-------------------------------------: | :----------------------------------------------: | :----------------------------------------------: |
  | λ°μ΄ν°λ₯Ό train setκ³Ό  test setμΌλ‘ λΆλ¦¬ | train setμ μ¬μ©ν΄μ νμ΅νκ³ , test setμΌλ‘ νκ° | μμ° νμ κ³Ό μ€μ  νμ  μ°¨μ΄λ₯Ό κ³μ° ν μ νλ μΈ‘μ  |

1. κ° μμ΄νμ μμ νμ κ³Ό μ€μ  νμ  μ°¨μ΄
   * RMSE β μ°μμ μΈ
2. μΆμ²ν μμ΄νκ³Ό μ¬μ©μ μ€μ  μ νκ³Ό λΉκ΅

![](./image/3_9-1.png)

* `TPR`κ³Ό `FPR`

  ![](./image/3_9-2.png)

  

  * νμ€μμλ TNμ΄ λλ¬΄ μ»€μ§λ κ²½ν₯μ΄ μμ κ·Έλμ TNμ΄ λ€μ΄κ° μ§νλ₯Ό μ¬μ©νκΈ° μ΄λ €μ.
  * μμ΄νμ΄ λ§μ μμ€νμ μ±κ³ΌμΈ‘μ μ§νλ μ λ°λλ, μ¬νμ¨μ μΆμ²ν¨.
