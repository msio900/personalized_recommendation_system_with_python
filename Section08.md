# ๐Section 08_ ๋๊ท๋ชจ ๋ฐ์ดํฐ์ ์ฒ๋ฆฌ๋ฅผ ์ํ Sparse Matrix ์ฌ์ฉ[โฉ](../../)

## contents๐<a id='contents'></a>

* 0_ ๋ค์ด๊ฐ๊ธฐ ์ ์[โ๏ธ](#0)
* 1_ Sparse Matrix์ ๊ฐ๋๊ณผ Python์์์ ์ฌ์ฉ[โ๏ธ](#1)
* 2_Sparse Matrix๋ฅผ ์ถ์ฒ ์๊ณ ๋ฆฌ์ฆ์ ์ ์ฉํ๊ธฐ[โ๏ธ](#2)

## 0_ ๋ค์ด๊ฐ๊ธฐ ์ ์[๐](#contents)<a id='0'></a>

![](./image/8_0-1.png)

* `movie_lens` ๋ฐ์ดํฐ์์ `full matrix`๋ก ์ด์ฉํ๊ธฐ๋ ์ด๋ ค์

![](./image/8_0-2.png)

* ๋๋ถ๋ถ 0์ธ `sparse matrix`์.

![](./image/8_0-3.png)

* 25๊ฐ ์์ 4๊ฐ์ ์์๋ก 12๊ฐ๋ฅผ ํํํจ
* ๊ทธ๋ฌ๋ ํฌ์ํ๋ ฌ์ ์ฌ์ฉํ๊ฒ ๋๋ฉด ๋ฐ์ดํฐ ์ฒ๋ฆฌ์ ๋ํ Overhead Cost(๊ฐ์ ๋น)๊ฐ ์ปค์ง : ๋ฐฐ๋ณด๋ค ๋ฐฐ๊ผฝ์ด ๋ ์ปค์ง ์ ์์. ๋ฐ๋ผ์ ๋ฐ์ดํฐ๊ฐ ์ง์ง ํด ๋๋ง ์ด์ฉํ๋ฉด ์ข์.
* `python`์ ๊ฒฝ์ฐ์๋ `scipy`๋ฅผ ์ด์ฉํ๋ฉด ๋จ. 

## 1_ Sparse Matrix์ ๊ฐ๋๊ณผ Python์์์ ์ฌ์ฉ[๐](#contents)<a id='1'></a>

![](./image/8_1-1.png)

* COO(Coordinate :์ขํ) ํ์
  * ๊ฐ : [3, 1, 2]
  * ํ : [0, 2, 1]
  * ์ด : [0, 0, 1]
* CSR(Compressed Sparse Row)ํ์ : COOํ์์ ๋นํจ์จ์ฑ์ ๊ฐ์ ํจ.
  * ๊ฐ : [1 5 1 4 3 2 5 6 3 2 7 8 1]
  * ํ : [0 0 1 1 1 1 1 2 2 3 4 4 5] โ [0 2 3 7 9 10 12 13]
  * ์ด : [2 5 0 1 3 4 5 1 3 0 3 5 0]
  * ์ด์ง ๋นํจ์จ์ ์์ ๋ณผ ์ ์์. -> ํ์ ์ค๋ณต๊ฐ์ด ๋ํ๋๋ ๋ชจ์ต์ ๋ณผ ์ ์์.
  * ๋ญ๋น๋ฅผ ์ค์ฌ์ผ!

* full matrix ๊ตฌํ

  ```python
  import numpy as np
  import pandas as pd
  #sparse matrix๋ฅผ ์ฌ์ฉํ๊ธฐ ์ํ scipy ๋ผ์ด๋ธ๋ฌ๋ฆฌ 
  from scipy.sparse import csr_matrix 
  
  #๊ฐ๋จํ ํ์คํธ๋ฅผ ์ํ ์์ ๋ฐ์ดํฐ
  ratings = {'user_id' : [1, 2, 4],
             'movie_id':[2, 3, 7],
             'rating' : [4, 3, 1]}
  ratings = pd.DataFrame(ratings)
  
  #Pandas pivot์ ์ด์ฉํด์ full matrix ๋ณํ
  #์ผ๋ฐ์ ์ธ DataFrame์ pivot ๊ธฐ๋ฅ์ ์ฌ์ฉํด์ full matrix ๋ณํ
  rating_matrix = ratings.pivot(index = 'user_id',
                                columns = 'movie_id',
                                values = 'rating').fillna(0)
        
  full_matrix1 = np.array(rating_matrix)
  print(full_matrix1)
  
  # ์คํ ๊ฒฐ๊ณผ
  [[4. 0. 0.]
   [0. 3. 0.]
   [0. 0. 1.]]
  
  rating_matrix
  # ์คํ ๊ฒฐ๊ณผ
  movie_id	2	3	7
  user_id			
  1	4.0	0.0	0.0
  2	0.0	3.0	0.0
  4	0.0	0.0	1.0
  ```

* sparse matrix ๊ตฌํ

  ```python
  #Sparse matrix๋ฅผ ์ด์ฉํด์ full matrix ๋ณํ
  #์์์ ๊ฐ(ํ์ ) ์ง์  
  data = np.array(ratings['rating'])
  #row ์ธ๋ฑ์ค ์ง์  
  row_indices = np.array(ratings['user_id'])
  #column ์ธ๋ฑ์ค ์ง์ 
  col_indices = np.array(ratings['movie_id'])
  #์๋ ๋ฐ์ดํฐ๋ฅผ ์๊น ์ค๋ชํ๋ csr_matrix๋ก ๋ณํ 
  rating_matrix = csr_matrix((data,(row_indices, col_indices)),dtype=int)
  print(rating_matrix)
  
  # ์คํ ๊ฒฐ๊ณผ
  (1, 2)	4
  (2, 3)	3
  (4, 7)	1
  
  rating_matrix[1, 2]
  # ์คํ ๊ฒฐ๊ณผ
  4
  
  full_matrix2 = rating_matrix.toarray()
  print(full_matrix2)
  # ์คํ ๊ฒฐ๊ณผ
  [[0 0 0 0 0 0 0 0]
   [0 0 4 0 0 0 0 0]
   [0 0 0 3 0 0 0 0]
   [0 0 0 0 0 0 0 0]
   [0 0 0 0 0 0 0 1]]
  ```

## 2_ Sparse Matrix๋ฅผ ์ถ์ฒ ์๊ณ ๋ฆฌ์ฆ์ ์ ์ฉํ๊ธฐ[๐](#contents)<a id='2'></a>

* `python`์์ ๋์ฉ๋ ๋ฐ์ดํฐ๋ฅผ ๋ถ๋ฌ์ฌ ๊ฒฝ์ฐ

  ```python
  from sklearn.model_selection import train_test_split
  import random
  import numpy as np
  import pandas as pd
  import os 
  
  base_src = './Data'
  ratings_20m_src = os.path.join(base_src, 'ratings-20m.csv')
  r_cols = ["user_id","movie_id",'rating','timestamp']
  #20M data ์ฝ์ด์ค๊ธฐ 
  ratings = pd.read_csv(ratings_20m_src,
                      sep = ',',
                      names = r_cols,
                      encoding='latin-1')
  
  R_temp = ratings.pivot(index='user_id', columns = 'movie_id', values = 'rating').fillna(0)
  
  # ์คํ ๊ฒฐ๊ณผ
  ---------------------------------------------------------------------------
  ValueError                                Traceback (most recent call last)
  <ipython-input-1-d276daa88f26> in <module>
       16                     encoding='latin-1')
       17 
  ---> 18 R_temp = ratings.pivot(index='user_id', columns = 'movie_id', values = 'rating').fillna(0)
  
  C:\Python\Python36\lib\site-packages\pandas\core\frame.py in pivot(self, index, columns, values)
     6676         from pandas.core.reshape.pivot import pivot
     6677 
  -> 6678         return pivot(self, index=index, columns=columns, values=values)
     6679 
     6680     _shared_docs[
  
  C:\Python\Python36\lib\site-packages\pandas\core\reshape\pivot.py in pivot(data, index, columns, values)
      475         else:
      476             indexed = data._constructor_sliced(data[values]._values, index=index)
  --> 477     return indexed.unstack(columns)
      478 
      479 
  
  C:\Python\Python36\lib\site-packages\pandas\core\series.py in unstack(self, level, fill_value)
     3901         from pandas.core.reshape.reshape import unstack
     3902 
  -> 3903         return unstack(self, level, fill_value)
     3904 
     3905     # ----------------------------------------------------------------------
  
  C:\Python\Python36\lib\site-packages\pandas\core\reshape\reshape.py in unstack(obj, level, fill_value)
      423             return _unstack_extension_series(obj, level, fill_value)
      424         unstacker = _Unstacker(
  --> 425             obj.index, level=level, constructor=obj._constructor_expanddim,
      426         )
      427         return unstacker.get_result(
  
  C:\Python\Python36\lib\site-packages\pandas\core\reshape\reshape.py in __init__(self, index, level, constructor)
      116 
      117         if num_rows > 0 and num_columns > 0 and num_cells <= 0:
  --> 118             raise ValueError("Unstacked DataFrame is too big, causing int32 overflow")
      119 
      120         self._make_selectors()
  
  ValueError: Unstacked DataFrame is too big, causing int32 overflow
  ```

  * ๋ฐ์ดํฐํ๋ ์์ด ๋๋ฌด ํฌ๋ค๊ณ  ์ค๋ฅ๊ฐ ๋์ด.

* `sparse matrix` ์ฌ์ฉ

  ```python
  #Sparse Matrix ์ฌ์ฉ์ ์ํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ 
  from scipy.sparse import csr_matrix 
  from sklearn.model_selection import train_test_split
  import random
  import numpy as np
  import pandas as pd
  import os 
  
  base_src = './Data'
  ratings_20m_src = os.path.join(base_src, 'ratings-20m.csv')
  r_cols = ["user_id","movie_id",'rating','timestamp']
  #20M data ์ฝ์ด์ค๊ธฐ 
  ratings = pd.read_csv(ratings_20m_src,
                      sep = ',',
                      names = r_cols,
                      encoding='latin-1')
  ratings = ratings[['user_id','movie_id','rating']].astype(int)
  
  #๋ฐ์ดํฐ ์ง์  
  data = np.array(ratings['rating'])
  #row ์ธ๋ฑ์ค ์ ์ฅ 
  row_indices = np.array(ratings['user_id'])
  #column ์ธ๋ฑ์ค ์ง์ 
  col_indices = np.array(ratings['movie_id'])
  #csr_matrix ํ์์ผ๋ก ๋ฐ์ดํฐ๋ฅผ ๋ณํํด์ ratings์ ์ ์ฅํ๋ค. 
  R_temp = csr_matrix((data,(row_indices, col_indices)),dtype=int)
  
  class NEW_MF(): 
    def __init__(self, ratings, hyper_params): 
      self.R = ratings
      #์ฌ์ฉ์ ์(num_users)์ ์์ดํ ์(num_iterms)๋ฅผ ๋ฐ์์จ๋ค.
      self.num_users, self.num_items = np.shape(self.R)
      #์๋๋ MF weight ์กฐ์ ์ ์ํ ํ์ดํผํ๋ผ๋ฏธํฐ์ด๋ค. 
      #K : ์ ์ฌ์์ธ์ ์ 
      self.K = hyper_params['K'] #key๊ฐ 
      self.alpha = hyper_params['alpha'] #ํ์ต๋ฅ 
      self.beta = hyper_params['beta'] #์ ๊ทํ ๊ณ์ 
      self.iterations = hyper_params['iterations'] #๋ฐ๋ณต ํ์
      self.verbose = hyper_params['verbose'] #ํ์ต๊ณผ์  ์ถ๋ ฅ ์ฌ๋ถ ๊ฒฐ์  
    
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
  
    def get_prediction(self, i, j): 
      #์ฌ์ฉ์ i, ์์ดํ j์ ๋ํ ํ์  ์์ธก์น๋ฅผ ์์์ ๋ฐฐ์ ๋ ์์ ์ด์ฉํด์ ๊ตฌํ๋ค.
      prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i,:].dot(self.Q[j,].T) #์ ์ฒด ํ์  + ์ ์ ์ ๋ํ ํ๊ฐ ๊ฒฝํญ + ์์ดํ์ ๋ํ ํ๊ฐ ๊ฒฝํฅ + ์ฌ์ฉ์ ์์ธ๊ฐ*์์ดํ ์์ธ๊ฐ
      return prediction 
  
    def sgd(self): 
      for i,j,r in self.samples:  #i,j : ์ธ๋ฑ์ค, r : ํ์  
        #์ฌ์ฉ์ i, ์์ดํ j์ ๋ํ ํ์  ์์ธก์น ๊ณ์ฐ 
        prediction = self.get_prediction(i,j)
        #์ค์  ํ์ ๊ณผ ๋น๊ตํ ์ค์ฐจ ๊ณ์ฐ 
        e = (r-prediction)
        
        #์ฌ์ฉ์ ํ๊ฐ ๊ฒฝํฅ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
        self.b_u[i] += self.alpha * (e - (self.beta * self.b_u[i]))
        #์์ดํ ํ๊ฐ ๊ฒฝํฅ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
        self.b_d[j] += self.alpha * (e- (self.beta * self.b_d[j]))
  
        #P ํ๋ ฌ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
        self.P[i,:] += self.alpha * ((e * self.Q[j,:] - self.beta * self.P[i,:]))
        #Q ํ๋ ฌ ๊ณ์ฐ ๋ฐ ์๋ฐ์ดํธ
        self.Q[j,:] += self.alpha * ((e * self.P[i,:])- (self.beta * self.Q[j, :]))
  
    #Test set ์ ์  
    def set_test(self, ratings_test):
      test_set = []
      for i in range(len(ratings_test)):
        x,y,z = ratings_test.iloc[i]
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
      return self.get_prediction(user_id, item_id)
    
    def full_prediction(self): #์ ์ฒด ์์ธก 
      return self.b + self.b_u[:, np.newaxis] + self.b_d[np.newaxis, :] + self.P.dot(self.Q.T)
  
  ratings_train, ratings_test = train_test_split(ratings, 
                                                 test_size = 0.2,
                                                 shuffle = True,
                                                 random_state = 2021)
        
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
  ```

* ํฐ ๋ฐ์ดํฐ๋ฅผ `sparse matrix`๋ฅผ ํ์ฉํ์ฌ `numpy.array()`๋ฅผ ์ฌ์ฉํ๋ค๋ฉด ๋ถ๊ฐ๋ฅ ํ์ ๊ฒ์ ๊ฐ๋ฅํ๊ฒ ํจ.
