# πSection 06_ λ₯λ¬λμ μ¬μ©ν μΆμ² μμ€ν[β©](../../)

## contentsπ<a id='contents'></a>

* 0_ λ€μ΄κ°κΈ° μ μ[βοΈ](#0)
* 1_ Maxtrix Factorization(MF)μ μ κ²½λ§μΌλ‘ λ³ννκΈ°[βοΈ](#1)
* 2_ Kerasλ‘ MF κ΅¬ννκΈ°[βοΈ](#2)
* 3_ λ₯λ¬λμ μ μ©ν μΆμ² μμ€ν[βοΈ](#3)

## 0_ λ€μ΄κ°κΈ° μ μ[π](#contents)<a id='0'></a>

* λ₯λ¬λ(Deep Learning : DL)μ λ€μμ μλμΈ΅(hidden layer)μ κ°μ§ μΈκ³΅μ κ²½λ§μ μ μ©ν κΈ°λ²

## 1_ Maxtrix Factorization(MF)μ μ κ²½λ§μΌλ‘ λ³ννκΈ°[π](#contents)<a id='1'></a>

![](./image/6_1-1.png)

* `MF`λ₯Ό `Keras`λ‘
* input layer : κ° μ¬μ©μμ itemμΌλ‘ λΆν° μλ ₯μ λ°λ λΆλΆ
  * one-hot representation : μν«μΈμ½λ©κ³Ό κ°μ. 1μΈμ§ 0μΈμ§λ₯Ό binaryν ννλ‘ λ°κΏμ€. 
* μ¬μ©μμ One-Hot Represention μλ ₯

|        | feature 1 | feature 2 | feature 3 | ...  | feature M |
| ------ | --------- | --------- | --------- | ---- | --------- |
| User 1 | 1         | 0         | 0         | 0    | 0         |
| User 2 | 0         | 1         | 0         | 0    | 0         |
| User 3 | 0         | 0         | 1         | 0    | 0         |
| ...    | 0         | 0         | 0         | 1    | 0         |
| User M | 0         | 0         | 0         | 0    | 1         |

* embedding Layer : μ μ¬μμΈ Kλ₯Ό κ·μ 

  ![](./image/6_1-2.png)

  * μ¬μ©μ λΈλμ λͺ¨λ μ°κ²°λ¨. μ¦, νμ΄νκ° κ°―μκ° M*Kκ°κ° μ°κ²°λμ΄ μμ. 
  * ν μ¬μ©μλΉ Kκ°μ νμ΄νκ° μμ.
  * λ§μ½ itemμ΄λ©΄ N* Kκ° μ°κ²°λμ΄ μμ.

* MP : P(M * K)μ Q(N * K)κ° μ°κ²°λ¨. 

* Element-wise Product Layer

  ![](./image/6_1-3.png)

  * μ¬μ©μμ μμ΄νμ κ° νλ‘λνΈ μ°μ°μ μν layer
  * P * Q<sup>T</sup>

* μ¬μ©μμ μμ΄νμ νκ°κ²½ν₯(bias)

  ![](./image/6_1-4.png)

* μ΄ μ λ¦¬

  ![](./image/6_1-5.png)

  1. μ¬μ©μ μμ΄ν λκ°μ§μ μν« λ νλ¦¬μ  νμ΄μμ μ§νν¨.
  2. μ¬μ©μ μλ ₯μ Kκ°μ λΈλλ₯Ό κ°λ μ μ , μμ΄ν μλ² λ©κ³Ό μ°κ²°
  3. μ μ  μλ² λ©κ³Ό μμ΄ν μλ² λ©μ DOTνλ‘λνΈμ°μ°μΌλ‘ μ°κ²°
  4. λ€μ μλ ₯μΌλ‘ λμμ 1κ°μ μ¬μ©μκ° κ°κ° μμ΄ν νκ° κ²½ν₯, μ μ  νκ° κ²½ν₯ μλ² λ©κ³Ό μ°κ²°λ¨.
  5. λ§μ§λ§μΌλ‘ μ μ νκ° κ²½ν₯ μλ² λ©, μμ΄ν νκ° κ²½ν₯μλ² λ©, DOT μ΄ κ²°ν¨λ¨. 
  6. Flatttenμ μ°¨μμ μ€μ¬μ£Όλ μ­ν μ ν¨. 

* μμμ BUμ BDλ κ΅¬νμ νλλ°. B(μμ)λ₯Ό κ΅¬ννμ§ λͺ»ν¨.

  * μ μ²΄ νκ· μ μΌλ₯ μ μΌλ‘ λΉΌμ νΈμ±νκ² λ¨. 

## 2_ Kerasλ‘ MF κ΅¬ννκΈ°[π](#contents)<a id='2'></a>

* νμΌ λΆλ¬μ€κΈ°

  ```python
  # csv νμΌμμ λΆλ¬μ€κΈ°
  import  pandas as pd
  
  # train setκ³Ό test setμ λλκΈ° μν λΌμ΄λΈλ¬λ¦¬
  from sklearn.model_selection import train_test_split
  
  # νμν tensorflow λͺ¨λλ€μ κ°μ Έμ¨λ€.
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
  from tensorflow.keras.regularizers import l2
  from tensorflow.keras.optimizers import SGD, Adamax
  
  # DataFrame ννλ‘ λ°μ΄ν°λ₯Ό μ½μ΄μ¨λ€.
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv('./Data/u.data',
                          names=r_cols,
                          sep='\t',
                          encoding='latin-1')
  
  ratings_train, ratings_test = train_test_split(ratings,
                                                  test_size=0.2,
                                                  shuffle=True,
                                                  random_state=2021)
  ```

* λ°μ΄ν° μ€μ 

  ```python
  K = 200
  
  mu = ratings_train.rating.mean()
  
  M = ratings.user_id.max() + 1
  N = ratings.movie_id.max() + 1      # bias_comμ ν¬κΈ° 1μ κ°μνλ κ²!
  
  def RMSE(y_true, y_pred):
      return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
  ```

* P, Q μ¬μ©μ νκ° κ²½ν₯ μλ² λ©

  ```python
  user = Input(shape=(1,))
  item = Input(shape=(1,))
  
  P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)
  Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)
  
  user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)
  item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)
  ```

* μμ½ μΆλ ₯

  ```python
  R = layers.dot([P_embedding, Q_embedding], axes=(2,2)) # kμ kλΌλ¦¬ μ°μ°μ νκ² λ€.
  
  R = layers.add([R, user_bias, item_bias])
  
  R = Flatten()(R)
  
  model = Model(inputs=[user, item], outputs=R)
  model.compile(
      loss=RMSE,
      optimizer=SGD(),
      metrics=[RMSE]
  )
  
  model.summary()
  
  # μ€ν κ²°κ³Ό
  Model: "model"
  __________________________________________________________________________________________________
  Layer (type)                    Output Shape         Param #     Connected to                     
  ==================================================================================================
  input_3 (InputLayer)            [(None, 1)]          0                                            
  __________________________________________________________________________________________________
  input_4 (InputLayer)            [(None, 1)]          0                                            
  __________________________________________________________________________________________________
  embedding (Embedding)           (None, 1, 200)       188800      input_3[0][0]                    
  __________________________________________________________________________________________________
  embedding_1 (Embedding)         (None, 1, 200)       336600      input_4[0][0]                    
  __________________________________________________________________________________________________
  dot (Dot)                       (None, 1, 1)         0           embedding[0][0]                  
                                                                   embedding_1[0][0]                
  __________________________________________________________________________________________________
  embedding_2 (Embedding)         (None, 1, 1)         944         input_3[0][0]                    
  __________________________________________________________________________________________________
  embedding_3 (Embedding)         (None, 1, 1)         1683        input_4[0][0]                    
  __________________________________________________________________________________________________
  add (Add)                       (None, 1, 1)         0           dot[0][0]                        
                                                                   embedding_2[0][0]                
                                                                   embedding_3[0][0]                
  __________________________________________________________________________________________________
  flatten (Flatten)               (None, 1)            0           add[0][0]                        
  ==================================================================================================
  Total params: 528,027
  Trainable params: 528,027
  Non-trainable params: 0
  __________________________________________________________________________________________________
  
  ```

* λͺ¨λΈ μμ±

  ```python
  result = model.fit(
      x = [ratings_train.user_id.values,
      ratings_train.movie_id.values],
      y = ratings_train.values - mu,
  
      epochs=60,
      batch_size=256,
      validation_data=(
          [ratings_test.user_id.values,
          ratings_test.movie_id.values],
          ratings_test.rating.values - mu
      )
  
  )
  
  # μ€ν κ²°κ³Ό
  Epoch 1/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441776128.0000 - val_loss: 5.0007 - val_RMSE: 1.1188
  Epoch 2/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441776928.0000 - val_loss: 4.5440 - val_RMSE: 1.1186
  Epoch 3/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776448.0000 - RMSE: 441776960.0000 - val_loss: 4.1412 - val_RMSE: 1.1185
  Epoch 4/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776192.0000 - RMSE: 441776384.0000 - val_loss: 3.7860 - val_RMSE: 1.1184
  Epoch 5/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441775712.0000 - val_loss: 3.4727 - val_RMSE: 1.1184
  Epoch 6/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776352.0000 - RMSE: 441776704.0000 - val_loss: 3.1965 - val_RMSE: 1.1183
  Epoch 7/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441776544.0000 - val_loss: 2.9528 - val_RMSE: 1.1183
  Epoch 8/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776352.0000 - RMSE: 441776704.0000 - val_loss: 2.7380 - val_RMSE: 1.1183
  Epoch 9/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776448.0000 - RMSE: 441776544.0000 - val_loss: 2.5486 - val_RMSE: 1.1184
  Epoch 10/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441776512.0000 - val_loss: 2.3816 - val_RMSE: 1.1184
  Epoch 11/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776384.0000 - RMSE: 441777280.0000 - val_loss: 2.2343 - val_RMSE: 1.1184
  Epoch 12/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776192.0000 - RMSE: 441775808.0000 - val_loss: 2.1044 - val_RMSE: 1.1185
  Epoch 13/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776448.0000 - RMSE: 441776128.0000 - val_loss: 1.9900 - val_RMSE: 1.1185
  Epoch 14/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776128.0000 - RMSE: 441776928.0000 - val_loss: 1.8890 - val_RMSE: 1.1186
  Epoch 15/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776448.0000 - RMSE: 441776192.0000 - val_loss: 1.8001 - val_RMSE: 1.1186
  Epoch 16/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776192.0000 - RMSE: 441776512.0000 - val_loss: 1.7217 - val_RMSE: 1.1187
  Epoch 17/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776384.0000 - RMSE: 441776384.0000 - val_loss: 1.6525 - val_RMSE: 1.1188
  Epoch 18/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441775360.0000 - val_loss: 1.5916 - val_RMSE: 1.1188
  Epoch 19/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441775392.0000 - val_loss: 1.5379 - val_RMSE: 1.1189
  Epoch 20/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441776192.0000 - val_loss: 1.4906 - val_RMSE: 1.1190
  Epoch 21/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776384.0000 - RMSE: 441776288.0000 - val_loss: 1.4489 - val_RMSE: 1.1190
  Epoch 22/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441776608.0000 - val_loss: 1.4122 - val_RMSE: 1.1191
  Epoch 23/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776512.0000 - RMSE: 441775968.0000 - val_loss: 1.3798 - val_RMSE: 1.1191
  Epoch 24/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776544.0000 - RMSE: 441775936.0000 - val_loss: 1.3513 - val_RMSE: 1.1192
  Epoch 25/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776192.0000 - RMSE: 441775936.0000 - val_loss: 1.3262 - val_RMSE: 1.1193
  Epoch 26/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441776704.0000 - val_loss: 1.3041 - val_RMSE: 1.1193
  Epoch 27/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776448.0000 - RMSE: 441777280.0000 - val_loss: 1.2846 - val_RMSE: 1.1194
  Epoch 28/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776608.0000 - RMSE: 441777248.0000 - val_loss: 1.2674 - val_RMSE: 1.1194
  Epoch 29/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441776224.0000 - val_loss: 1.2523 - val_RMSE: 1.1195
  Epoch 30/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441776224.0000 - val_loss: 1.2391 - val_RMSE: 1.1195
  Epoch 31/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776352.0000 - RMSE: 441777376.0000 - val_loss: 1.2274 - val_RMSE: 1.1196
  Epoch 32/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776512.0000 - RMSE: 441776448.0000 - val_loss: 1.2171 - val_RMSE: 1.1196
  Epoch 33/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441775712.0000 - val_loss: 1.2080 - val_RMSE: 1.1196
  Epoch 34/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776352.0000 - RMSE: 441775968.0000 - val_loss: 1.2000 - val_RMSE: 1.1197
  Epoch 35/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776128.0000 - RMSE: 441776384.0000 - val_loss: 1.1930 - val_RMSE: 1.1197
  Epoch 36/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441776448.0000 - val_loss: 1.1869 - val_RMSE: 1.1197
  Epoch 37/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776352.0000 - RMSE: 441776192.0000 - val_loss: 1.1814 - val_RMSE: 1.1198
  Epoch 38/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441777248.0000 - val_loss: 1.1767 - val_RMSE: 1.1198
  Epoch 39/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776128.0000 - RMSE: 441775648.0000 - val_loss: 1.1725 - val_RMSE: 1.1198
  Epoch 40/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776512.0000 - RMSE: 441776768.0000 - val_loss: 1.1688 - val_RMSE: 1.1199
  Epoch 41/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776128.0000 - RMSE: 441775360.0000 - val_loss: 1.1656 - val_RMSE: 1.1199
  Epoch 42/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776384.0000 - RMSE: 441776512.0000 - val_loss: 1.1628 - val_RMSE: 1.1199
  Epoch 43/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441776128.0000 - val_loss: 1.1603 - val_RMSE: 1.1200
  Epoch 44/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441776128.0000 - val_loss: 1.1581 - val_RMSE: 1.1200
  Epoch 45/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776544.0000 - RMSE: 441776512.0000 - val_loss: 1.1562 - val_RMSE: 1.1200
  Epoch 46/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776352.0000 - RMSE: 441775808.0000 - val_loss: 1.1545 - val_RMSE: 1.1200
  Epoch 47/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776224.0000 - RMSE: 441775968.0000 - val_loss: 1.1530 - val_RMSE: 1.1200
  Epoch 48/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776448.0000 - RMSE: 441776544.0000 - val_loss: 1.1517 - val_RMSE: 1.1201
  Epoch 49/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776512.0000 - RMSE: 441776384.0000 - val_loss: 1.1506 - val_RMSE: 1.1201
  Epoch 50/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776352.0000 - RMSE: 441776512.0000 - val_loss: 1.1496 - val_RMSE: 1.1201
  Epoch 51/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776352.0000 - RMSE: 441776224.0000 - val_loss: 1.1488 - val_RMSE: 1.1201
  Epoch 52/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776192.0000 - RMSE: 441776032.0000 - val_loss: 1.1480 - val_RMSE: 1.1201
  Epoch 53/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441776512.0000 - val_loss: 1.1474 - val_RMSE: 1.1201
  Epoch 54/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776448.0000 - RMSE: 441775936.0000 - val_loss: 1.1468 - val_RMSE: 1.1201
  Epoch 55/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776512.0000 - RMSE: 441776768.0000 - val_loss: 1.1463 - val_RMSE: 1.1202
  Epoch 56/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441775936.0000 - val_loss: 1.1459 - val_RMSE: 1.1202
  Epoch 57/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776384.0000 - RMSE: 441776512.0000 - val_loss: 1.1455 - val_RMSE: 1.1202
  Epoch 58/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776288.0000 - RMSE: 441776768.0000 - val_loss: 1.1452 - val_RMSE: 1.1202
  Epoch 59/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776192.0000 - RMSE: 441776352.0000 - val_loss: 1.1449 - val_RMSE: 1.1202
  Epoch 60/60
  313/313 [==============================] - 1s 3ms/step - loss: 441776192.0000 - RMSE: 441776224.0000 - val_loss: 1.1446 - val_RMSE: 1.1202
  ```

* RMSE μκ°ν

  ```python
  # plot RMSE
  import matplotlib.pyplot as plt
  
  plt.plot(result.history['RMSE'], label="Train RMSE")
  plt.plot(result.history['val_RMSE'], label = 'Test RMSE')
  plt.legend()
  plt.show()
  ```

  ![](./image/6_2.png)

* μ μ©ν΄λ³΄κΈ°

  ```python
  user_ids = ratings_test.user_id.values[0:6] # μ μ©ν΄λ³΄κΈ°
  movie_ids = ratings_test.movie_id.values[0:6]
  
  predictions = model.predict([user_ids, movie_ids]) + mu #μ μ²΄ νκ·  λ€μ λνκΈ°
  # μ€μ  κ°
  print(ratings_test[0:6])
         user_id  movie_id  rating  timestamp
  23307      468        51       3  875293386
  36679       92       780       3  875660494
  36626      555       489       5  879975455
  83753      940        69       2  885921265
  52604      181      1350       1  878962120
  49877      320       195       5  884749255
  # μμΈ‘ κ°
  print(predictions)
  [[3.5550046]
   [3.4723089]
   [3.5492196]
   [3.5690255]
   [3.1871848]
   [3.5966241]]
  ```

* RMSE μ€μ 

  ```python
  import numpy as np
  
  def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))
  ```

* RMSE κ³μ°

  ```python
  user_ids = ratings_test.user_id.values
  movie_ids = ratings_test.movie_id.values
  
  y_pred = model.predict([user_ids, movie_ids]) + mu
  y_pred = np.ravel(y_pred,order="C") #1μ°¨μ ννλ‘ λ°κΏμ€.
  
  y_true = np.array(ratings_test.rating)
  
  RMSE2(y_true, y_pred)
  
  # μ€ν κ²°κ³Ό
  1.0913778530076552
  ```

  * μ΅μ νλ νλμ΄ μλμ΄ μκΈ°λλ¬Έμ μ±λ₯μ΄ μ’μ§ μμ.

## 3_ λ₯λ¬λμ μ μ©ν μΆμ² μμ€ν[π](#contents)<a id='3'></a>

![](./image/6_3-1.png)

* RAW data λ₯Ό embedding
* μ΄μ κ³Ό λ€λ₯Έ μ μ `μλμΈ΅`(hidden layer)λ₯Ό μΆκ°ν¨.

![](./image/6_3-2.png)

* layer1 : user latent vector + item latent vector = `concanating`(κ²°ν©)μ μ΄μ©

* κ΅¬ν

  ```python
  import pandas as pd
  import numpy as np
  
  #train setκ³Ό test setμ λλκΈ° μν λΌμ΄λΈλ¬λ¦¬
  from sklearn.model_selection import train_test_split
  
  #νμν tensorflow λͺ¨λλ€μ κ°μ Έμ¨λ€.
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
  from tensorflow.keras.regularizers import l2
  from tensorflow.keras.optimizers import SGD, Adamax
  
  #layer κ΅¬μ±μ νμν λΌμ΄λΈλ¬λ¦¬ λΆλ¬μ€κΈ° 
  from tensorflow.keras.layers import Dense, Concatenate, Activation # μ΄ λΆλΆμ λ μ΄μ΄ κ΅¬μ±μ μν΄ νμν λ¦¬μλΈλ¬λ¦¬
  from tensorflow.keras.regularizers import l2 
  from tensorflow.keras.optimizers import SGD, Adamax
  
  #DataFrame ννλ‘ λ°μ΄ν°λ₯Ό μ½μ΄μ¨λ€.
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv('./Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
  
  ratings_train, ratings_test = train_test_split(ratings,
                                                 test_size = 0.2,
                                                 shuffle = True,
                                                 random_state = 2021)
  ### Defining RMSE measure ###
  # y_true, y_predμ μ κ²½λ§μμ μ€μ κ°, μμΈ‘κ°μ λνλ΄λ Tensorflow/Keras νμ€ λ³μ
  def RMSE(y_true, y_pred):
    # Tensorflowμ mathν΄λμ€μ λ―Έλ¦¬ μ μλ
    # μ κ³±κ·Ό(sqrt), νκ· (reduce_mean), μ κ³±(square) ν¨μλ₯Ό ν΅ν΄ RMSE κ³μ°
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))
  
  ### Variable μ΄κΈ°ν ###
  #μ μ¬μμΈ μ 200μΌλ‘ μ§μ νλ€.
  K = 200
  
  #μ μ²΄ νκ·  κ³μ°νλ€.
  mu = ratings_train.rating.mean()
  
  #μ¬μ©μ μμ΄λμ μν μμ΄λμ μ΅λκ° -> λ³΄ν΅μ uniqueν κ°μ κ°μ + 1λ‘ ν΄μΌν¨ 
  #1μ λνλ μ΄μ  : bias term μΆκ° κ³ λ €
  M = ratings.user_id.max() + 1
  N = ratings.movie_id.max() + 1
  
  #################################################################################
  #kreas λͺ¨λΈ 
  # μλλΆλΆμ μμμμ λμΌνκ² μ¬μ©μμ μμ΄ν λ°μ΄ν°λ₯Ό embeddingμ ν΅ν΄
  # κ°κ° Kκ°μ λΈλλ₯Ό κ°λ layerλ‘ λ³ννκ³ 
  # μ¬μ©μ biasμ μμ΄ν biasλ₯Ό 1κ°μ λΈλλ₯Ό κ°λ layerλ‘ λ³ννλ€.
  user = Input(shape=(1,))
  item = Input(shape=(1,))
  
  #Embedding
  P_embedding = Embedding(M,K,embeddings_regularizer=l2())(user) #regularizer : κ·μ  -> κ³Όμ ν© λ°©μ§
  Q_embedding = Embedding(N,K,embeddings_regularizer=l2())(item)
  
  #bias
  user_bias = Embedding(M,1,embeddings_regularizer=l2())(user)
  item_bias = Embedding(N,1,embeddings_regularizer=l2())(item)
  
  #μκ³Ό λ€λ₯Ό νμ€λ‘ λΆμ΄κΈ°
  # μ΄λ₯Ό μν΄μ 1μνμΌλ‘ λ°μ΄ν°λ₯Ό λ§λ¦.
  P_embedding = Flatten()(P_embedding)
  Q_embedding = Flatten()(Q_embedding)
  user_bias = Flatten()(user_bias)
  item_bias = Flatten()(item_bias)
  
  R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias])
  
  R = Dense(2048)(R) #λΈλκ° 2048κ°μΈ νλμ layerλ₯Ό λ§λ  ν Rκ³Ό μ°κ²° 
  R = Activation('linear')(R) # μμ μλ Denseλ μ΄μ΄λ₯Ό activation functionμ μ§μ ν¨.
  
  R = Dense(256)(R)
  R = Activation('linear')(R)
  
  R = Dense(1)(R) #μΆλ ₯ layer
  
  model = Model(inputs = [user,item],outputs=R)
  
  model.compile(loss=RMSE,
      optimizer=SGD(), #Adamaxλ κ°λ₯
      metrics = [RMSE]
  )
  
  model.summary()
  
  # μ€ν κ²°κ³Ό
  Model: "model_1"
  __________________________________________________________________________________________________
  Layer (type)                    Output Shape         Param #     Connected to                     
  ==================================================================================================
  input_3 (InputLayer)            [(None, 1)]          0                                            
  __________________________________________________________________________________________________
  input_4 (InputLayer)            [(None, 1)]          0                                            
  __________________________________________________________________________________________________
  embedding_4 (Embedding)         (None, 1, 200)       188800      input_3[0][0]                    
  __________________________________________________________________________________________________
  embedding_5 (Embedding)         (None, 1, 200)       336600      input_4[0][0]                    
  __________________________________________________________________________________________________
  embedding_6 (Embedding)         (None, 1, 1)         944         input_3[0][0]                    
  __________________________________________________________________________________________________
  embedding_7 (Embedding)         (None, 1, 1)         1683        input_4[0][0]                    
  __________________________________________________________________________________________________
  flatten_1 (Flatten)             (None, 200)          0           embedding_4[0][0]                
  __________________________________________________________________________________________________
  flatten_2 (Flatten)             (None, 200)          0           embedding_5[0][0]                
  __________________________________________________________________________________________________
  flatten_3 (Flatten)             (None, 1)            0           embedding_6[0][0]                
  __________________________________________________________________________________________________
  flatten_4 (Flatten)             (None, 1)            0           embedding_7[0][0]                
  __________________________________________________________________________________________________
  concatenate (Concatenate)       (None, 402)          0           flatten_1[0][0]                  
                                                                   flatten_2[0][0]                  
                                                                   flatten_3[0][0]                  
                                                                   flatten_4[0][0]                  
  __________________________________________________________________________________________________
  dense (Dense)                   (None, 2048)         825344      concatenate[0][0]                
  __________________________________________________________________________________________________
  activation (Activation)         (None, 2048)         0           dense[0][0]                      
  __________________________________________________________________________________________________
  dense_1 (Dense)                 (None, 256)          524544      activation[0][0]                 
  __________________________________________________________________________________________________
  activation_1 (Activation)       (None, 256)          0           dense_1[0][0]                    
  __________________________________________________________________________________________________
  dense_2 (Dense)                 (None, 1)            257         activation_1[0][0]               
  ==================================================================================================
  Total params: 1,878,172
  Trainable params: 1,878,172
  Non-trainable params: 0
  __________________________________________________________________________________________________
  ```

* λͺ¨λΈ κ΅¬μ±

  ```python
  # Model fitting
  # λͺ¨λΈ μλ ₯μ νμν λ°μ΄ν° μ λ¦¬
  train_user_ids = ratings_train.user_id.values
  train_movie_ids = ratings_train.movie_id.values
  train_ratings = ratings_train.rating.values
  
  test_user_ids = ratings_test.user_id.values
  test_movie_ids = ratings_test.movie_id.values
  test_ratings = ratings_test.rating.values
  
  #μ κ²½λ§ νμ΅
  result = model.fit(
      x = [train_user_ids, train_movie_ids],
      y = train_ratings - mu, #μ μ²΄ νκ·  λΉΌκΈ° 
      epochs = 65,
      batch_size = 512, #batch_size : μ μ²΄ train_setμμ 512κ°μ© νμ΅μν€κ² λ€.
      validation_data = (
          [test_user_ids, test_movie_ids],
           test_ratings - mu
      )
  )
  
  # μ€ν κ²°κ³Ό
  Epoch 1/65
  157/157 [==============================] - 4s 20ms/step - loss: 5.3929 - RMSE: 1.1267 - val_loss: 5.2497 - val_RMSE: 1.1177
  Epoch 2/65
  157/157 [==============================] - 3s 20ms/step - loss: 5.1316 - RMSE: 1.1248 - val_loss: 4.9967 - val_RMSE: 1.1162
  Epoch 3/65
  157/157 [==============================] - 3s 21ms/step - loss: 4.8863 - RMSE: 1.1233 - val_loss: 4.7592 - val_RMSE: 1.1147
  Epoch 4/65
  157/157 [==============================] - 3s 20ms/step - loss: 4.6559 - RMSE: 1.1225 - val_loss: 4.5361 - val_RMSE: 1.1133
  Epoch 5/65
  157/157 [==============================] - 3s 20ms/step - loss: 4.4394 - RMSE: 1.1207 - val_loss: 4.3265 - val_RMSE: 1.1119
  Epoch 6/65
  157/157 [==============================] - 3s 20ms/step - loss: 4.2360 - RMSE: 1.1188 - val_loss: 4.1296 - val_RMSE: 1.1104
  Epoch 7/65
  157/157 [==============================] - 3s 21ms/step - loss: 4.0449 - RMSE: 1.1173 - val_loss: 3.9444 - val_RMSE: 1.1087
  Epoch 8/65
  157/157 [==============================] - 3s 20ms/step - loss: 3.8653 - RMSE: 1.1156 - val_loss: 3.7702 - val_RMSE: 1.1069
  Epoch 9/65
  157/157 [==============================] - 3s 20ms/step - loss: 3.6962 - RMSE: 1.1134 - val_loss: 3.6065 - val_RMSE: 1.1049
  Epoch 10/65
  157/157 [==============================] - 3s 21ms/step - loss: 3.5371 - RMSE: 1.1115 - val_loss: 3.4523 - val_RMSE: 1.1025
  Epoch 11/65
  157/157 [==============================] - 3s 21ms/step - loss: 3.3874 - RMSE: 1.1087 - val_loss: 3.3073 - val_RMSE: 1.1000
  Epoch 12/65
  157/157 [==============================] - 3s 21ms/step - loss: 3.2463 - RMSE: 1.1057 - val_loss: 3.1704 - val_RMSE: 1.0967
  Epoch 13/65
  157/157 [==============================] - 3s 20ms/step - loss: 3.1132 - RMSE: 1.1021 - val_loss: 3.0417 - val_RMSE: 1.0935
  Epoch 14/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.9875 - RMSE: 1.0984 - val_loss: 2.9194 - val_RMSE: 1.0890
  Epoch 15/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.8687 - RMSE: 1.0941 - val_loss: 2.8041 - val_RMSE: 1.0840
  Epoch 16/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.7562 - RMSE: 1.0881 - val_loss: 2.6950 - val_RMSE: 1.0785
  Epoch 17/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.6496 - RMSE: 1.0825 - val_loss: 2.5915 - val_RMSE: 1.0721
  Epoch 18/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.5483 - RMSE: 1.0756 - val_loss: 2.4933 - val_RMSE: 1.0648
  Epoch 19/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.4521 - RMSE: 1.0669 - val_loss: 2.4002 - val_RMSE: 1.0571
  Epoch 20/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.3606 - RMSE: 1.0582 - val_loss: 2.3107 - val_RMSE: 1.0476
  Epoch 21/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.2735 - RMSE: 1.0490 - val_loss: 2.2264 - val_RMSE: 1.0381
  Epoch 22/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.1910 - RMSE: 1.0396 - val_loss: 2.1465 - val_RMSE: 1.0285
  Epoch 23/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.1125 - RMSE: 1.0295 - val_loss: 2.0711 - val_RMSE: 1.0191
  Epoch 24/65
  157/157 [==============================] - 3s 21ms/step - loss: 2.0385 - RMSE: 1.0192 - val_loss: 1.9992 - val_RMSE: 1.0092
  Epoch 25/65
  157/157 [==============================] - 3s 21ms/step - loss: 1.9688 - RMSE: 1.0090 - val_loss: 1.9322 - val_RMSE: 1.0005
  Epoch 26/65
  157/157 [==============================] - 3s 21ms/step - loss: 1.9033 - RMSE: 1.0003 - val_loss: 1.8695 - val_RMSE: 0.9925
  Epoch 27/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.8419 - RMSE: 0.9922 - val_loss: 1.8103 - val_RMSE: 0.9849
  Epoch 28/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.7844 - RMSE: 0.9845 - val_loss: 1.7553 - val_RMSE: 0.9784
  Epoch 29/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.7305 - RMSE: 0.9781 - val_loss: 1.7042 - val_RMSE: 0.9729
  Epoch 30/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.6803 - RMSE: 0.9723 - val_loss: 1.6562 - val_RMSE: 0.9679
  Epoch 31/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.6334 - RMSE: 0.9662 - val_loss: 1.6110 - val_RMSE: 0.9632
  Epoch 32/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.5894 - RMSE: 0.9619 - val_loss: 1.5692 - val_RMSE: 0.9592
  Epoch 33/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.5483 - RMSE: 0.9578 - val_loss: 1.5323 - val_RMSE: 0.9582
  Epoch 34/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.5099 - RMSE: 0.9537 - val_loss: 1.4935 - val_RMSE: 0.9528
  Epoch 35/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.4740 - RMSE: 0.9503 - val_loss: 1.4595 - val_RMSE: 0.9505
  Epoch 36/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.4402 - RMSE: 0.9471 - val_loss: 1.4266 - val_RMSE: 0.9473
  Epoch 37/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.4087 - RMSE: 0.9441 - val_loss: 1.3974 - val_RMSE: 0.9460
  Epoch 38/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.3793 - RMSE: 0.9421 - val_loss: 1.3688 - val_RMSE: 0.9435
  Epoch 39/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.3515 - RMSE: 0.9398 - val_loss: 1.3424 - val_RMSE: 0.9418
  Epoch 40/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.3257 - RMSE: 0.9378 - val_loss: 1.3174 - val_RMSE: 0.9399
  Epoch 41/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.3014 - RMSE: 0.9355 - val_loss: 1.2954 - val_RMSE: 0.9397
  Epoch 42/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2786 - RMSE: 0.9342 - val_loss: 1.2726 - val_RMSE: 0.9373
  Epoch 43/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2575 - RMSE: 0.9331 - val_loss: 1.2521 - val_RMSE: 0.9361
  Epoch 44/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2375 - RMSE: 0.9320 - val_loss: 1.2359 - val_RMSE: 0.9378
  Epoch 45/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2187 - RMSE: 0.9306 - val_loss: 1.2158 - val_RMSE: 0.9349
  Epoch 46/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2011 - RMSE: 0.9296 - val_loss: 1.1984 - val_RMSE: 0.9334
  Epoch 47/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1849 - RMSE: 0.9282 - val_loss: 1.1827 - val_RMSE: 0.9328
  Epoch 48/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1696 - RMSE: 0.9287 - val_loss: 1.1705 - val_RMSE: 0.9348
  Epoch 49/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1549 - RMSE: 0.9272 - val_loss: 1.1541 - val_RMSE: 0.9316
  Epoch 50/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1416 - RMSE: 0.9269 - val_loss: 1.1438 - val_RMSE: 0.9338
  Epoch 51/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1289 - RMSE: 0.9261 - val_loss: 1.1300 - val_RMSE: 0.9317
  Epoch 52/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1171 - RMSE: 0.9258 - val_loss: 1.1184 - val_RMSE: 0.9313
  Epoch 53/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1059 - RMSE: 0.9253 - val_loss: 1.1068 - val_RMSE: 0.9301
  Epoch 54/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0952 - RMSE: 0.9245 - val_loss: 1.0984 - val_RMSE: 0.9313
  Epoch 55/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0856 - RMSE: 0.9243 - val_loss: 1.0873 - val_RMSE: 0.9296
  Epoch 56/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0763 - RMSE: 0.9249 - val_loss: 1.0819 - val_RMSE: 0.9328
  Epoch 57/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0676 - RMSE: 0.9237 - val_loss: 1.0702 - val_RMSE: 0.9292
  Epoch 58/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0595 - RMSE: 0.9235 - val_loss: 1.0627 - val_RMSE: 0.9292
  Epoch 59/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0521 - RMSE: 0.9234 - val_loss: 1.0547 - val_RMSE: 0.9286
  Epoch 60/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0446 - RMSE: 0.9233 - val_loss: 1.0482 - val_RMSE: 0.9289
  Epoch 61/65
  157/157 [==============================] - 3s 21ms/step - loss: 1.0379 - RMSE: 0.9234 - val_loss: 1.0424 - val_RMSE: 0.9293
  Epoch 62/65
  157/157 [==============================] - 3s 21ms/step - loss: 1.0319 - RMSE: 0.9235 - val_loss: 1.0352 - val_RMSE: 0.9282
  Epoch 63/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0259 - RMSE: 0.9232 - val_loss: 1.0344 - val_RMSE: 0.9333
  Epoch 64/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0203 - RMSE: 0.9233 - val_loss: 1.0244 - val_RMSE: 0.9284
  Epoch 65/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0147 - RMSE: 0.9227 - val_loss: 1.0206 - val_RMSE: 0.9297
  ```

* μ΅μ μ λͺ¨λΈ κ·Έλν

  ```python
  #plot RMSE
  import matplotlib.pyplot as plt 
  plt.plot(result.history['RMSE'], label = 'Train RMSE')
  plt.plot(result.history['val_RMSE'], label = 'Test RMSE')
  plt.legend()
  plt.show()
  ```

  ![](./image/6_3-3.png)

## 6.4 λ₯λ¬λ λͺ¨λΈμ λ³μ μΆκ°νκΈ°[π](#contents)<a id='4'></a>

* 'μ§μ' λ³μλ₯Ό μΆκ°ν λ₯λ¬λ λͺ¨λΈ

  ```python
  # csv νμΌμμ λΆλ¬μ€κΈ°
  import pandas as pd
  
  #train setκ³Ό test setμ λλκΈ° μν λΌμ΄λΈλ¬λ¦¬
  from sklearn.model_selection import train_test_split
  
  #νμν tensorflow λͺ¨λλ€μ κ°μ Έμ¨λ€.
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
  from tensorflow.keras.regularizers import l2
  from tensorflow.keras.optimizers import SGD, Adamax
  from tensorflow.keras.layers import Dense, Concatenate, Activation
  
  ### Defining RMSE measure ###
  # y_true, y_predμ μ κ²½λ§μμ μ€μ κ°, μμΈ‘κ°μ λνλ΄λ Tensorflow/Keras νμ€ λ³μ
  def RMSE(y_true, y_pred):
    # Tensorflowμ mathν΄λμ€μ λ―Έλ¦¬ μ μλ
    # μ κ³±κ·Ό(sqrt), νκ· (reduce_mean), μ κ³±(square) ν¨μλ₯Ό ν΅ν΄ RMSE κ³μ°
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))
  
  #DataFrame ννλ‘ λ°μ΄ν°λ₯Ό μ½μ΄μ¨λ€.
  r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
  ratings = pd.read_csv('./Data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
  
  ratings_train, ratings_test = train_test_split(ratings,
                                                 test_size = 0.2,
                                                 shuffle = True,
                                                 random_state = 2021)
  
  #μ¬μ©μ(user) λ°μ΄ν° κ°μ Έμ€κΈ° 
  u_cols = ["user_id","age",'sex',"occupation","zip_code"]
  users = pd.read_csv('./Data/u.user',
                      sep = '|',
                      names = u_cols,
                      encoding='latin-1')
  #μ¬μ©μ IDμ μ§μλ§ λ¨κΈ΄λ€.
  users = users[['user_id','occupation']]
  
  occupation = {} # μ§μμ dict ννλ‘
  def convert_occ(x):
    if x in occupation:
      return occupation[x]
    else: 
      occupation[x] = len(occupation) # μ§μμ λν΄ μΈλ±μ€λ₯Ό λΆμ¬ ex) {μ μλ : 0}
      return occupation[x]
  
  users['occupation'] = users['occupation'].apply(convert_occ)
  
  L = len(occupation) # bias termμ΄ μκΈ° λλ¬Έμ +1 νμ X
  
  train_occ = pd.merge(ratings_train, users, on = 'user_id')['occupation']
  test_occ = pd.merge(ratings_test, users, on = 'user_id')['occupation']
  
  #μ μ¬μμΈ μ 
  K = 200
  
  #μ μ²΄ νκ·  κ³μ°
  mu = ratings_train.rating.mean()
  
  #μ¬μ©μ μμ΄λμ μν μμ΄λμ μ΅λκ° -> λ³΄ν΅μ uniqueν κ°μ κ°μ + 1λ‘ ν΄μΌν¨ 
  #1μ λνλ μ΄μ  : bias term μΆκ° κ³ λ €
  M = ratings.user_id.max() + 1
  N = ratings.movie_id.max() + 1
  
  #kreas λͺ¨λΈ 
  user = Input(shape=(1,))
  item = Input(shape=(1,))
  
  #Embedding
  P_embedding = Embedding(M,K,embeddings_regularizer=l2())(user) #regularizer : κ·μ  -> κ³Όμ ν© λ°©μ§
  Q_embedding = Embedding(N,K,embeddings_regularizer=l2())(item)
  
  #bias
  user_bias = Embedding(M,1,embeddings_regularizer=l2())(user)
  item_bias = Embedding(N,1,embeddings_regularizer=l2())(item)
  
  #μκ³Ό λ€λ₯Ό νμ€λ‘ λΆμ΄κΈ°μν΄ Flatten μν 
  P_embedding = Flatten()(P_embedding)
  Q_embedding = Flatten()(Q_embedding)
  
  user_bias = Flatten()(user_bias)
  item_bias = Flatten()(item_bias)
  
  #μ§μ λ³μ μΆκ° 
  occ = Input(shape = (1,))
  OCC_embedding = Embedding(L,3, embeddings_regularizer=l2())(occ)
  OCC_layer = Flatten()(OCC_embedding) 
  
  R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias, OCC_layer])
  
  R = Dense(2048)(R) #λΈλκ° 2048κ°μΈ νλμ layerλ₯Ό λ§λ  ν Rκ³Ό μ°κ²° 
  R = Activation('linear')(R) 
  
  R = Dense(256)(R)
  R = Activation('linear')(R)
  
  R = Dense(1)(R) #μΆλ ₯ layer
  
  model = Model(inputs = [user, item, occ],outputs=R) # μ§μμ΄λΌλ λ³μκ° μΆκ°λ¨.
  model.compile(loss=RMSE,
                optimizer=SGD(), #Adamaxλ κ°λ₯
                metrics = [RMSE])
  
  model.summary()
  
  # μ€ν κ²°κ³Ό
  Model: "model_2"
  __________________________________________________________________________________________________
  Layer (type)                    Output Shape         Param #     Connected to                     
  ==================================================================================================
  input_5 (InputLayer)            [(None, 1)]          0                                            
  __________________________________________________________________________________________________
  input_6 (InputLayer)            [(None, 1)]          0                                            
  __________________________________________________________________________________________________
  input_7 (InputLayer)            [(None, 1)]          0                                            
  __________________________________________________________________________________________________
  embedding_8 (Embedding)         (None, 1, 200)       188800      input_5[0][0]                    
  __________________________________________________________________________________________________
  embedding_9 (Embedding)         (None, 1, 200)       336600      input_6[0][0]                    
  __________________________________________________________________________________________________
  embedding_10 (Embedding)        (None, 1, 1)         944         input_5[0][0]                    
  __________________________________________________________________________________________________
  embedding_11 (Embedding)        (None, 1, 1)         1683        input_6[0][0]                    
  __________________________________________________________________________________________________
  embedding_12 (Embedding)        (None, 1, 3)         63          input_7[0][0]                    
  __________________________________________________________________________________________________
  flatten_5 (Flatten)             (None, 200)          0           embedding_8[0][0]                
  __________________________________________________________________________________________________
  flatten_6 (Flatten)             (None, 200)          0           embedding_9[0][0]                
  __________________________________________________________________________________________________
  flatten_7 (Flatten)             (None, 1)            0           embedding_10[0][0]               
  __________________________________________________________________________________________________
  flatten_8 (Flatten)             (None, 1)            0           embedding_11[0][0]               
  __________________________________________________________________________________________________
  flatten_9 (Flatten)             (None, 3)            0           embedding_12[0][0]               
  __________________________________________________________________________________________________
  concatenate_1 (Concatenate)     (None, 405)          0           flatten_5[0][0]                  
                                                                   flatten_6[0][0]                  
                                                                   flatten_7[0][0]                  
                                                                   flatten_8[0][0]                  
                                                                   flatten_9[0][0]                  
  __________________________________________________________________________________________________
  dense_3 (Dense)                 (None, 2048)         831488      concatenate_1[0][0]              
  __________________________________________________________________________________________________
  activation_2 (Activation)       (None, 2048)         0           dense_3[0][0]                    
  __________________________________________________________________________________________________
  dense_4 (Dense)                 (None, 256)          524544      activation_2[0][0]               
  __________________________________________________________________________________________________
  activation_3 (Activation)       (None, 256)          0           dense_4[0][0]                    
  __________________________________________________________________________________________________
  dense_5 (Dense)                 (None, 1)            257         activation_3[0][0]               
  ==================================================================================================
  Total params: 1,884,379
  Trainable params: 1,884,379
  Non-trainable params: 0
  __________________________________________________________________________________________________
  
  ```

* λͺ¨λΈ κ΅¬μ±

  ```python
  # Model fitting
  # λͺ¨λΈ μλ ₯μ νμν λ°μ΄ν° μ λ¦¬
  train_user_ids = ratings_train.user_id.values
  train_movie_ids = ratings_train.movie_id.values
  train_ratings = ratings_train.rating.values
  train_occs = train_occ.values
  
  test_user_ids = ratings_test.user_id.values
  test_movie_ids = ratings_test.movie_id.values
  test_ratings = ratings_test.rating.values
  test_occs = test_occ.values
  
  #μ κ²½λ§ νμ΅
  result = model.fit(
      x = [train_user_ids, train_movie_ids, train_occs],
      y =  train_ratings - mu, #μ μ²΄ νκ·  λΉΌκΈ° 
      epochs = 65,
      batch_size = 512, #batch_size : μ μ²΄ train_setμμ 512κ°μ© νμ΅μν€κ² λ€.
      validation_data = (
          [test_user_ids, test_movie_ids, test_occs],
          test_ratings- mu
      )
  )
  
  # μ€ν κ²°κ³Ό
  Epoch 1/65
  157/157 [==============================] - 4s 21ms/step - loss: 5.3868 - RMSE: 1.1266 - val_loss: 5.2440 - val_RMSE: 1.1174
  Epoch 2/65
  157/157 [==============================] - 3s 20ms/step - loss: 5.1256 - RMSE: 1.1245 - val_loss: 4.9914 - val_RMSE: 1.1159
  Epoch 3/65
  157/157 [==============================] - 3s 20ms/step - loss: 4.8806 - RMSE: 1.1227 - val_loss: 4.7538 - val_RMSE: 1.1141
  Epoch 4/65
  157/157 [==============================] - 3s 21ms/step - loss: 4.6502 - RMSE: 1.1210 - val_loss: 4.5308 - val_RMSE: 1.1125
  Epoch 5/65
  157/157 [==============================] - 3s 21ms/step - loss: 4.4338 - RMSE: 1.1190 - val_loss: 4.3213 - val_RMSE: 1.1108
  Epoch 6/65
  157/157 [==============================] - 3s 20ms/step - loss: 4.2305 - RMSE: 1.1175 - val_loss: 4.1244 - val_RMSE: 1.1090
  Epoch 7/65
  157/157 [==============================] - 3s 20ms/step - loss: 4.0395 - RMSE: 1.1158 - val_loss: 3.9393 - val_RMSE: 1.1071
  Epoch 8/65
  157/157 [==============================] - 3s 20ms/step - loss: 3.8597 - RMSE: 1.1139 - val_loss: 3.7652 - val_RMSE: 1.1050
  Epoch 9/65
  157/157 [==============================] - 3s 21ms/step - loss: 3.6906 - RMSE: 1.1114 - val_loss: 3.6013 - val_RMSE: 1.1026
  Epoch 10/65
  157/157 [==============================] - 3s 21ms/step - loss: 3.5313 - RMSE: 1.1088 - val_loss: 3.4470 - val_RMSE: 1.0997
  Epoch 11/65
  157/157 [==============================] - 3s 20ms/step - loss: 3.3813 - RMSE: 1.1049 - val_loss: 3.3016 - val_RMSE: 1.0966
  Epoch 12/65
  157/157 [==============================] - 3s 21ms/step - loss: 3.2400 - RMSE: 1.1024 - val_loss: 3.1645 - val_RMSE: 1.0929
  Epoch 13/65
  157/157 [==============================] - 3s 21ms/step - loss: 3.1064 - RMSE: 1.0982 - val_loss: 3.0349 - val_RMSE: 1.0885
  Epoch 14/65
  157/157 [==============================] - 3s 21ms/step - loss: 2.9802 - RMSE: 1.0929 - val_loss: 2.9125 - val_RMSE: 1.0835
  Epoch 15/65
  157/157 [==============================] - 3s 21ms/step - loss: 2.8608 - RMSE: 1.0876 - val_loss: 2.7968 - val_RMSE: 1.0779
  Epoch 16/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.7476 - RMSE: 1.0814 - val_loss: 2.6871 - val_RMSE: 1.0715
  Epoch 17/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.6404 - RMSE: 1.0741 - val_loss: 2.5825 - val_RMSE: 1.0637
  Epoch 18/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.5382 - RMSE: 1.0661 - val_loss: 2.4835 - val_RMSE: 1.0554
  Epoch 19/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.4414 - RMSE: 1.0572 - val_loss: 2.3895 - val_RMSE: 1.0465
  Epoch 20/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.3491 - RMSE: 1.0476 - val_loss: 2.3002 - val_RMSE: 1.0369
  Epoch 21/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.2617 - RMSE: 1.0378 - val_loss: 2.2156 - val_RMSE: 1.0271
  Epoch 22/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.1789 - RMSE: 1.0275 - val_loss: 2.1356 - val_RMSE: 1.0173
  Epoch 23/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.1006 - RMSE: 1.0175 - val_loss: 2.0608 - val_RMSE: 1.0084
  Epoch 24/65
  157/157 [==============================] - 3s 20ms/step - loss: 2.0269 - RMSE: 1.0079 - val_loss: 1.9891 - val_RMSE: 0.9987
  Epoch 25/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.9576 - RMSE: 0.9987 - val_loss: 1.9224 - val_RMSE: 0.9904
  Epoch 26/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.8927 - RMSE: 0.9902 - val_loss: 1.8602 - val_RMSE: 0.9830
  Epoch 27/65
  157/157 [==============================] - 3s 19ms/step - loss: 1.8318 - RMSE: 0.9825 - val_loss: 1.8021 - val_RMSE: 0.9765
  Epoch 28/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.7751 - RMSE: 0.9764 - val_loss: 1.7479 - val_RMSE: 0.9710
  Epoch 29/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.7219 - RMSE: 0.9699 - val_loss: 1.6972 - val_RMSE: 0.9659
  Epoch 30/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.6724 - RMSE: 0.9646 - val_loss: 1.6503 - val_RMSE: 0.9621
  Epoch 31/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.6262 - RMSE: 0.9599 - val_loss: 1.6052 - val_RMSE: 0.9574
  Epoch 32/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.5827 - RMSE: 0.9549 - val_loss: 1.5641 - val_RMSE: 0.9544
  Epoch 33/65
  157/157 [==============================] - 3s 21ms/step - loss: 1.5424 - RMSE: 0.9521 - val_loss: 1.5250 - val_RMSE: 0.9510
  Epoch 34/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.5043 - RMSE: 0.9483 - val_loss: 1.4906 - val_RMSE: 0.9501
  Epoch 35/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.4688 - RMSE: 0.9454 - val_loss: 1.4580 - val_RMSE: 0.9494
  Epoch 36/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.4357 - RMSE: 0.9432 - val_loss: 1.4231 - val_RMSE: 0.9439
  Epoch 37/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.4044 - RMSE: 0.9404 - val_loss: 1.3934 - val_RMSE: 0.9421
  Epoch 38/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.3752 - RMSE: 0.9384 - val_loss: 1.3656 - val_RMSE: 0.9406
  Epoch 39/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.3480 - RMSE: 0.9365 - val_loss: 1.3399 - val_RMSE: 0.9395
  Epoch 40/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.3225 - RMSE: 0.9347 - val_loss: 1.3151 - val_RMSE: 0.9379
  Epoch 41/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2983 - RMSE: 0.9333 - val_loss: 1.2929 - val_RMSE: 0.9375
  Epoch 42/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2762 - RMSE: 0.9322 - val_loss: 1.2744 - val_RMSE: 0.9396
  Epoch 43/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2549 - RMSE: 0.9309 - val_loss: 1.2505 - val_RMSE: 0.9348
  Epoch 44/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2351 - RMSE: 0.9302 - val_loss: 1.2317 - val_RMSE: 0.9341
  Epoch 45/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.2167 - RMSE: 0.9289 - val_loss: 1.2161 - val_RMSE: 0.9357
  Epoch 46/65
  157/157 [==============================] - 3s 19ms/step - loss: 1.1994 - RMSE: 0.9280 - val_loss: 1.1972 - val_RMSE: 0.9327
  Epoch 47/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1831 - RMSE: 0.9273 - val_loss: 1.1827 - val_RMSE: 0.9334
  Epoch 48/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1678 - RMSE: 0.9263 - val_loss: 1.1678 - val_RMSE: 0.9324
  Epoch 49/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1536 - RMSE: 0.9266 - val_loss: 1.1542 - val_RMSE: 0.9322
  Epoch 50/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1401 - RMSE: 0.9256 - val_loss: 1.1415 - val_RMSE: 0.9319
  Epoch 51/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1275 - RMSE: 0.9253 - val_loss: 1.1285 - val_RMSE: 0.9307
  Epoch 52/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1160 - RMSE: 0.9245 - val_loss: 1.1168 - val_RMSE: 0.9300
  Epoch 53/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.1045 - RMSE: 0.9245 - val_loss: 1.1092 - val_RMSE: 0.9325
  Epoch 54/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0943 - RMSE: 0.9241 - val_loss: 1.0960 - val_RMSE: 0.9293
  Epoch 55/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0848 - RMSE: 0.9239 - val_loss: 1.0873 - val_RMSE: 0.9300
  Epoch 56/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0755 - RMSE: 0.9236 - val_loss: 1.0808 - val_RMSE: 0.9324
  Epoch 57/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0671 - RMSE: 0.9242 - val_loss: 1.0695 - val_RMSE: 0.9289
  Epoch 58/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0586 - RMSE: 0.9230 - val_loss: 1.0626 - val_RMSE: 0.9298
  Epoch 59/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0513 - RMSE: 0.9231 - val_loss: 1.0562 - val_RMSE: 0.9304
  Epoch 60/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0438 - RMSE: 0.9227 - val_loss: 1.0572 - val_RMSE: 0.9381
  Epoch 61/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0373 - RMSE: 0.9232 - val_loss: 1.0428 - val_RMSE: 0.9299
  Epoch 62/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0310 - RMSE: 0.9226 - val_loss: 1.0356 - val_RMSE: 0.9287
  Epoch 63/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0253 - RMSE: 0.9229 - val_loss: 1.0292 - val_RMSE: 0.9281
  Epoch 64/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0200 - RMSE: 0.9229 - val_loss: 1.0243 - val_RMSE: 0.9285
  Epoch 65/65
  157/157 [==============================] - 3s 20ms/step - loss: 1.0147 - RMSE: 0.9230 - val_loss: 1.0276 - val_RMSE: 0.9367
  ```

* μ΅μ  λͺ¨λΈ κ·Έλν

  ```python
  #plot RMSE
  import matplotlib.pyplot as plt 
  plt.plot(result.history['RMSE'], label = 'Train RMSE')
  plt.plot(result.history['val_RMSE'], label = 'Test RMSE')
  plt.legend()
  plt.show()
  ```

  ![](./image/6_4-1.png)
