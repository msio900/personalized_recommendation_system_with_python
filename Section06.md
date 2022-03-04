# 👍Section 06_ 딥러닝을 사용한 추천 시스템[↩](../../)

## contents📑<a id='contents'></a>

* 0_ 들어가기 전에[✏️](#0)
* 1_ Maxtrix Factorization(MF)을 신경망으로 변환하기[✏️](#0)

## 0_ 들어가기 전에[📑](#contents)<a id='0'></a>

* 딥러닝(Deep Learning : DL)은 다수의 은닉층(hidden layer)을 가진 인공신경망을 적용한 기법

## 1_ Maxtrix Factorization(MF)을 신경망으로 변환하기[📑](#contents)<a id='1'></a>

![](./image/6_1-1.png)

* `MF`를 `Keras`로
* input layer : 각 사용자와 item으로 부터 입력을 받는 부분
  * one-hot representation : 원핫인코딩과 같음. 1인지 0인지를 binary한 형태로 바꿔줌. 
* 사용자의 One-Hot Represention 입력

|        | feature 1 | feature 2 | feature 3 | ...  | feature M |
| ------ | --------- | --------- | --------- | ---- | --------- |
| User 1 | 1         | 0         | 0         | 0    | 0         |
| User 2 | 0         | 1         | 0         | 0    | 0         |
| User 3 | 0         | 0         | 1         | 0    | 0         |
| ...    | 0         | 0         | 0         | 1    | 0         |
| User M | 0         | 0         | 0         | 0    | 1         |

* embedding Layer : 잠재요인 K를 규정

  ![](./image/6_1-2.png)

  * 사용자 노드에 모두 연결됨. 즉, 화살표가 갯수가 M*K개가 연결되어 있음. 
  * 한 사용자당 K개의 화살표가 있음.
  * 만약 item이면 N* K가 연결되어 있음.

* MP : P(M * K)와 Q(N * K)가 연결됨. 

* Element-wise Product Layer

  ![](./image/6_1-3.png)

  * 사용자와 아이템의 각 프로덕트 연산을 위한 layer
  * P * Q<sup>T</sup>

* 사용자와 아이템의 평가경향(bias)

  ![](./image/6_1-4.png)

* 총 정리

  ![](./image/6_1-5.png)

  1. 사용자 아이템 두가지의 원핫 레프리젠테이션을 진행함.
  2. 사용자 입력은 K개의 노드를 갖는 유저, 아이템 임베딩과 연결
  3. 유저 임베딩과 아이템 임베딩은 DOT프로덕트연산으로 연결
  4. 다시 입력으로 돌아와 1개의 사용자가 각각 아이템 평가 경향, 유저 평가 경향 임베딩과 연결됨.
  5. 마지막으로 유저평가 경향 임베딩, 아이템 평가 경향임베딩, DOT 이 결함됨. 
  6. Flattten은 차원을 줄여주는 역할을 함. 

* 위에서 BU와 BD는 구현을 했는데. B(상수)를 구현하진 못함.

  * 전체 평균을 일률적으로 빼서 편성하게 됨. 

## 2_ Keras로 MF 구현하기[📑](#contents)<a id='2'></a>

* 파일 불러오기

  ```python
  # csv 파일에서 불러오기
  import  pandas as pd
  
  # train set과 test set을 나누기 위한 라이브러리
  from sklearn.model_selection import train_test_split
  
  # 필요한 tensorflow 모듈들을 가져온다.
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
  from tensorflow.keras.regularizers import l2
  from tensorflow.keras.optimizers import SGD, Adamax
  
  # DataFrame 형태로 데이터를 읽어온다.
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

* 데이터 설정

  ```python
  K = 200
  
  mu = ratings_train.rating.mean()
  
  M = ratings.user_id.max() + 1
  N = ratings.movie_id.max() + 1      # bias_com의 크기 1을 감안하는 것!
  
  def RMSE(y_true, y_pred):
      return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
  ```

* P, Q 사용자 평가 경향 임베딩

  ```python
  user = Input(shape=(1,))
  item = Input(shape=(1,))
  
  P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)
  Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)
  
  user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)
  item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)
  ```

* 요약 출력

  ```python
  R = layers.dot([P_embedding, Q_embedding], axes=(2,2)) # k와 k끼리 연산을 하겠다.
  
  R = layers.add([R, user_bias, item_bias])
  
  R = Flatten()(R)
  
  model = Model(inputs=[user, item], outputs=R)
  model.compile(
      loss=RMSE,
      optimizer=SGD(),
      metrics=[RMSE]
  )
  
  model.summary()
  
  # 실행 결과
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

* 모델 생성

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
  
  # 실행 결과
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

* RMSE 시각화

  ```python
  # plot RMSE
  import matplotlib.pyplot as plt
  
  plt.plot(result.history['RMSE'], label="Train RMSE")
  plt.plot(result.history['val_RMSE'], label = 'Test RMSE')
  plt.legend()
  plt.show()
  ```

  ![](./image/6_2.png)
