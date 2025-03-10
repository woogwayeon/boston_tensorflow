# Boston 집값 데이터

---

[boston.csv](attachment:c90caa72-f842-4e64-87fc-d42c30227e84:boston.csv)

![image.png](attachment:d813a26f-7759-4da2-93a8-88084d8f1bbc:image.png)

여러가지 정보가 담겨있는 보스턴 집값 데이터입니다

![image.png](attachment:e1b72a1a-05c5-40b6-a287-bbb4c606be0b:image.png)

원인과 결과가 있다면 y값은 지금 medv(중앙값)!

13개의 원인이 있다고 보면 된다

그러니까 중앙값에 가중치를 곱해서 공식을 만들면 되는데, 이걸 tensorflow 통해 진행한다

### 진행순서

---

1. 과거의 데이터 준비 (csv 파일 등)
2. 모델의 구조 만들기
3. 데이터로 모델 학습
4. 모델 이용하기

[boston.ipynb](attachment:5c72b610-4469-4f61-8d07-72881497971d:boston.ipynb)

![image.png](attachment:c9034d44-c2f7-4798-8bb7-ca4cd7efe235:image.png)

```python
import tensorflow as tf
import pandas as pd

# 데이터 준비
location = 'boston.csv'
boston = pd.read_csv(location)

print(boston.columns)
boston.head()

# 독립변수(iv) / 종속변수(dv) 준비
iv = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 
							'tax', 'ptratio', 'b', 'lstat']]
dv = boston[['medv']]
print(iv.shape, dv.shape)

# 모델의 구조 만들기
X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)

model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')

# 데이터로 모델을 학습(filter?)합니다
model.fit(iv,dv, epochs=10000, verbose=0)
```

구글  Colab을 이용해서 boston 집값 데이터를 갖고 모델 제작!

집값에 영향을 미치는 요소들을 독립변수(`X`)에 두고 중앙값을 종속변수(`Y`)로 설정한다

![image.png](attachment:ca618506-0ce7-4b3b-b016-a76735ca6421:image.png)

학습시킨 다음 모델을 이용해본다!
