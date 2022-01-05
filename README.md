# fashion_mnist
---

### 1. Subject
 fasion_mnist데이터를 사용해서 cnn모델을 만져보고자 함.
 
---

### 2. Data
torchvision.datasets.FashionMNIST에서 다운받아서 사용 <br>
transforms를 통해서 tensor로 변환해서 사용

---

### 3. File
파일 별 의미 대략 정리

 + **main.py** : 모델을 돌릴 때 실행하는 파일, device를 설정해줌
 + **data_pro.py** : fasion_mnist데이터를 다운받고 dataloader설정해주는 파일
 + **make_model.py** : 모델을 만들어주는 파일, torchvision에서 제공해주는 resnet과 직접 만든 CNN모델 존재
 + **fit_mode.py** : 만든 모델과 데이터를 이용해서 학습을 돌리는 파일

---

### 4. Result
 - model : 제공해주는 resnet (직접만든 CNN모델과 비교했을 때 미세하게 성능이 더 좋음 / 다만 시간이 더 걸림)
 - train 중간마다 성능을 확인해주기 위해서 test 실행
 - epoch : 5번 (데이터를 몇번 실행한 것인가)
 - optimizer : Adam (데이터와 손실 함수를 바탕으로 모델을 업데이트하는 방식)
 - learning_rate : 0.001
 - loss function : crossentropy (훈련 과정에서 모델의 오차를 측정하는데 사용)
 
 <br>

   iteration|loss|accuracy
   :---:|:---:|:---:
   500|0.4425|89.6
   1000|0.3757|91.6
   1500|0.1191|91.1
   2000|0.2566|90.4
   2500|0.0691|91.4
   3000| 0.1364|91.9
 
 ![fig1](https://user-images.githubusercontent.com/55525705/148201144-042e64d6-2ded-4e0e-b00c-427aae051f94.png)

-> iteration별 loss
  
  
 ![fig2](https://user-images.githubusercontent.com/55525705/148201205-a4682293-686a-4783-a2c8-3850e8eecda1.png)
 
-> iteration별 accuracy
 
 
### 5. 추후에 고려할 것들
 - 제공해주는 resnet이 아닌 직접 만든 모델을 변형해서 성능 개선
 - learning_rate, optimizer등 변경 가능한 hyperparameter 조절
