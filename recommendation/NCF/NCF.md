## Neural network based Collaborative Filtering(NCF)

### 개요(Abstract)

- [Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf) (2017, Xiangnan He)
- 딥러닝이 다른 곳에서 엄청난 성능을 거둔것에 비해, 추천시스템에서는 연구가 덜 되었다고 소개.
- 여전히 MF에 의존했었는데, linear 방식으로 인해 깊게 학습을 못한다는 점(한계) 지적
- NCF를 사용하면 이를 더 일반화할 수 있게 만들 수 있고, 깊은 신경망일 수록 더 좋은 성능을 가진다는 것을 실험적으로 증명했다고 함.

### 소개(Introduction)

- 추천시스템은 정보 폭증 시대에서 정보 과부화를 완화하는데 중추적인 역할을 함
- MF를 향상시키기 위한 연구는 많이 진행되었지만, DNN(deep neural network)를 적용하기 위한 연구는 부족하다고 똑같이 말함…
- 이 논문에서는 DNN 적용과 더불어, 암시적 피드백(implicit feedback)에 중점을 둬서 진행한다고 함
  - 암시적 피드백은 항목클릭, 제품구매, 영상 시청 등 자동으로 축적되며 더 쉽게 수집될 수 있다는 장점이 존재.
  - 하지만 사용자의 만족도가 관찰되지 않고, 부정적 피드백을 알 수 없기 때문에 활용하기는 더 어렵다.
- NCF의 주요 목표(기여)를 3가지로 소개.
  1. DNN에 기반한 Collaborative Filtering 방법 제기(NCF)
  2. multi-layer 퍼셉트론을 통한 높은 수준의 non-linear 학습이 가능하다는 것을 증명
  3. 다양한 실험을 통해 NCF의 효율성 증명

## Preliminaries(예선, 사전학습)

### Learning from Implicit Data
