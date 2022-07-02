## 1. Neural network based Collaborative Filtering(NCF)

### 개요(Abstract)

- [Neural Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf) (2017, Xiangnan He)
- 딥러닝이 다른 곳에서 엄청난 성능을 거둔것에 비해, 추천시스템에서는 연구가 덜 되었다고 소개.
- 여전히 MF에 의존했었는데, linear 방식으로 인해 깊게 학습을 못한다는 점(한계) 지적
- NCF를 사용하면 이를 더 일반화할 수 있게 만들 수 있고, 깊은 신경망일 수록 더 좋은 성능을 가진다는 것을 실험적으로 증명했다고 함.
- 한국어 문서 참고 : [https://leehyejin91.github.io/post-ncf/](https://leehyejin91.github.io/post-ncf/)

<br>

### 소개(Introduction)

- 추천시스템은 정보 폭증 시대에서 정보 과부화를 완화하는데 중추적인 역할을 함
- MF를 향상시키기 위한 연구는 많이 진행되었지만, DNN(deep neural network)를 적용하기 위한 연구는 부족하다고 똑같이 말함…
- 이 논문에서는 DNN 적용과 더불어, **암시적 피드백**(implicit feedback)에 중점을 둬서 진행한다고 함
  - 암시적 피드백은 항목클릭, 제품구매, 영상 시청 등 자동으로 축적되며 더 쉽게 수집될 수 있다는 장점이 존재.
  - 하지만 사용자의 만족도가 관찰되지 않고, 부정적 피드백을 알 수 없기 때문에 활용하기는 더 어렵다.
- NCF의 주요 목표(기여)를 3가지로 소개.
  1. DNN에 기반한 Collaborative Filtering 방법 제기(NCF)
  2. multi-layer 퍼셉트론을 통한 높은 수준의 non-linear 학습이 가능하다는 것을 증명
  3. 다양한 실험을 통해 NCF의 효율성 증명

<br>
<br>

## 2. Preliminaries(사전 학습)

### Learning from Implicit Data

$M$(users)과 $N$(items)과 있고 user-item matrix $Y$가 있다.

$$
Y_{m,n} =
 \begin{pmatrix}
  y_{1,1} & y_{1,2} & \cdots & y_{1,n} \\
  y_{2,1} & y_{2,2} & \cdots & y_{2,n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  y_{m,1} & y_{m,2} & \cdots & y_{m,n}
 \end{pmatrix}
$$

$$
\begin{align*}
where, \ \ \  y_{u,i} = \begin{cases}1, \ \text{if interaction(user $u$, item $i$) is observed} \cr 0, \ \text{otherwise} \cr\end{cases}
\end{align*}
\\
$$

- 여기서 $y_{u,i}=1$는 상호작용이 있었다는 것을 의미하며 예를 들어 구매했거나 열람했다는 것을 의미한다.
- 다만 상호작용이 있었다고 해서 그것이 선호를 뜻하진 않는다. (그 반대도 마찬가지)
- 논문에서는 이런걸 noisy signals라고 표현하며 도전적인 과제라고 말한다.
- 결국에는 상호작용이 있었다는 건 최소한 관심은 있었다는 것을 뜻하며, 다만 없었다는 건 이게 진짜 관심이 없었던 건지 데이터가 노출조차 안되서 사용자가 몰랐던 건지는 알 수 없다.

<br>

결국 우리의 목표는 관찰되지 않은 항목의 점수(순위)를 추정하는 것이다. 그리고 이는 다음과 같이 표현할 수 있다. (예측값은 0~1 사이 값)

$$
\hat{y}_{u,i} = f(u,i|\Theta),\ \\  where \ f \ \  \text{is interaction function}, \Theta \  \text{is model parameters} \\
$$

<br>

여기서 보통 2가지 종류의 loss function을 사용할 수 있는데 다음과 같다.

$$
\begin{align*}
L_{1} &= min \ \dfrac{1}{2}(\hat{y}_{u,i} - y_{u,i})^{2} \\
L_{2} &= max(0, f(y_{unobs})-f(y_{obs})+\alpha)  \ \ s.t  \ \  rank(y_{obs}) >rank(y_{unobs})
\end{align*}
$$

- L1은 point-wise loss라고 불리며 한번에 하나의 아이템만 고려해서 각각 (user, item)별로 계산해서 업데이트하는 굉장히 직관적이 일반적인 방법이다. 주로 분류, 회귀 모델에서 쓰인다.
- L2는 pair-wise loss라고 불리며 한번에 2개의 아이템을 고려해서 (pos, neg item pair)과 같이 미리 rank를 고려해서 학습을 시키는 방법이다.
- NCF는 두가지 모두 사용 가능하다고 설명하고 있다.

<br>

### Matrix Factorization

MF는 $Y$를 보다 저차원 행렬 2개 ($P$, $Q$)로 분해해서 표현하는 방법이다.

$$
\begin{array}{c}
    Y(user-item)\\
    \left[\begin{array}{ccc}
        y_{1,1} & \cdots & y_{1,n}\\
        \vdots  & \ddots & \vdots\\
        y_{m,1} & \cdots & y_{m,n}\\
    \end{array}\right]\\
    m\times n
\end{array}
=

\begin{array}{c}
    P(user)\\
    \begin{bmatrix}
        p_{11} & \cdots & p_{1k}\\
        \vdots  & \ddots & \vdots\\
        p_{m1} & \cdots & p_{mk}\\
    \end{bmatrix}\\
    m\times k
\end{array}

\begin{array}{c}
    Q(item)\\
    \begin{bmatrix}
        q_{11} & \cdots & q_{1n}\\
        \vdots  & \ddots & \vdots\\
        q_{k1} & \cdots & q_{kn}\\
    \end{bmatrix}\\
    k\times n
\end{array}
$$

<br>

이 때 MF는 $y_{u, i}$를 아래와 같이 내적으로 추정하게 된다.

$$

\hat{y}_{u,i}= f(u,i|\mathbf{p}_{u}, \mathbf{q}_{i}) = \mathbf{p}_{u} \mathbf{q}_{i}^{T}=  \sum_{k=1}^K p_{uk} q_{ki}


$$

- 잠재공간의 각 차원이 서로 독립적이라고 가정하며 동일한 가중치로 선형결합하는 양방향 상호작용을 모델링한다고 볼 수 있다.
- 다만 내적과 같은 linear 모델은 user-item의 복잡한 관계 표현에 한계가 있다는 것을 지적한다.

![이미지](/assets/images/MF_limitation.png)

- 기존에 u1, u2, u3만 있었을 때는 u2와 u3 연관성이 더 높고 비교적 u1은 u2, u3보다 연관성이 없다.
- linear space의 한계는 새로운 user 4가 등장했을 때 발생한다. u4는 u1과 가장 연관이 높고, u3, u2 순으로 연관성이 있다.
- 하지만 오른쪽 그림에서 보여주듯이 user 1, 2, 3이 만든 공간에서 user 4를 표현할 때 절대 표현할 수 없는 한계가 있다. 즉, 4를 1과 2사이에 배치하면 1과 2가 가까워지게 되는 불일치가 발생한다…

<aside>
💡 MF는 복잡한 관계를 저차원의 단순한 공간에서 표현할 수 없다는 것을 보여준다. 새로운 관계로 표현하는데 유연하지 못함…
</aside>

<br>
<br>

## 3. NCF

- MF가 NCF에서 표현되고 일반화 될 수 있음을 보여준다.
- MLP(multi-layer perceptron)를 사용하여 상호작용 함수를 학습하는 것을 제안한다.
- 마지막으로 new neural matrix factorizations model 제시 (MF 와 MLP 앙상블을 통해 선형성과 비선형성 강점을 통합)

### General Framework

![이미지](/assets/images/NCF.png)

- Input layer (sparse) : 단순 원핫 벡터이다.

<br>

- Embedding layer (fully connected layer)
  - input 단계의 sparse 벡터를 dense 벡터로 맵핑하는 단계를 의미한다.
    ![이미지](/assets/images/embeddinglayer.png)
  - 그니까 원핫 벡터를 가중치 행렬 $P$와 곱해 m 차원의 sparse 벡터를 $k(<m)$차원 공간의 projection(투영)해서 dense 벡터로 변환시킨다고 보면 된다.

<br>

- Neural CF layers: user latent vector 와 item latent vector를 concatenation한 벡터를 input으로 받아 deep neural network 통과하는 단계이다.

  - user latent vector = $P^{T}v_{u}^{U}$ (이때, $v_{u}^{U}$는 user $u$를 나타내는 one-hot 벡터)
  - item latent vector = $Q^{T}v_{i}^{I}$ (이때, $v_{i}^{I}$는 item $i$를 나타내는 one-hot 벡터)
  - deep neural network : $\phi_{X}(…\phi_{2}(\phi_{1}(P^{T}v_{u}^{U}, Q^{T}v_{i}^{I}))…)$

<br>

- Output layers : 0~1 사이의 예측값 $\hat{y}_{u,i}$를 구하는 단계이다. 이때 $ϕ_{out}$ 에는 logistic 함수나 probit 함수를 사용한다.
