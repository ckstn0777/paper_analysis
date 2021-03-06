## Matrix Factorization (MF)
### 논문 소개
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf) (2009, Yehuda Koren)
- 무려 2009년 논문이다. 아무래도 초창기이지만 추천시스템에서 가장 기초적인 부분이 아닐까 생각
- 기본적인 논문이다보니 한국어 자료도 많은데 그중에서 [https://greeksharifa.github.io/machine_learning/2019/12/20/Matrix-Factorization/](https://greeksharifa.github.io/machine_learning/2019/12/20/Matrix-Factorization/) 여기를 참고

<br>

### 추천시스템 개요
- 현대 사용자들은 컨텐츠가 많아서 선택하기 어려운 문제를 가지고 있다. ex) 넷플릭스, 왓챠 등
- 그러므로 사업체(기업)에서는 사용자의 니즈를 어떻게 해결(충족)하는가에 대해 매우 중요하게 생각하고 있다. 그래서 **추천시스템**을 계속 연구하고 시도하는 것.
- 특히 이러한 시스템은 영화, 음악, TV 쇼 등에 적용하기 쉽다. 왜냐면 사용자가 기꺼이 자신의 의견(평가)를 노출하려 하기 때문이다. 기업은 이러한 방대한 데이터를 이용할 수 있게 된다.

<br>

### 추천시스템 전략
- 추천시스템에는 크게 2가지 전략이 있다. 컨텐츠 기반 필터링(CBF)와 협업 필터링(CF)
- CBF는 사용자나 아이템에 대한 구체적인 프로파일을 먼저 만들어야 한다. 예를 들어, 영화 프로파일에는 장르, 출연배우, 인기도 등이 있을 수 있다. (결국 외부 정보 수집과 분석 필요성..)
- CF는 과거 유저 행동(이력) 기반에 의존한다. 예를 들어, 사용자가 매긴 평점, 행동이력, 구매이력 등. 결국 프로파일을 만들 필요 없고, Domain-free(=이 분야에 지식 필요없음)하다는 장점이 있다.
- 다만 CF는 cold start라는 문제를 가지고 있다. cold start란 새로운 유저나 아이템에 대해 제대로 적응하지 못하는 것이다. 이런 측면에서는 CBF가 더 낫다.

<br>

### 협업 필터링(CF)
- 협업 필터링에는 2가지 방법이 있는데, 근접 이웃 방법(neighborhood method)과 잠재 요인 방법(latent factor method)이 있다.
- 근접 이웃 방법은 유사한 사용자 혹은 유사한 아이템을 찾아서 추천하는 것이다. A라는 사용자가 라이언 일병 구하기라는 영화를 좋아하는데, 이런 사용자와 유사한 사용자를 찾아서 추천해주는 것이다.
- 잠재 요인 방법은 평점 패턴에서 20~100가지 요인을 추론하는 것에 있다.

<br>

### Matrix factorization methods
- 잠재 요인 방법을 구현하는 기본적인 방법은 바로 행렬분해(MF)이다.
- 행렬분해는 평점 행렬(패턴)을 이용해 아이템과 유저 요인을 추론해내는 것이다. 이 때 사용자와 아이템 사이의 강한 관련성이 있다면 추천이 시행된다.
- 추천 시스템은 여러 종류의 input data를 사용할 수 있다.
    - 명시적 피드백(Explicit feedback) : 개발자 입장에서 가장 좋은 피드백. 아이템에 대한 평점이나 좋아요/싫어요가 해당된다. 다만 너무 희소행렬이다. 아이템 수는 많은데 그중 평가는 매우 일부일테니..
    - 암시적 피드백(Implicit feedback) : 명시적 피드백을 사용하기 힘들경우, 혹은 명시적 피드백과 적절히 조화를 해서 사용할 수 있는 피드백이다. 사용자 검색기록, 브라우저 히스토리, 구매 이력 등에 해당된다.

<br>

### A Basic Matrix Factorization Model
- Matrix Factorization(MF) 모델은 사용자와 아이템을 차원 f의 결합 잠재요인 공간에 매핑하는데, 사용자-아이템 상호작용은 이 공간에서 **내적으로 모델링** 된다.
- 아이템 i는 $q_i$로, 사용자 u는 $p_u$라는 벡터로 표현된다. 이 둘의 내적은 **사용자-아이템 사이의 상호작용**
을 반영하며 이는 곧 아이템에 대한 사용자의 전반적인 관심을 표현한다고 볼 수 있다.

$$
\hat{r_{ui}} = q^{T}_i p_u
$$

- 이 모델은 사실 **SVD**(Singular Vector Decomposition)과 매우 유사한데, 추천 시스템에서는 결측값의 존재로 이 SVD를 직접적으로 사용하는 것은 불가능하다. 결측값을 채워 넣는 것 역시 효율적이지 못하고 데이터의 왜곡 가능성 때문에 고려하기 힘들다.
- 따라서 오직 관측된 평점만을 직접적으로 모델링하는 방법이 제시되었으며, 이 때 과적합을 방지하기 위해 규제 항이 포함되었다.
- 요인 벡터 $q_i, p_u$를 학습하기 위해 시스템은 관측된 평점 세트를 바탕으로 아래 식을 최소화하는 것을 목적으로 한다.

$$
\min_{q, p} \sum_{(u, i) \in K} ( r_{ui} - q^T_i p_u  )^2 + \lambda (\Vert{q_i}\Vert^2 + \Vert{p_u}\Vert^2)
$$

- 결과적으로 이 모델은 알려지지 않은 평점을 예측하는 것이 목적이기 때문에 과적합을 방지해야 하고, 이를 위해 규제항이 필요하고 λ가 이 규제의 정도를 제어한다