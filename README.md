# Implement LARS

### Task
1. LARS를 사용하지 않았을 때의 결과 확인
    * 논문에 나온 비교군들을 확인할 필요 있음
    * batch 16K is out of memory
    
2. LARS 구현 
    * LARS가 제대로 동작하고 있다는 것을 확인하는 Test 진행
    * 만약 여유가 있다면, cuda를 사용해서 학습 속도를 가속시켜보는 것으로 한다.

3. Tensor board 등을 사용해서 각 Layer 별 학습 양상 확인 (optional)