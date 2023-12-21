# 오픈소스AI 팀플 




## 실행에 사용한 코드의 출처
[One-Pixel attack 코드](https://github.com/Hyperparticle/one-pixel-attack-keras.git) 에 있습니다.
해당 github에서 제공하는 이미지 분류 모델과 코드를 사용하였습니다.

이미지 분류 모델의 경우 아래 링크에서 다운로드 후 /networks에 models 폴더를 만들고 넣어주어야 합니다. (용량이 커서 업로드 되지 않음)
[사전 학습 이미지 분류 모델](https://www.dropbox.com/sh/dvatkpjl0sn79kn/AAC9L4puJ_sdFUkDZfr5SFkLa?dl=0)


## 실행 방법

아나콘다 가상환경, jupyter notebook 사용
코드가 옛날꺼라 약간의 수정을 하여 수정본을 깃허브에 업로드하였습니다.

```bash
cd one-pixel-attack-keras
jupyter notebook 1_one-pixel-attack-cifar10.ipynb
```

오토인코더는 코랩 환경에서 진행하였습니다.
[프로젝트 AE](https://colab.research.google.com/drive/107N6lA76RsqeA-cb5Lgp2syy_BxfSJRn?usp=sharing)


## 전체적인 실험 환경 요약

1. 아나콘다 가상환경, jupyter notebook 실행
2. 공격 샘플 생성
3. 공격 샘플 저장 후 오토인코더로 재구성
4. 재구성한 오토인코더 저장
5. 재구성 이미지 평가

   
## 작업 환경
작업 환경 세팅은 requirement.txt으로 만들어 두었습니다.



