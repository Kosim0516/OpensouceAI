# 오픈소스AI 팀플 




## 실험에 사용한 코드의 출처
[One-Pixel attack 코드](https://github.com/Hyperparticle/one-pixel-attack-keras.git) 에 있습니다.
해당 github에서 제공하는 이미지 분류 모델과 코드를 사용하였습니다.

사전 학습된 이미지 분류 모델의 경우 아래 링크에서 다운로드 후 /networks에 models 폴더를 만들고 넣어주어야 합니다. (용량이 커서 업로드 되지 않음) 이 또한 위 논문의 github에서 가져왔습니다.


[사전 학습 이미지 분류 모델](https://www.dropbox.com/sh/dvatkpjl0sn79kn/AAC9L4puJ_sdFUkDZfr5SFkLa?dl=0)




## 실행 방법

아나콘다 가상환경, jupyter notebook 사용
코드가 옛날꺼라 약간의 수정을 하여 수정본을 깃허브에 업로드하였습니다.

```bash
pip install requirements_opensouceAI.txt 
```
환경 세팅 후

```bash
cd one-pixel-attack-keras
jupyter notebook
```

jupyter notebook이 실행되고, 창이 뜨면, 오픈소스AI 제출용 코드에 들어가서 markdown 되어 있는대로 실행하면 됩니다.
Google colab에서도 Markdown 되어 있는대로 실행하면 됩니다.
순서대로 코드를 정리해두어서 차례대로 실행하면 됩니다.

오토인코더는 코랩 환경에서 진행하였습니다.
[프로젝트 AE](https://colab.research.google.com/drive/107N6lA76RsqeA-cb5Lgp2syy_BxfSJRn?usp=sharing)

오토인코더 + PSD 코랩에서 진행하였습니다.
[Our Method + PSD](https://colab.research.google.com/drive/1bXt-Vi_QGN5JuYA14IFbt2VQlBzUnVfg?usp=sharing)



## 전체적인 실험 과정 요약

1. 아나콘다 가상환경, jupyter notebook 실행
2. 공격 샘플 생성
3. 공격 샘플 저장 후 오토인코더로 재구성
4. 재구성한 오토인코더 저장
5. 재구성 이미지 평가

   
## 작업 환경
작업 환경 세팅은 requirement.txt으로 만들어 두었습니다.



