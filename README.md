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

## 코드 실행 our method
1. 이미지 불러오기(오토인코더)

```bash
import os
import cv2
import re
import matplotlib.pyplot as plt

# 이미지 파일이 있는 디렉토리 경로
image_directory = "파일 경로"

# 디렉토리 내의 모든 이미지 파일 가져오기
image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]

# 파일 이름에서 숫자 부분을 추출하는 함수
def extract_number_from_filename(filename):
    # 정규식을 사용하여 "predicted_image_" 부분을 제거하고 숫자 부분만 추출
    number = re.search(r'\d+', filename)
    if number:
        return int(number.group())
    else:
        return -1  # 숫자가 없는 경우 -1 반환 또는 다른 예외 처리 방법 선택

# 파일 이름을 숫자 기준으로 정렬
image_files.sort(key=lambda x: extract_number_from_filename(x))

# 이미지를 저장할 리스트 변수 생성
image_list = []

# 이미지 파일을 하나씩 로드하고 리스트 변수에 추가
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    # 이미지 로드 (BGR 형식)
    reconstructed_image = cv2.imread(image_path)

    # 이미지가 유효한 경우 BGR에서 RGB로 변환하여 리스트 변수에 추가
    if reconstructed_image is not None:
        reconstructed_image_rgb = cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB)
        image_list.append(reconstructed_image_rgb)
    else:
        print(f"{image_file} 이미지를 로드할 수 없습니다.")

# image_list에는 순서대로 이미지가 저장됩니다
```

2. 함수 Import

```bash
# If running in Google Colab, import files
try:
    import google.colab
    in_colab = True
except:
    in_colab = False

if in_colab:
    !git clone https://github.com/Hyperparticle/one-pixel-attack-keras.git
    !mv -v one-pixel-attack-keras/* .
    !rm -rf one-pixel-attack-keras

# Python Libraries
%matplotlib inline
import pickle
import numpy as np
import pandas as pd
import matplotlib
from keras.datasets import cifar10
from keras import backend as K

# Custom Networks
from networks.lenet import LeNet
from networks.pure_cnn import PureCnn
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet
from networks.densenet import DenseNet
from networks.wide_resnet import WideResNet
from networks.capsnet import CapsNet

# Helper functions
from differential_evolution import differential_evolution
import helper

matplotlib.style.use('ggplot')
np.random.seed(100)
```

3. 데이터 로드(Cifar-10 불러오기 및, Our image 불러오기)

```bash
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

image_list = np.array(image_list)
print(image_list.ndim)
print(len(image_list))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

4. 모델 로드 및 성능 평가

```bash
# 모델 선언
lenet = LeNet()
pure_cnn = PureCnn()
net_in_net = NetworkInNetwork()
resnet = ResNet()
densenet = DenseNet()
wide_resnet = WideResNet()

models = [lenet, pure_cnn, net_in_net, resnet, densenet, wide_resnet]
```

```bash
network_stats, correct_imgs = helper.evaluate_models(models, x_test, y_test)
correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

network_stats
```

5. 기타 커스텀 함수들 for predction

```bash
def perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])

    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
    xs = xs.astype(int)

    for x,img in zip(xs, imgs):
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb
    
    return imgs

def predict_classes(xs, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image(xs, img)
    predictions = model.predict(imgs_perturbed)[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img)

    confidence = model.predict(attack_image)[0]
    predicted_class = np.argmax(confidence)
    # If the prediction is what we want (misclassification or 
    # targeted classification), return True
    if verbose:
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
        (not targeted_attack and predicted_class != target_class)):
        return True
    # NOTE: return None otherwise (not False), due to how Scipy handles its callback function

def attack(img_id, model, target=None, pixel_count=1, 
           maxiter=75, popsize=400, verbose=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else y_test[img_id, 0]
    
    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0,32), (0,32), (0,256), (0,256), (0,256)] * pixel_count
    
    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))
    
    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, x_test[img_id], target_class, 
                               model, target is None)
    
    def callback_fn(x, convergence):
        return attack_success(x, x_test[img_id], target_class, 
                              model, targeted_attack, verbose)
    
    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, x_test[img_id])[0]
    prior_probs = model.predict_one(x_test[img_id])
    predicted_probs = model.predict_one(attack_image)
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_test[img_id, 0]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    helper.plot_image(attack_image, actual_class, class_names, predicted_class)

    return [attack_image,model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result.x]



def attack_all(models, samples=500, pixels=(1,3,5), targeted=False, 
               maxiter=75, popsize=400, verbose=False):
    results = []
    for model in models:
        model_results = []
        valid_imgs = correct_imgs[correct_imgs.name == model.name].img
        img_samples = np.random.choice(valid_imgs, samples, replace=False)
        
        for pixel_count in pixels:
            for i, img_id in enumerate(img_samples):
                print('\n', model.name, '- image', img_id, '-', i+1, '/', len(img_samples))
                targets = [None] if not targeted else range(10)
                
                for target in targets:
                    if targeted:
                        print('Attacking with target', class_names[target])
                        if target == y_test[img_id, 0]:
                            continue
                    result = attack(img_id, model, target, pixel_count, 
                                    maxiter=maxiter, popsize=popsize, 
                                    verbose=verbose)
                    model_results.append(result)
                    
        results += model_results
        helper.checkpoint(results, targeted)
    return results
```

6. Untargeted attack 수행

```bash
untargeted = attack_all(models, samples=1000, targeted=False)
```

*주의할 점: 공격 설정이 픽셀 1개, 3개, 5개로 되어있어서 이미지 분류 모델 별로 1000개씩 끊어서 해야합니다....

7. 공격 결과 보기

```bash
# Load the results
untargeted, targeted = helper.load_results()
```

```bash
columns = ['attack image', 'model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs', 'perturbation']
untargeted_results = pd.DataFrame(untargeted[:1000], columns=columns)
targeted_results = pd.DataFrame(targeted, columns=columns)
```

```bash
helper.attack_stats(untargeted_results, models, network_stats)
```

8. 공격/방어 평가

```bash
# 공격 샘플 번호 받기

untargeted_results = np.array(untargeted_results)
mod_num = []
count = 0

for i in range(1000):
    mod_num.append(untargeted_results[i][3])
    print(mod_num[i])
    count = count+1
    if(mod_num[i] == 16):
        print("카운트",count)
```

```bash
# 공격 샘플을 내림차순으로 정렬했기 때문에 내림차순으로 번호 받기

mod_num.sort()
print("정렬된 리스트:", mod_num) #공격 받은 이미지의 id 인덱스 순서
```

```bash
Origin_DATA = []
for i in range(1000):
    Origin_DATA.append(x_test[mod_num[i]]) #공격 샘플에 해당하는 원본 1000개 받기
Origin_DATA = np.array(Origin_DATA)

Origin_prediction = models.predict(Origin_DATA) #원본 예측
Attacked_prediction = resnet.predict(image_list) #공격 받은거 예측

helper.plot_image(Origin_DATA[1])
helper.plot_image(image_list[1])
```

```bash
after_recon_attack_success = 0
after_recon_attack_false = 0

for i in range(len(attack_true_id)):
    for j in range(len(mod_num)): 
        if(attack_true_id[i] == mod_num[j]):       
            Origin_prediction = models.predict_one(Origin_DATA[i]) #원본 예측
            Attacked_prediction = models.predict_one(image_list[i]) #공격 받은거 예측

            
            max_value_index_attacked_recons = np.argmax(Attacked_prediction) #공격당한 이미지 분류
            max_value_index_origin = np.argmax(Origin_prediction) #원본 이미지 분류
            
            if(max_value_index_origin != max_value_index_attacked_recons): 
                after_recon_attack_success = after_recon_attack_success + 1
                
            
            
print(len(attack_true_id)) #공격 성공한 이미지들의 개수 재구성 전임
print(len(correct_id)) #공격 받았던 이미지 1000개 재구성 한 후 원본과 비교했을 때 맞다고 한 결과
print(after_recon_attack_success) #올바르게 분류한 이미지 correct_id 중에 공격 성공한것이 남아있을텐데 그것의 개수


print("초기 공격 성공률 : ", len(attack_true_id)/1000) # 공격 성공한 이미지
print("초기 공격 성공 개수 : ", len(attack_true_id)) #재구성 전 공격 성공 이미지
print("초기 공격 받았을 때 올바르게 분류한 이미지 개수 : ", 1000 - len(attack_true_id))
print("재구성한 후 올바르게 분류한 이미지 개수 : ", len(correct_id)) #약 3% 모델 평가 성능 감소
print("재구성 후 공격 성공한 이미지 개수 : ", after_recon_attack_success)
print("재구성 후 방어 성공한 이미지 개수 : ", len(attack_true_id) - after_recon_attack_success)
print("재구성 후 방어율 : ", (len(attack_true_id) - after_recon_attack_success)/len(attack_true_id)) #모델 평가 성능 3% 감소로 46.8% 방어율을 보여줌
```


이 후 공격 샘플들 저장 (코랩 환경으로 옮겨서 작업하기 위해)
```bash
import pickle
import os
from PIL import Image
import numpy as np

# 이미지를 저장할 디렉터리 생성
if not os.path.exists('공격 분류 모델 이름'):
    os.makedirs('공격 분류 모델 이름')

# 파일 읽기
with open('networks/results/' + 'untargeted' + '_results.pkl', 'rb') as f:
    data = pickle.load(f)

# 재구성된 이미지 저장
n = 1000  # 모든 테스트 이미지 저장
for i in range(n):
    # 이미지 크기를 32x32로 조정
    resized_image = np.array(Image.fromarray(data[i][0]).resize((32, 32)))

    # 이미지를 그대로 저장
    img = Image.fromarray(resized_image)
    img.save(f'wide_resnet/{mod_num[i]}.png')

print(f"{n} 재구성된 이미지를 32x32 크기로 저장했습니다.")
```


OPA2D-DEF 코드 
```bash
import tensorflow as tf
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

    
bounds = [(0, 256), (0, 256), (0, 256)]
#original_pred = wide_resnet.predict(image_list)
for i in range(len(image_list)):
    pixels = image_list[i].copy()
    #ori_conf = wide_resnet.predict_one(pixels).max()
    original_class = np.argmax(wide_resnet.predict_one(image_list[i]))
    res = []

    for x in range(32):
        for y in range(32):

            if(pixels[x][y][0] > 128):
                pixels[x][y][0] = 0
            else:
                pixels[x][y][0] = 255

            if(pixels[x][y][1] > 128):
                pixels[x][y][1] = 0
            else:
                pixels[x][y][1] = 255

            if(pixels[x][y][2] > 128):
                pixels[x][y][2] = 0
            else:
                pixels[x][y][2] = 255



            modified = pixels.copy()
            
            conf = wide_resnet.predict(modified).max()
            ori_conf = wide_resnet.predict(image_list[i]).max()
            diff = abs(conf - ori_conf)

            res.append((x,y,diff))


    sorted_res = sorted(res, key=lambda x: x[2], reverse=True)    
    
    
   
    #ori_class = resnet.predict_one(image_list[i])
    #ori_class = np.argmax(ori_class)

    image = image_list[i].copy()

    for j in range(30):
        pixel_position = (sorted_res[j][0], sorted_res[j][1])

        result = differential_evolution(perturbation, bounds, args=(image, original_class, pixel_position),
                                    popsize=50, maxiter=5)

        optimal_pixel_values = result.x.astype(int)

        # 이미지에 공격적인 픽셀을 적용하여 시각화합니다.
        perturbed_image = image.copy()
        perturbed_image[pixel_position[0], pixel_position[1]] = optimal_pixel_values

        change_class = np.argmax(wide_resnet.predict(perturbed_image)) #DE를 통해 공격 당한 이미지 class 구하기, original은 반복문 전에 predict로 한번에 했음
        #original_class = np.argmax(original_pred[i])
        print("----------------- ", i, "번째 이미지", j,"번째 픽셀 하는 중")

        if(original_class != change_class):
            print("변경 전 class ", class_names[original_class])
            print("변경 후 class ", class_names[change_class])
            print(i,"번째 confidence 좌표에서 방어 성공")
            defense_success_count = defense_success_count + 1
            break



        
        
print(defense_success_count)
```



jupyter notebook이 실행되고, 창이 뜨면, 오픈소스AI 제출용 코드에 들어가서 markdown 되어 있는대로 실행하면 됩니다.
Google colab에서도 Markdown 되어 있는대로 실행하면 됩니다.
순서대로 코드를 정리해두어서 차례대로 실행하면 됩니다.
-> Windows 아나콘다 가상환경에서 진행


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
작업 환경 세팅은 requirements_opensouceAI.txt으로 만들어 두었습니다.


대표적인 것 몇 가지
- tensorflow==2.7.0
- python==3.7.0
- jupyter notebook




