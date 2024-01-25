#터미널에 해당 명령어 실행 => pip install -U -r yolov5/requirements.txt

import torch

# 이미지 경로 list로 넣기
train_img_list = glob('train/images/*.jpg')# + glob('./train/images/*.jpeg')
valid_img_list = glob('valid/images/*.jpg')# + glob('./valid/images/*.jpeg')
test_img_list = glob('test/images/*.jpg')# + glob('./valid/images/*.jpeg')

# txt 파일에 write
with open('train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('valid.txt', 'w') as f:
    f.write('\n'.join(valid_img_list) + '\n')

with open('test.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')
    

#터미널에 해당 명령어 실행 => yolov5/train.py --img 320 --batch 16 --epochs 2 --data '강아지이름'/data.yaml --weights yolov5x.pt --name result_E --cfg yolov5/models/yolov5x.yaml