# README

## Installtion
```
$ pip install -r requirements.txt
```

## Training
```bash
$ python main.py \
    --epochs <epochs> \
    --batch_size <batch size> \
    --encoder <encoder to use for deeplabv3> \
    --img_size <img size> \
    --lr <model learning rate> \
    --model_name <model to use, Unet, Deeplabv3> \
    --warmup_epochs <epochs for warming up> \
    --img_root <training image> \
    --mask_root <training mask> \
    --test_img <testing image> \
    --test_mask <testing mask>
```

## Generate image
```bash
$ wget https://github.com/JoyPang123/segmentation/releases/download/weights/best.pt
$ python predict.py \
    --model_name <model to use, Unet, Deeplabv3> \
    --encoder <encoder to use for deeplabv3> \
    --img_size <img size> \
    --thresh <threshold for the image size> \
    --ttype <test time augmentation strategy> \
    --sum_thresh <threshold if ttype is sum> \
    --img_root <testing image> \
    --save_root <root to save image> \
    --weight <weight file>
```
