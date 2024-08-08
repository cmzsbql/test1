# Usage
## Install the environment using yaml file
~~~
conda env create -f environment.yaml
~~~
## Train Stage 1
~~~
python train_vq.py --batch-size 128 --width 512 --lr 1e-4 --total-iter 100000 --lr-scheduler 200000 --code-dim 512 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir ./output/output_energy --dataname energy --vq-act relu --quantizer ema_reset2 --exp-name VQVAE --window-size 24 --commit 0.001 --gpu 0 
~~~

## Train Stage 2

### SDformer-ar
~~~
export CUDA_VISIBLE_DEVICES="0"
deepspeed train_arlm.py --exp-name ARTM --batch-size 4096 --num-layers 1 --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512 --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset2 --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
~~~

### SDformer-m
~~~
export CUDA_VISIBLE_DEVICES="0"
deepspeed train_mlm.py --exp-name MTM --batch-size 4096 --num-layers 1  --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512  --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset2 --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
~~~

## Test
### SDformer-ar
~~~
export CUDA_VISIBLE_DEVICES="0"
deepspeed train_arlm.py --exp-name test --if-test --resume-trans ./output/output_energy/ARTM/net_best_ds.pth --batch-size 4096 --num-layers 1 --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512 --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset2 --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
~~~

### SDformer-m
~~~
export CUDA_VISIBLE_DEVICES="0"
deepspeed train_mlm.py --exp-name ARTM_test  --if-test --resume-trans ./output/output_energy/MTM/net_best_ds.pth --batch-size 4096 --num-layers 1  --embed-dim-gpt 1024 --width 512 --code-dim 512 --nb-code 512  --n-head-gpt 16 --block-size 6 --ff-rate 4 --drop-out-rate 0.1 --resume-pth ./output/output_energy/VQVAE/net_best_ds.pth --vq-name VQVAE --out-dir ./output/output_energy/ --total-iter 62500 --lr-scheduler 150000 --lr 0.0008 --dataname energy --down-t 2 --depth 3 --quantizer ema_reset2 --eval-iter 2500 --print-iter 500 --pkeep 0.9 --dilation-growth-rate 3 --vq-act relu --window-size 24
~~~

## eval

##### Replace the root_dir in eval.py with the location where the generated result is saved, then run eval.py.

~~~
python eval.py
~~~

