TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=1 python eval_fgsm.py \
  --pt-data ori_neigh \
  --pt-method dir_adv \
  --adv-dir both \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 20 \
  --att-restart 1 \
  --log-file logs/log_exp04_${TIMESTAMP}.txt