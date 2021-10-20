TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=6 python eval_fgsm.py \
  --pt-data ori_neigh \
  --pt-method dir_adv \
  --adv-dir both \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 50 \
  --att-restart 10 \
  --log-file logs/log_exp16_${TIMESTAMP}.txt