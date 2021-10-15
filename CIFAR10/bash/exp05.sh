TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=5 python eval_fgsm.py \
  --pt-data ori_neigh \
  --pt-method adv \
  --adv-dir na \
  --neigh-method targeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 20 \
  --att-restart 1 \
  --log-file logs/log_exp05_${TIMESTAMP}.txt