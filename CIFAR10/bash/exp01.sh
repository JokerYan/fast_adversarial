TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
script logs/log_exp01_${TIMESTAMP}.txt
CUDA_VISIBLE_DEVICES=1 python eval_fgsm.py \
  --pt-data ori_neigh \
  --pt-method adv \
  --adv-dir na \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 20 \
  --att-restart 1
exit