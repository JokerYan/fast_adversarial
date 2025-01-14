TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=7 python evaluate_mnist_post.py \
  --pt-data ori_neigh \
  --pt-method dir_adv \
  --adv-dir both \
  --neigh-method untargeted \
  --pt-iter 150 \
  --pt-lr 0.001 \
  --att-iter 40 \
  --att-restart 1 \
  --log-file logs/log_exp17_${TIMESTAMP}.txt