TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=4 python evaluate_mnist_post.py \
  --pt-data ori_neigh \
  --pt-method adv \
  --adv-dir na \
  --neigh-method targeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 40 \
  --att-restart 1 \
  --log-file logs/log_exp05_${TIMESTAMP}.txt