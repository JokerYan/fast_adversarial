TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=7 python evaluate_mnist_post.py \
  --pt-data ori_neigh \
  --pt-method normal \
  --adv-dir na \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 40 \
  --att-restart 1 \
  --log-file logs/log_exp08_${TIMESTAMP}.txt