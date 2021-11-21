TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
CUDA_VISIBLE_DEVICES=1 python evaluate_mnist_post.py \
  --pt-data ori_neigh \
  --pt-method dir_adv \
  --adv-dir pos \
  --neigh-method untargeted \
  --pt-iter 50 \
  --pt-lr 0.001 \
  --att-iter 20 \
  --att-restart 1 \
  --log-file logs/log_exp02_${TIMESTAMP}.txt