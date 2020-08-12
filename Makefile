run_siamese:
python run_siamese_base.py \
	--out_dir=./results/base \
	--train_batch_size=128 \
	--n_train=90000 \
	--idx_gpu=0 \
	--log_step=100

python run_siamese.py \
	--out_dir=./results/with_tricks \
	--train_batch_size=256 \
	--n_train=90000 \
	--idx_gpu=1 \
	--log_step=50

python run_siamese.py \
	--out_dir=./results/toy \
	--train_batch_size=128 \
	--eval_batch_size=40 \
	--n_train=1500 \
	--idx_gpu=0 \
	--log_step=10