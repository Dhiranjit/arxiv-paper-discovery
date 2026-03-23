accelerate launch --multi_gpu --num_processes 2 scripts/run_training.py \
--config configs/scibert_kaggle.yaml \
--dataset-path data/processed/tok_scibert_scivocab_uncased \
--output-dir outputs/run_300k
