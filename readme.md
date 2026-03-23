accelerate launch --multi_gpu --num_processes 2 \
--mixed_precision fp16 \
--dynamo_backend no \
scripts/run_training.py \
--config configs/scibert_kaggle.yaml \
--dataset-path /kaggle/input/datasets/dhiranjitdaimary/arxiv-tokenized-dataset/upload/tok_scibert_train_val_200k \
--output-dir /kaggle/working/saved_models/run_300k
