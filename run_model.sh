python3 improved_consistency.py  \
--data_dir "/data3/juliew/datasets/butterflies/" \
--image_size 32 32 \
--batch_size 32 \
--num_workers 12 \
--max_steps 100000 \
--sample_every_n_steps 5 \
--devices 1 \
--lr 1e-4 \
--lr_scheduler_start_factor 1e-5 \
--lr_scheduler_iters 10_000 \
--num_samples 32 \
--env "consistency_lightning_butterfly" \
--train_continue

