python3 ckpt_interpolation.py  \
--data_dir "/data3/juliew/datasets/anime_faces" \
--image_size 32 32 \
--batch_size 12 \
--num_workers 12 \
--max_steps 50000 \
--devices 1 \
--sample_every_n_steps 100 \
--pretrained_model "/data3/juliew/projet2_diffusion/consistency_models/checkpoints/anime_faces" \
--sampling_sigmas 80.0 24.4 5.84 0.9 0.661
#--train_continue

