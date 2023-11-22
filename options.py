import argparse


def parse_opts():
    parser = argparse.ArgumentParser(
        description="Run the training model with given configurations"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data3/juliew/datasets/butterflies/",
        help="Directory of the data",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=(32, 32),
        help="Size of the images (width height)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--max_steps", type=int, default=4000, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=100,
        help="Frequency of sampling during training steps",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to train on (e.g., 1 for single GPU)",
    )
    parser.add_argument(
        "--train_continue",
        action="store_true",
        help="Continue training from the last checkpoint",
    )

    return parser.parse_args()
