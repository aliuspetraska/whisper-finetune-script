import logging

import torch

from services.model_trainer import ModelTrainerService

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s][%(name)s] %(message)s', force=True)


def main():
    model_trainer = ModelTrainerService()

    num_gpus = torch.cuda.device_count()
    logging.info('num_gpus: ', num_gpus)

    try:
        model_trainer.login()
        model_trainer.load()
        model_trainer.get_data_source()
        model_trainer.train()
    except KeyboardInterrupt:
        print("program interrupted by user")
    except Exception as e:
        print(f"an unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
