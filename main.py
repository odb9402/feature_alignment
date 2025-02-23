import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from src.experiment import GenericExperimentFactory
from src.utils import set_all_seeds

@hydra.main(config_path='config', config_name='config')
def run_experiment(cfg: DictConfig):
    set_all_seeds(cfg.seed)

    # Add a log file for detailed logging with rotation and retention policies
    logger.add("experiment.log", rotation="1 MB", retention="10 days", level="DEBUG")
    
    # Save the configuration as a YAML file for reproducibility
    with open("config_dump.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Log the configuration in the console with colors for better readability
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Initialize and set up the experiment
    logger.info("Experiment starting...")
    factory = GenericExperimentFactory()
    model = factory.create_model(cfg)
    dataloader = factory.create_dataloader(cfg)
    trainer = factory.create_trainer(model, dataloader, cfg)

    # Start training and log progress
    logger.info("Training started...")
    loss_history = trainer.train()
    logger.info("Training completed!")

    return loss_history

if __name__ == "__main__":
    run_experiment()