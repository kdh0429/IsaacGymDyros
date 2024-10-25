from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.utils.utils import retry
from isaacgymenvs.utils.reformat import omegaconf_to_dict


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb
        from datetime import datetime
        wandb_unique_id = f"uid_{experiment_name}_{datetime.now().strftime('%m%d_%H%M')}"
        print(f"Wandb using unique id {wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=wandb_unique_id,
                name=datetime.now().strftime('%m%d_%H%M'),
                # name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='fork'),
            )
            import os 
            if cfg.wandb_project == 'TocabiAMPLower':
                print('Saving files for TocabiAMPLower.....')
                wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/cfg/task/TocabiAMPLower.yaml'), policy='now')
                wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/cfg/train/TocabiAMPLowerPPO.yaml'), policy='now')
                wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/tasks/amp/tocabi_amp_lower_base.py'), policy='now')
                wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/tasks/tocabi_amp_lower.py'), policy='now')
                wandb.save(os.path.join(os.getcwd(), 'assets/amp/tocabi_motions/tocabi_motions.yaml'), policy='now')
                wandb.save(os.path.join(os.getcwd(), 'isaacgymenvs/learning/amp_continuous.py'), policy='now')
       
            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        try:
            init_wandb()
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)
