import wandb
import click
import os


@click.command()
@click.argument("project_name")
@click.argument("out_dir")
@click.argument("total_timesteps")
def run(project_name, out_dir, total_timesteps):
    # Directory containing this script.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    program = os.path.join(dir_path, "train.py")

    sweep_configuration = {
        "name": "sweep",
        "method": "bayes",
        "metric": {"name": "eval/mean_reward", "goal": "maximize"},
        "program": program,
        "parameters": {
            "out_dir": {"value": out_dir},
            "total_timesteps": {"value": total_timesteps},
            "free_cpu_frac": {"value": 0.5},
            "eval_freq": {"value": 1e4},
            "n_eval_episodes": {"value": 10},
            "gamma": {"value": 1.0},
            "batch_size": {"values": [1024, 2048, 4096]},
            "n_steps_over_batch": {"values": [1, 2, 3]},
            "ent_coef": {"min": 0.0, "max": 0.02},
            "n_layers": {"values": [2, 3]},
            "units_per_layer": {"values": [64, 128, 256]},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=project_name,
        entity="allen_adastra"
    )
    print(f"sweep_id: {sweep_id}")


if __name__ == "__main__":
    run()
