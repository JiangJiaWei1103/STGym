"""
Script for running experiments with multiple random seeds.
Author: JiaWei Jiang
"""
import os
import random
import subprocess
from argparse import Namespace
from typing import List

import yaml

from engine.defaults import TrainEvalArgParser
from paths import CONFIG_PATH

# Define base command
CMD_BASE = [
    "python",
    "-m",
    "tools.main",
    "--project-name",
    "",
    "--model-name",
    "",
    "--eval-on-test",
    "True",
    "--input-path",
    "",
    "--exp-id",
    "",
]


class MutiSeedArgParser(TrainEvalArgParser):
    def __init__(self) -> None:
        super(MutiSeedArgParser, self).__init__()

        self.argparser.add_argument(
            "--n-seeds",
            type=int,
            default=10,
            help="number of seeds (i.e., how many runs to run the same experiments)",
        )


class MultiSeedProcWrapper(object):
    """Wrapper for running the same process with `n_seeds` different
    random seeds.

    Parameters:
        cmd: command to run
        n_seeds: number of seeds
    """

    def __init__(self, cmd: List[str], n_seeds: int, debug: bool = False):
        self.cmd = cmd
        self.n_seeds = n_seeds
        self.debug = debug

    def run(self) -> None:
        seeds = [random.randint(1, 2**32 - 1) for _ in range(self.n_seeds)]
        for seed in seeds:
            self._seed_proc(seed)
            if self.cmd[-1] == "--exp-id":
                cmd = self.cmd.copy()
                cmd.append(str(seed))
            else:
                cmd = self.cmd.copy()

            subprocess.run(cmd)

    def _seed_proc(self, seed: int) -> None:
        """Seed the process."""
        cfg_file = "defaults.yaml" if not self.debug else "defaults_debug.yaml"
        with open(os.path.join(CONFIG_PATH, cfg_file), "r") as f:
            proc_cfg = yaml.full_load(f)
            proc_cfg["seed"] = seed
        with open(os.path.join(CONFIG_PATH, cfg_file), "w") as f:
            yaml.dump(proc_cfg, f)


def main(args: Namespace) -> None:
    # Construct command to run
    cmd = CMD_BASE.copy()
    cmd[4] = str(args.project_name)
    cmd[6] = str(args.model_name)
    cmd[10] = str(args.input_path)
    if args.exp_id is not None:
        cmd[-1] = str(args.exp_id)

    # Run multi-seed processes
    multi_seed_proc_wrapper = MultiSeedProcWrapper(cmd, args.n_seeds)
    multi_seed_proc_wrapper.run()


if __name__ == "__main__":
    # Parse arguments
    args = MutiSeedArgParser().parse()

    # Launch main function
    main(args)
