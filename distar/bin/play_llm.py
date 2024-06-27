# main_program.py

import os
import argparse
import shutil
import torch
import multiprocessing as mp
from queue import Empty
from distar.actor import Actor
from distar.ctools.utils import read_config
from distar.bin.input_receiver import input_receiver  # Import input_receiver function


def input_receiver_process(queue):
    input_receiver(queue)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default=None, help="Model 1 path")
    parser.add_argument("--model2", type=str, default=None, help="Model 2 path")
    parser.add_argument("--cpu", action="store_true", help="Use CPU inference")
    parser.add_argument(
        "--game_type", type=str, default="agent_vs_bot", help="Game type"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Initialize environment and configurations
    sc2path = os.getenv("SC2PATH", "")
    assert os.path.exists(sc2path), f"SC2PATH: {sc2path} does not exist!"

    # Load user configuration
    user_config = read_config(
        os.path.join(os.path.dirname(__file__), "user_config.yaml")
    )
    user_config.actor.job_type = "eval_test"
    user_config.common.type = "play"
    user_config.actor.episode_num = 1
    user_config.env.realtime = False

    # Parse command line arguments
    args = get_args()

    # Set default model path
    default_model_path = os.path.join(os.path.dirname(__file__), "rl_model.pth")

    # Set model paths based on command line arguments
    if args.model1 is not None:
        model1 = os.path.join(os.path.dirname(__file__), f"{args.model1}.pth")
        user_config.actor.model_paths["model1"] = model1
    else:
        model1 = user_config.actor.model_paths["model1"]
        if model1 == "default":
            user_config.actor.model_paths["model1"] = default_model_path
            model1 = default_model_path

    if args.model2 is not None:
        model2 = os.path.join(os.path.dirname(__file__), f"{args.model2}.pth")
        user_config.actor.model_paths["model2"] = model2
    else:
        model2 = user_config.actor.model_paths["model2"]
        if model2 == "default":
            user_config.actor.model_paths["model2"] = default_model_path
            model2 = default_model_path

    # Validate model paths
    assert os.path.exists(
        model1
    ), f"Model 1 file: {model1} does not exist, please download the model first!"
    assert os.path.exists(
        model2
    ), f"Model 2 file: {model2} does not exist, please download the model first!"

    # Configure CUDA usage
    if not args.cpu:
        assert (
            torch.cuda.is_available()
        ), "CUDA is not available, please install CUDA first!"
        user_config.actor.use_cuda = True
    else:
        user_config.actor.use_cuda = False
        print(
            "Warning: CUDA is not activated, this will cause significant performance degradation!"
        )

    # Configure game type
    assert args.game_type in [
        "agent_vs_agent",
        "agent_vs_bot",
        "human_vs_agent",
    ], "Game type only supports 'agent_vs_agent', 'agent_vs_bot', or 'human_vs_agent'!"

    if args.game_type == "agent_vs_agent":
        user_config.env.player_ids = [
            os.path.basename(model1).split(".")[0],
            os.path.basename(model2).split(".")[0],
        ]
    elif args.game_type == "agent_vs_bot":
        user_config.env.player_ids = [os.path.basename(model1).split(".")[0], "bot10"]
    elif args.game_type == "human_vs_agent":
        user_config.env.player_ids = [os.path.basename(model1).split(".")[0], "human"]

    # Start input receiver process
    queue = mp.Queue()
    input_process = mp.Process(target=input_receiver_process, args=(queue,))
    input_process.start()

    # Initialize and run actor
    actor = Actor(user_config)
    actor.run(queue)  # Pass the queue to Actor.run()

    try:
        while True:
            # Check the queue for any new inputs
            try:
                message = queue.get_nowait()
                print(f"Received input in main program: {message}")
                # Do something with the received input, e.g., update configuration or control logic
            except Empty:
                pass
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping main program.")

    # Clean up
    input_process.terminate()
    input_process.join()
