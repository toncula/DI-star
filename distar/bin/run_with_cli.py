# run_actor.py

import subprocess

if __name__ == "__main__":
    # Replace 'python' with your actual Python interpreter executable if needed
    subprocess.Popen(
        ["python", "./DI-star/distar/bin/cli.py"], creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    # Run the Actor program in the current command prompt
    subprocess.run(["python", "./DI-star/distar/bin/play_llm.py"])
