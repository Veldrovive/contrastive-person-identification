import os
import subprocess

def get_git_hash():
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Run the git command
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=script_dir)

        # Decode the output from bytes to string and strip newline
        return git_hash.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        # Handle the case where the script is not in a git repository
        return "Not in a Git repository"
    except FileNotFoundError:
        # Handle the case where git is not installed
        return "Git is not installed"