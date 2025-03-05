"""
import socket
def main():
    print(socket.gethostname())
    print('test')

if __name__ == "__main__":
    main()
"""

import subprocess

def install_requirements():
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")

if __name__ == "__main__":
    install_requirements()
