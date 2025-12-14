import subprocess
import sys

def main():
    try:
        subprocess.run(
            [sys.executable, "-m", "piptools", "compile", "requirements.in", "--output-file", "requirements.txt"],
            check=True
        )
        print("✅ requirements.txt generated successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to compile requirements: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
