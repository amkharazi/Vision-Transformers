# Author: A.M.Kharazi
# License: BSD 3 clause

import os

def generate_scripts():
    # Create 16 scripts, each containing 8 test cases
    for script_num in range(1, 17):
        script_name = f"script_{script_num:02}.sh"
        with open(script_name, "w") as file:
            file.write("#!/bin/bash\n\n")
            file.write("# Author: A.M.Kharazi\n")
            file.write("# License: BSD 3 clause\n\n")
            file.write("# Comment test cases you wish not to run, then run the bash file\n\n")
            start = (script_num - 1) * 8 + 1
            end = start + 7
            for i in range(start, end + 1):
                file.write(f"python3 TEST_{i:03}.py\n")
                file.write(f"python3 TEST_{i:03}_val.py\n")
        # Make the script executable
        os.chmod(script_name, 0o755)
        print(f"Generated: {script_name}")

def main():
    print("Generating bash scripts...")
    generate_scripts()
    print("All scripts generated successfully!")

if __name__ == "__main__":
    main()
