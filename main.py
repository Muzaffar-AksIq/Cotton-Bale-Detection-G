import sys
import subprocess
import os
import signal

def kill_process_on_port(port):
    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        pids_to_kill = set()

        for line in lines:
            if f"0.0.0.0:{port}" in line or f"127.0.0.1:{port}" in line:
                parts = line.split()
                pid = parts[-1]
                pids_to_kill.add(int(pid))
    
        if not pids_to_kill:
            print(f"No process is running on port {port}.")
        else:
            for pid in pids_to_kill:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Killed process {pid} on port {port}.")
                except Exception as e:
                    print(f"Failed to kill PID {pid}: {e}")

    except Exception as e:
        print(f"Error checking port {port}: {e}")

if __name__ == "__main__":
    ports = [9000,7862,7863,7860,7000,7861,7864]
    for port in ports:
        kill_process_on_port(port)

    python_executable = sys.executable  # current environment's python

    # Launch app.py
    try:
        # subprocess.Popen([python_executable, "app.py"])
        print("Started app.py")
    except Exception as e:
        print(f"Failed to start app.py: {e}")

    # Launch app2.py
    try:
        # subprocess.Popen([python_executable, "app2.py"])
        print("Started app2.py")
    except Exception as e:
        print(f"Failed to start app2.py: {e}")

    # try:
    #     subprocess.Popen([python_executable, "app3.py"])
    #     print("Started app3.py")
    # except Exception as e:
    #     print(f"Failed to start api_server.py: {e}")
