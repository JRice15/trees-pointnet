import time
import signal
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pid",type=int,required=True)
parser.add_argument("--hrs",type=float,required=True)
ARGS = parser.parse_args()

print("checking pid exists...")
os.kill(ARGS.pid, 0)
print("all good.")


print("sleeping for", ARGS.hrs, "hours")

start = time.perf_counter()
while True:
    elapsed = (time.perf_counter() - start) / 3600
    print(elapsed, "hrs elapsed")
    if elapsed > ARGS.hrs:
        print("killing...")
        print("  sigint")
        os.kill(ARGS.pid, signal.SIGINT)
        time.sleep(15)
        print("  sigint")
        os.kill(ARGS.pid, signal.SIGINT)
        time.sleep(15)
        print("  sigint")
        os.kill(ARGS.pid, signal.SIGINT)
        time.sleep(15)
        print("  sigterm")
        os.kill(ARGS.pid, signal.SIGTERM)
        time.sleep(5)
        print("  sigkill")
        os.kill(ARGS.pid, signal.SIGKILL)
        break
    time.sleep(60 * 15)
    

print("done")

