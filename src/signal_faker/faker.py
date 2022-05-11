import os.path
import time
from datetime import datetime

import numpy as np

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
FILE = "signal.csv"


def signal(zero, now) -> str:
    delta = (now - zero).total_seconds()
    intercept = 50
    noise = np.random.normal(loc=0, scale=3)
    new_point = np.sin(delta * 2 * np.pi / 120) + intercept
    if delta >= 200:
        noise += (delta // 100) % 10
        if delta % 10 == 0:
            noise += 100
    return f"{now.strftime(DT_FORMAT)},{new_point+noise}\n"


def main():
    if not os.path.exists(FILE):
        with open(FILE, "w"):
            pass
    with open(FILE, "r") as f:
        n_lines = len(f.readlines())
    if n_lines == 0:
        with open(FILE, "a") as f:
            f.write("date,value\n")
    zero = datetime.now()
    while True:
        time.sleep(1)
        with open(FILE, "a") as f:
            f.write(signal(zero, datetime.now()))
        n_lines += 1


if __name__ == "__main__":
    main()
