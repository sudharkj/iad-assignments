import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    args_len = len(sys.argv)
    lim = int(args_len/2) + 1
    x = sys.argv[1:lim]
    y = [float("%.2f" % float(time)) for time in sys.argv[lim:]]

    plt.clf()
    plt.plot(x, y)
    plt.ylabel('time')
    plt.xlabel('cpus')
    plt.title('Time vs CPUs')
    plt.savefig('time_vs_cpus.png')
