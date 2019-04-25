## Homework 1

Homework files can be run by `qsub launch`. Note that this assumes that all files are at location `~/iad/hw1`.

File `launch` contains the bash script to run both `ray_tutorial.py` and `map_reduce.py` in the following order.
1. runs `ray_tutorial.py`, if that is success then goes to next line otherwise terminates
2. runs `map_reduce.py` 4 time each with different CPU numbers (1, 2, 4, 8) and saves the total time for each value
3. runs `plot_performance.py` with the values from the previous batch execution and plots a graph of 
total time against number of CPUs.

Sample standard error and standard output is available at `hw1.e14278` and `hw1.o14278` respectively.

Sample plots of total time against number of CPUs is available at `time_vs_cpus.png`.
We observe that there is no significant improvement from 4 to 8 in the graph which could be because of less data.

**NOTE** (map_reduce.py): 
1. in reduce_parallel each summation is dependent on the result so far which is O(n)
2. whereas in reduce_parallel_tree there is an iterative summation of two terms 
and take that as a new term which is O(log(n))