from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# noinspection PyUnresolvedReferences
import numpy as np
import ray
import time
import sys

if len(sys.argv) < 2:
    cpus = 4
else:
    cpus = int(sys.argv[1])
print('running map reduce with %d cpus' % cpus)
ray.init(num_cpus=int(cpus),
         include_webui=False,
         ignore_reinit_error=True,
         redis_max_memory=1000000000,
         object_store_memory=10000000000)


def map_serial(function, data):
    return [function(x) for x in data]


def map_parallel(function, data):
    if not isinstance(data, list):
        raise ValueError('The xs argument must be a list.')

    if not hasattr(function, 'remote'):
        raise ValueError('The function argument must be a remote function.')

    # EXERCISE: Modify the list comprehension below to invoke "function"
    # remotely on each element of "xs". This should essentially submit
    # one remote task for each element of the list and then return the
    # resulting list of ObjectIDs.
    return [function.remote(x) for x in data]


def increment_regular(x):
    return x + 1


@ray.remote
def increment_remote(x):
    return x + 1


xs = [1, 2, 3, 4, 5]
result_ids = map_parallel(increment_remote, xs)
assert isinstance(result_ids, list), 'The output of "map_parallel" must be a list.'
assert all([isinstance(x, ray.ObjectID) for x in result_ids]), 'The output of map_parallel must be a list of ObjectIDs.'
assert ray.get(result_ids) == map_serial(increment_regular, xs)
print('Congratulations, the test passed!')


def sleep_regular(x):
    time.sleep(1)
    return x + 1


@ray.remote
def sleep_remote(x):
    time.sleep(1)
    return x + 1


serial_time = 0
parallel_time = 0
start_time = time.time()
results_serial = map_serial(sleep_regular, [1, 2, 3, 4])
serial_time += time.time() - start_time

start_time = time.time()
result_ids = map_parallel(sleep_remote, [1, 2, 3, 4])
results_parallel = ray.get(result_ids)
# noinspection PyRedeclaration
parallel_time = time.time() - start_time

assert results_parallel == results_serial


def reduce_serial(function, data):
    if len(data) == 1:
        return data[0]

    result = data[0]
    for i in range(1, len(data)):
        result = function(result, data[i])

    return result


def add_regular(x, y):
    time.sleep(0.3)
    return x + y


assert reduce_serial(add_regular, [1, 2, 3, 4, 5, 6, 7, 8]) == 36


def reduce_parallel(function, data):
    if not isinstance(data, list):
        raise ValueError('The xs argument must be a list.')

    if not hasattr(function, 'remote'):
        raise ValueError('The function argument must be a remote function.')

    if len(data) == 1:
        return data[0]

    result = data[0]
    for i in range(1, len(data)):
        result = function.remote(result, data[i])

    return result


@ray.remote
def add_remote(x, y):
    time.sleep(0.3)
    return x + y


xs = [1, 2, 3, 4, 5, 6, 7, 8]
result_id = reduce_parallel(add_remote, xs)
assert ray.get(result_id) == reduce_serial(add_regular, xs)
print('Congratulations, the test passed!')


def reduce_parallel_tree(function, data):
    if not isinstance(data, list):
        raise ValueError('The xs argument must be a list.')

    if not hasattr(function, 'remote'):
        raise ValueError('The function argument must be a remote function.')

    # EXERCISE: Think about why that exposes more parallelism.
    # in the previous one each summation is dependent on the result so far which is O(n)
    # but here we do a iterative summation of two terms and take that as a new term which is O(log(n))
    while len(data) > 1:
        r_id = function.remote(data[0], data[1])
        data = data[2:]
        data.append(r_id)
    return data[0]


xs = [1, 2, 3, 4, 5, 6, 7, 8]
result_id = reduce_parallel_tree(add_remote, xs)
assert ray.get(result_id) == reduce_serial(add_regular, xs)

start_time = time.time()
results_serial = reduce_serial(add_regular, [1, 2, 3, 4, 5, 6, 7, 8])
serial_time += time.time() - start_time

result_ids = reduce_parallel(add_remote, [1, 2, 3, 4, 5, 6, 7, 8])
results_parallel = ray.get(result_ids)

assert results_parallel == results_serial

print('\ncalling reduce_parallel_tree')
start_time = time.time()
result_tree_ids = reduce_parallel_tree(add_remote, [1, 2, 3, 4, 5, 6, 7, 8])
results_parallel_tree = ray.get(result_tree_ids)
reduce_time = time.time() - start_time
parallel_time += reduce_time
print('reduce time:', reduce_time)
print('total time:', parallel_time)

assert results_parallel_tree == results_serial
