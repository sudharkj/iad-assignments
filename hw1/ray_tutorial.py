#!/usr/bin/env python
# coding: utf-8


# ## Part 1 - Remote Functions (15 pts)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import time
import numpy as np
import pickle


ray.init(num_cpus=4,
         include_webui=False,
         ignore_reinit_error=True,
         redis_max_memory=1000000000,
         object_store_memory=10000000000)

# **EXERCISE:** The function below is slow. Turn it into a remote function using the `@ray.remote` decorator.


@ray.remote
def slow_function(i):
    time.sleep(1)
    return i


# **EXERCISE:** The loop below takes too long. The four function calls could be executed in parallel.
# Instead of four seconds, it should only take one second. Once `slow_function` has been made a remote function,
# execute these four tasks in parallel by calling `slow_function.remote()`.
# Then obtain the results by calling `ray.get` on a list of the resulting object IDs.

time.sleep(10.0)
start_time = time.time()

result_ids = [slow_function.remote(i) for i in range(4)]
results = ray.get(result_ids)

end_time = time.time()
duration = end_time - start_time

print('The results are {}. This took {} seconds. Run the next cell to see if the exercise was done correctly.'
      .format(results, duration))

assert results == [0, 1, 2, 3], 'Did you remember to call ray.get?'
assert duration < 1.1, ('The loop took {} seconds. This is too slow.'.format(duration))
assert duration > 1, ('The loop took {} seconds. This is too fast.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))


# ## Part 2 - Parallel Data Processing with Task Dependencies (15 pts)

# **EXERCISE:** You will need to turn all of these functions into remote functions.
# When you turn these functions into remote function, you do not have to worry about whether the caller passes in an
# object ID or a regular object. In both cases, the arguments will be regular objects when the function executes.
# This means that even if you pass in an object ID, you **do not need to call `ray.get`**
# inside of these remote functions.


# noinspection PyUnusedLocal,SpellCheckingInspection
@ray.remote
def load_data(fname):
    time.sleep(0.1)
    return np.ones((1000, 100))


@ray.remote
def normalize_data(x):
    time.sleep(0.1)
    return x - np.mean(x, axis=0)


@ray.remote
def extract_features(norm_x):
    time.sleep(0.1)
    return np.hstack([norm_x, norm_x ** 2])


@ray.remote
def compute_loss(fs):
    num_data, dim = fs.shape
    time.sleep(0.1)
    return np.sum((np.dot(fs, np.ones(dim)) - np.ones(num_data)) ** 2)


assert hasattr(load_data, 'remote'), 'load_data must be a remote function'
assert hasattr(normalize_data, 'remote'), 'normalize_data must be a remote function'
assert hasattr(extract_features, 'remote'), 'extract_features must be a remote function'
assert hasattr(compute_loss, 'remote'), 'compute_loss must be a remote function'


# **EXERCISE:** The loop below takes too long. Parallelize the four passes through the loop by turning
# `load_data`, `normalize_data`, `extract_features`, and `compute_loss` into remote functions and
# then retrieving the losses with `ray.get`.

time.sleep(2.0)
start_time = time.time()

loss_ids = []
for filename in ['file1', 'file2', 'file3', 'file4']:
    inner_start = time.time()

    data = load_data.remote(filename)
    normalized_data = normalize_data.remote(data)
    features = extract_features.remote(normalized_data)
    loss = compute_loss.remote(features)
    loss_ids.append(loss)

    inner_end = time.time()

    if inner_end - inner_start >= 0.1:
        raise Exception('You may be calling ray.get inside of the for loop! '
                        'Doing this will prevent parallelism from being exposed. '
                        'Make sure to only call ray.get once outside of the for loop.')

losses = ray.get(loss_ids)
print('The losses are {}.'.format(losses) + '\n')
loss = sum(losses)

end_time = time.time()
duration = end_time - start_time

print('The loss is {}. This took {} seconds. Run the next cell to see if the exercise was done correctly.'
      .format(loss, duration))

assert loss == 4000
assert duration < 0.8, ('The loop took {} seconds. This is too slow.'.format(duration))
assert duration > 0.4, ('The loop took {} seconds. This is too fast.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))


# ## Part 3 - Introducing Actors (15 pts)

# **EXERCISE:** Change the `Foo` class to be an actor class by using the `@ray.remote` decorator.


@ray.remote
class Foo(object):
    def __init__(self):
        self.counter = 0

    def reset(self):
        self.counter = 0

    def increment(self):
        time.sleep(0.5)
        self.counter += 1
        return self.counter


assert hasattr(Foo, 'remote'), 'You need to turn "Foo" into an actor with @ray.remote.'

# **EXERCISE:** Change the intantiations below to create two actors by calling `Foo.remote()`.

f1 = Foo.remote()
f2 = Foo.remote()


# **EXERCISE:** Parallelize the code below. The two actors can execute methods in parallel
# (though each actor can only execute one method at a time).

time.sleep(2.0)
start_time = time.time()

reset_ids = [f1.reset.remote(), f2.reset.remote()]
_ = [ray.get(reset_id) for reset_id in reset_ids]

result_ids = []
for _ in range(5):
    result_ids.extend([f1.increment.remote(), f2.increment.remote()])
results = ray.get(result_ids)

end_time = time.time()
duration = end_time - start_time

assert not any([isinstance(result, ray.ObjectID) for result in results]), \
    'Looks like "results" is {}. You may have forgotten to call ray.get.'.format(results)

assert results == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

assert duration < 3, ('The experiments ran in {} seconds. This is too slow.'.format(duration))
assert duration > 2.5, ('The experiments ran in {} seconds. This is too fast.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))


# ## Part 4 - Handling Slow Tasks (15 pts)

@ray.remote
def f(i):
    np.random.seed(5 + i)
    x = np.random.uniform(0, 4)
    time.sleep(x)
    return i, time.time()


# **EXERCISE:** Using `ray.wait`, change the code below so that `initial_results`
# consists of the outputs of the first three tasks to complete instead of the first three tasks that were submitted.

time.sleep(2.0)
start_time = time.time()

result_ids = [f.remote(i) for i in range(6)]
ready_ids, remaining_ids = ray.wait(result_ids, num_returns=3)
initial_results = ray.get(ready_ids)

end_time = time.time()
duration = end_time - start_time

# **EXERCISE:** Change the code below so that `remaining_results`
# consists of the outputs of the last three tasks to complete.

remaining_results = ray.get(remaining_ids)

assert len(initial_results) == 3
assert len(remaining_results) == 3

initial_indices = [result[0] for result in initial_results]
initial_times = [result[1] for result in initial_results]
remaining_indices = [result[0] for result in remaining_results]
remaining_times = [result[1] for result in remaining_results]

assert set(initial_indices + remaining_indices) == set(range(6))

assert duration < 1.5, ('The initial batch of ten tasks was retrieved in {} seconds. This is too slow.'
                        .format(duration))

assert duration > 0.8, ('The initial batch of ten tasks was retrieved in {} seconds. This is too slow.'
                        .format(duration))

assert max(initial_times) < min(remaining_times)

print('Success! The example took {} seconds.'.format(duration))


# ## Part 5 - Speed up Serialization (15 pts)

neural_net_weights = {'variable{}'.format(i): np.random.normal(size=1000000) for i in range(50)}

# **EXERCISE:** Compare the time required to serialize the neural net weights and copy them into
# the object store using Ray versus the time required to pickle and unpickle the weights.
# The big win should be with the time required for *deserialization*.

print('Ray - serializing')
start = time.time()
x_id = ray.put(neural_net_weights)
print('time: ', time.time() - start)
# noinspection SpellCheckingInspection
print('\nRay - deserializing')
start = time.time()
x_val = ray.get(x_id)
print('time: ', time.time() - start)

print('\npickle - serializing')
start = time.time()
serialized = pickle.dumps(neural_net_weights)
print('time: ', time.time() - start)
# noinspection SpellCheckingInspection
print('\npickle - deserializing')
start = time.time()
deserialized = pickle.loads(serialized)
print('time: ', time.time() - start)


# noinspection PyUnusedLocal,SpellCheckingInspection
@ray.remote
def use_weights(weights, i):
    return i


# **EXERCISE:** In the code below, use `ray.put`
# to avoid copying the neural net weights to the object store multiple times.

time.sleep(2.0)
start_time = time.time()

results = ray.get([use_weights.remote(x_id, i) for i in range(20)])

end_time = time.time()
duration = end_time - start_time

assert results == list(range(20))
assert duration < 1, ('The experiments ran in {} seconds. This is too slow.'.format(duration))

print('Success! The example took {} seconds.'.format(duration))
