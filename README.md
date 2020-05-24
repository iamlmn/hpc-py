
# High Performance Computing, distributed, concurrent and parallel programming in Python

### Takeaways
1. Use lambda expression,generators and iterators to speed up your code.
2. MultiProcessing & MultiThreading in Python.
3. Optimize performance and efficiency by leveraging Nummpy,Sccipy & Cython for numerical computations. 
4. Working on loading larger than memory data using Dask in a distributed and parallel setting.
5. Leverage the power of Numba to make your python program run faster.
6. Build reactive applications using python.

## 1.a. Lambda Expressions


An alternative way to write functions. If the lambda functions are not assigned to any variable then it is an anonymous function(generally meant for one time use only).
Can be used along with filter(),reduce() & map.


```python
# sqrt of a number
my_sqr_func = lambda x: x*x
my_sqr_func(5)
```




    25




```python
# adder
my_adder = lambda x,y : x+y
my_adder(3,4)
```




    7



###### Using map


```python
my_list = [1, 2, 3, 4]
my_sqr_list = map(my_sqr_func, my_list)
list(my_sqr_list)
```




    [1, 4, 9, 16]



##### Using filter


```python
# filters list of elements with even lenght of char
my_str_list = ['hello', 'bye','Damn', 'Yolo']
my_filtered_list = list(filter(lambda x : len(x)%2 == 0,my_str_list))
my_filtered_list
```




    ['Damn', 'Yolo']



##### Sort


```python
my_dict_list = [{'Name':'lmn', 'Age':25}, {'Name':'shinchan','Age':12}, {'Name':'Himavari','Age':6}]
# Sort by age
sorted(my_dict_list, key = lambda x : x['Age'])
# sort by nam
sorted(my_dict_list, key = lambda x : x['Name'])
```




    [{'Name': 'Himavari', 'Age': 6},
     {'Name': 'lmn', 'Age': 25},
     {'Name': 'shinchan', 'Age': 12}]



##### Nesting Lambda Expressions


```python
add_prefix = lambda suffix : (lambda my_str : my_str + ' ' + suffix)
add_world = add_prefix("World")
add_world("Hello")
```




    'Hello World'



## 1.b Comprehensions for speed ups
Comprehensions is a way of creating list or dictionaries based on existing iterables. 


```python
my_list = [1, 2, 3, 4, 5]
# List comprehension
[elem*100 for elem in my_list]
# Dict comprehension
{elem : elem*100 for elem in my_list}
```




    {1: 100, 2: 200, 3: 300, 4: 400, 5: 500}



I understand your confusion, you obviously know this? But I just wanted to highlight its importance as we often forget it while working with lists.


```python
import time
```


```python
start = time.time()
new_list = []
for x in range(0,9000000):
    new_list.append(x**2)
end = time.time()
print("Time taken is {}".format(end-start))
```

    Time taken is 7.174314022064209



```python
start = time.time()
new_list = []
new_list = [elem**2 for elem in range(0,9000000)]
end = time.time()
print("Time taken is {}".format(end-start))
```

    Time taken is 5.369340896606445


###### Amazed? I was
The reason why its fast is because they do not require individual append calls and append is done explicitly. For large lists List comprehensions are usually fast.  

## 1.c Generators & Iterators

##### What are Iterators?
1. Simply a type of object which allows us to go over each of it's elements.
2. Each iterator has an iterator protocol implemented inside it, whicg includes __iter__ & __next__ methods.
3. They use lazy evaluation to save memory and do not evaluate all values in one go.

Create your own multiplier iterator with the range you want


```python
class myIter:
    def __init__(self, start, final_n, multiplier):
        self.start = start
        self.final_n = final_n
        self.multiplier = multiplier
        
    def __iter__(self):
        self.curr_n = self.start
        return self
    
    def __next__(self):
        if self.curr_n <= self.final_n:
            multiple = self.curr_n
            self.curr_n += self.multiplier
            return multiple
        else:
            raise StopIteration
```


```python
my_iter_init = myIter(1,100,5)
iter(my_iter_init)
next(my_iter_init)
```




    1




```python
[x for x in range(1,200000000000000000000000000000,5)]

```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-22-0fe858eaaf61> in <module>
    ----> 1 [x for x in range(1,200000000000000000000000000000,5)]
    

    <ipython-input-22-0fe858eaaf61> in <listcomp>(.0)
    ----> 1 [x for x in range(1,200000000000000000000000000000,5)]
    

    KeyboardInterrupt: 


As you observed the code froze beccuase but the iterator we wrote will not


```python
my_iter_init = myIter(1,200000000000000000000000000000,5)
iter(my_iter_init)
next(my_iter_init)
```




    1




```python
next(my_iter_init)
next(my_iter_init)
next(my_iter_init)
```




    16



The reason for that is it does not try to create that list completely at one go. It uses lazy evaluation. Hence it will be faster and won't run out of memory while working on very large lists

## Generators
Generators are functions which return one value at a time.
They are just like iterators but with using Yield keyword
The Yield keyword is used to resume/halt the processing of generators.
If a functions contains Yield keyword then its considered to be a generators


```python
# Example
```


```python
def my_gen():
    x = 5
    y = 6
    print("Processing Halted1")
    yield x,y
    print("Processing Resumed2")
    x += y
    print("Processing Halted2")
    yield x,y
    print("Processing Resumed3")
    x += 1
    y += 10
    print("Processing Halted3")
    yield x,y
    
# Call the generator function

for x,y in my_gen():
    print(x,y)
```

    Processing Halted1
    5 6
    Processing Resumed2
    Processing Halted2
    11 6
    Processing Resumed3
    Processing Halted3
    12 16


###### Similar to iterators , or can be called as an alternative to Iterators except they use Yield and not the iterator protocal.

### Decorators 
for time analysis

- Enhances the functionality of an existing function by taking the functions as input and returning a new function
- Can be applied on any callable objects like functions, methods & classes.


```python
## Examples
```


```python
def my_simple_function():
    print("I am a normal function")
```


```python
my_simple_function()
```

    I am a normal function



```python
def my_decorator(my_fn):
    def wrapped_function():
        print("This is the start of decorated function!")
        my_fn()
        print("This is the end of decorated function!")
    return wrapped_function
```


```python
yo_fn = my_decorator(my_simple_function)
```


```python
yo_fn()
```

    This is the start of decorated function!
    I am a normal function
    This is the end of decorated function!


# Using @notations


```python
@my_decorator
def my_new_function():
    print("I do nothing")
```


```python
my_new_function()
```

    This is the start of decorated function!
    I do nothing
    This is the end of decorated function!


#### Using kwargs


```python
def my_better_decorator(my_fn):
    def modified_function(*args, **kwargs):
        print("This is the start of decorated function!")
        my_fn(*args, **kwargs)
        print("This is the end of decorated function!")
    return modified_function

```


```python
@my_better_decorator
def my_adder(x,y):
    print("Sum of x,y is {}".format(x + y))
```


```python
my_adder(6,7)
```

    This is the start of decorated function!
    Sum of x,y is 13
    This is the end of decorated function!


## Using decorator for Time analysis


```python
import time
def time_decor(my_fun):
    def modified_func(*args, **kwargs):
        start = time.time()
        my_fun(*args, **kwargs)
        end = time.time()
        print("Time taken in seconds {}".format(end - start))
    return modified_func
```


```python
@time_decor
def create_big_list(size):
    new_list = []
    for x in range(0,size):
        new_list.append(x)
    return new_list 
```


```python
create_big_list(100000000)
```

    Time taken in seconds 19.43321132659912



```python
@time_decor
def create_big_list_faster(size):
    return [x for x in range(0,size)]
```


```python
create_big_list_faster(100000000)
```

    Time taken in seconds 10.715726852416992


Lists creation with 100Million elements took 21.7s with out list comprehensions & 13.4 with list comprehensions. This approach could save a lot more time when we write actually computationally intensive programms in python like while dealing with biomdedicne or bioinformatics related computations on DNa/RNA data (Genomic Sequenceing etc).

Where and why decorators?
- Decorator enhances the functionality of existing funcitons.
- Arguments can also be used to decorators. 
- We can use them for time analysis as well.

# 2. Parallel Programming in Python

### 2.a Threading module

##### What are threads?
- Is a part of a process executing in memory.
- There can be multiple threads within the same process.
- Multiple threads within the same memory can share data.

Python threading module:
 - Threading module in python allows us to control the threads in Python.
 - It contains functions like for performing operations like spawning threads, synchronizing threads and so on.


```python
from threading import Thread

def simple_func(length):
    sum_f1 = 0
    for x in range(0,length):
        sum_f1 += x 
    print("Normal sum of is {}".format(sum_f1))
    
def simple_sqrt(length):
    sum_f2 = 0
    for x in range(0,length):
        sum_f2 += x*x 
    print("Square sum of is {}".format(sum_f2))
    
def simple_cubic(length):
    sum_f3 = 0
    for x in range(0,length):
        sum_f3 += x*x*x 
    print("Cubic sum of is {}".format(sum_f3))
    
def do_threading(length = 1000000):
    thread_simple = Thread(target=simple_func, args=(length,))
    thread_sqrt = Thread(target=simple_sqrt, args=(length,))
    thread_cubic = Thread(target=simple_cubic, args=(length,))
    
    # Start your threads
    thread_simple.start()
    thread_sqrt.start()
    thread_cubic.start()
    
    # wait for the threads to finish
    thread_simple.join()
    thread_sqrt.join()
    thread_cubic.join()
def without_threading(length = 1000000):
    simple_func(length)
    simple_sqrt(length)
    simple_cubic(length)
```


```python
%%time
do_threading()
```

    Normal sum of is 499999500000
    Square sum of is 333332833333500000
    Cubic sum of is 249999500000250000000000
    CPU times: user 524 ms, sys: 9.99 ms, total: 534 ms
    Wall time: 527 ms



```python
%%time
without_threading()
```

    Normal sum of is 499999500000
    Square sum of is 333332833333500000
    Cubic sum of is 249999500000250000000000
    CPU times: user 535 ms, sys: 7.18 ms, total: 542 ms
    Wall time: 542 ms


Threading is found to be slower in Python which is usually assumed to be caused due to GIL.
- A thread can be considered as a light weight process.
- Multiple threads can exist within the same process.
- Threading module allows us to control and manage threads in Python.

### 2.b  Threads with Locks

What is the need for Locks?
Sometimes multiple threads can try to access the same of piece of the data at the same time.
Race condition leads to inconsistent information. Inorder to avoid this we introduce Locks.

#### Using locks within threads
The threading module in Python contain locks for implementinng Synchronization mechanism.

It contains the following methods.
 - acquire  : This essentially locks the lock.
 - release : releases the lock.


```python
from threading import Thread,Lock
# define lock
thread_lock = Lock()

my_global_string = "Lmn"
global_check = True
def add_prefix(prefix_to_add):
    global my_global_string,global_check
    # Acquire the lock over the data shared between threads.
    thread_lock.acquire()
    global_check = False

    my_global_string = prefix_to_add + " " + my_global_string
    
    thread_lock.release()

def add_suffix(suffix_to_add):
    global my_global_string,global_check
    # Acquire the lock over the data shared between threads.
    thread_lock.acquire()
    #global_check = False
    if global_check:
        my_global_string =  my_global_string + " " + suffix_to_add
    
    thread_lock.release()
    
    
def do_threading():
    thread_prefix = Thread(target=add_prefix, args=("YOLO",))
    thread_suffix = Thread(target=add_suffix, args=(",Bye!!",))
   
    
    # Start your threads
    thread_prefix.start()
    thread_suffix.start()
    
    # wait for the threads to finish
    thread_prefix.join()
    thread_suffix.join()
    
    global my_global_string
    print("Final string is {}".format(my_global_string))

do_threading()
```

    Final string is YOLO Lmn


###### Why Rlocks?

- Problem with conventional locks is that we don't know which thread is holding the lock. So incase thread X is holding the lock and if X tries to acquire the lock,it gets blocked even though X is only holding the lock.

This can be overcome using Re-entrant locks or Rlock.


```python
from threading import RLock,Lock
my_lock = RLock()
# my_lock = Lock()
# if you try it with lock it gets blocked and wont finish executing.
my_lock.acquire()
my_global = 'hello'

my_lock.acquire()
my_global = '2'+ 'hello'

my_global
```




    '2hello'



Use case of Rlocks - recursion using threads
Summary:
- Locks are used for threaded synchronization.
- Without locks there can be inconsisted data with multiple threads trying to access at the same time.

### 2.c Global Interpreter Lock (GIL)

* Threads are managed by OS. Can be either POSIX threads or windows threads.
* GIL ensures that only one thread is run in an interpreter once, 
* This hampers the threading purpose indirectly limiting the parallelism.
* This is done to simplify memory management between threads.

### 2.d MultiProcessing

1. In multi processing we spawn multiple process to execute in parallel
2. Similar to threads but multiple processes are created instead of threads
3. Multiprocessing in Python, does not suffer from GIL limitation suffered by multi threading.
4. Each prrocess has its own memory space as opposed to of threads which uses shared memory.


```python
from multiprocessing import Process
```


```python
def simple_func(length):
    sum_f1 = 0
    for x in range(0,length):
        sum_f1 += x 
    print("Normal sum of is {}".format(sum_f1))
    
def simple_sqrt(length):
    sum_f2 = 0
    for x in range(0,length):
        sum_f2 += x*x 
    print("Square sum of is {}".format(sum_f2))
    
def simple_cubic(length):
    sum_f3 = 0
    for x in range(0,length):
        sum_f3 += x*x*x 
    print("Cubic sum of is {}".format(sum_f3))
    
def do_multi_processing():
    length = 5
    process_simple = Process(target = simple_func,args = (length,))
    process_square = Process(target = simple_sqrt,args = (length,))
    process_cubic = Process(target = simple_cubic,args = (length,))
    
        
    # Start your process
    process_simple.start()
    process_square.start()
    process_cubic.start()
    
    # wait for the processes to finish
    process_simple.join()
    process_square.join()
    process_cubic.join()
    
    
def nornal_sequentiial_fn():
    length = 5
    simple_func(length)
    simple_sqrt(length)
    simple_cubic(length)
```


```python
%%time
do_multi_processing()
```

    Normal sum of is 10
    Square sum of is 30
    Cubic sum of is 100
    CPU times: user 11.5 ms, sys: 22.4 ms, total: 33.9 ms
    Wall time: 34.5 ms



```python
%%time
nornal_sequentiial_fn()
```

    Normal sum of is 10
    Square sum of is 30
    Cubic sum of is 100
    CPU times: user 436 µs, sys: 316 µs, total: 752 µs
    Wall time: 520 µs



```python

```
