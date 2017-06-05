---
title: Python中multiprocessing.Pool
layout: post
tags: [Python]
---


# multiprocessing.Pool
有些情况下，所要完成的工作可以分解并独立地分布到多个工作进程，对于这种简单的情况，可以用Pool类来管理固定数目的工作进程。作业的返回值会收集并作为一个列表返回。（以下程序cpu数量为2）。

```
import multiprocessing

def do_calculation(data):
    return data*2
def start_process():
    print 'Starting',multiprocessing.current_process().name

if __name__=='__main__':
    inputs=list(range(10))
    print 'Inputs  :',inputs

    builtin_output=map(do_calculation,inputs)
    print 'Build-In :', builtin_output

    pool_size=multiprocessing.cpu_count()*2
    pool=multiprocessing.Pool(processes=pool_size,
        initializer=start_process,)

    pool_outputs=pool.map(do_calculation,inputs)
    pool.close()
    pool.join()

    print 'Pool  :',pool_outputs
```
运行结果：
```
Inputs  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Build-In : [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
Starting PoolWorker-2
Starting PoolWorker-1
Starting PoolWorker-3
Starting PoolWorker-4
Pool  : [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

默认情况下，Pool会创建固定数目的工作进程，并向这些工作进程传递作业，直到再没有更多作业为止。maxtasksperchild参数为每个进程执行task的最大数目，设置maxtasksperchild参数可以告诉池在完成一定数量任务之后重新启动一个工作进程，来避免运行时间很长的工作进程消耗太多的系统资源。

python 2.7.3 

Pool(processes=None, initializer=None, initargs=(), maxtasksperchild=None)

```
import multiprocessing

def do_calculation(data):
    return data*2
def start_process():
    print 'Starting',multiprocessing.current_process().name

if __name__=='__main__':
    inputs=list(range(10))
    print 'Inputs  :',inputs

    builtin_output=map(do_calculation,inputs)
    print 'Build-In :', builtin_output

    pool_size=multiprocessing.cpu_count()*2
    pool=multiprocessing.Pool(processes=pool_size,
        initializer=start_process,maxtasksperchild=2)

    pool_outputs=pool.map(do_calculation,inputs)
    pool.close()
    pool.join()

    print 'Pool  :',pool_outputs
```
运行结果：
```
Inputs  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Build-In : [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
Starting PoolWorker-1
Starting PoolWorker-2
Starting PoolWorker-3
Starting PoolWorker-4
Starting PoolWorker-5
Starting PoolWorker-6
Starting PoolWorker-7
Starting PoolWorker-8
Pool  : [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

池完成其所分配的任务时，即使没有更多的工作要做，也会重新启动工作进程。从这个输出可以看到，尽管只有10个任务，而且每个工作进程一次可以完成两个任务，但是这里创建了8个工作进程。
 

更多的时候，我们不仅需要多进程执行，还需要关注每个进程的执行结果。

```
import multiprocessing
import time

def func(msg):
    for i in xrange(3):
        print msg
        time.sleep(1)
    return "done " + msg

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)
    result = []
    for i in xrange(10):
        msg = "hello %d" %(i)
        result.append(pool.apply_async(func, (msg, )))
    pool.close()
    pool.join()
    for res in result:
        print res.get()
    print "Sub-process(es) done."
```

内容转自：

[python 进程池1 - Pool使用简介](http://www.cnblogs.com/congbo/archive/2012/08/23/2652433.html)

[python 进程池2 - Pool相关函数](http://www.cnblogs.com/congbo/archive/2012/08/23/2652490.html)