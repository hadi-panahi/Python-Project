# Fibonacci, Optimized Algorithms
# DP Naive Recursion

import datetime as dt

t1 = dt.datetime.now()


def fib(n):
    if n <= 2:
        f = 1
    else:
        f = fib(n-2) + fib(n-1)
    return f


print(fib(30))
t2 = dt.datetime.now()
print((t2 - t1).microseconds)

# DP Recursive Algorithm

memo = {}
t1 = dt.datetime.now()


def fib(n):
    if n in memo:
        return memo[n]
    else:
        if n <= 2:
            f = 1
        else:
            f = fib(n-2) + fib(n-1)
        memo[n] = f
        # print(memo)
        return f


print(fib(995))
t2 = dt.datetime.now()
print((t2 - t1).microseconds)

# Bottom_Up DP algorithm
fib = {}
t1 = dt.datetime.now()


def fib2(n):
    for k in range(1, n+1):
        if k <= 2:
            f = 1
        else:
            f = fib[k-2] + fib[k-1]
        fib[k] = f
    return fib[n]


print(fib2(995))
t2 = dt.datetime.now()
print((t2 - t1).microseconds)
