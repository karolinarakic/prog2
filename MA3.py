""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from functools import reduce

# Exc1
def approximate_pi(n):
    # n is the number of points
    x_inside, y_inside = [], []
    x_outside, y_outside = [], []

    for _ in range(n):
        x = random.uniform(-1, 1) 
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1: 
            x_inside.append(x)
            y_inside.append(y)
        else:
            x_outside.append(x)
            y_outside.append(y)

    n_c = len(x_inside)

    pi_estimate = 4 * n_c / n

    print(f"Antal punkter (n): {n}") 
    print(f"Approximation av pi: {pi_estimate}")

    plt.figure(figsize=(6, 6))
    plt.scatter(x_inside, y_inside, color="red", s=1, label="Innanför")
    plt.scatter(x_outside, y_outside, color="blue", s=1, label="Utanför")
    plt.title(f"pi approx {pi_estimate}, n = {n}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.savefig(f"pi_{n}.png") 
    plt.close()

    return pi_estimate


# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points
    # 1. Skapa n st punkter, där varje punkt har d dimensioner i [-1, 1] (List comprehension)
    points = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]

    # 2. Beräkna avståndet i kvadrat (summan av x_i^2) för varje punkt med map och lambda
    squared_distances = list(map(lambda p: sum(x**2 for x in p), points))

    inside_points = list(filter(lambda dist: dist <= 1.0, squared_distances))

    volume_approx = (2**d) * (len(inside_points) / n)

    # d is the number of dimensions of the sphere 
    return volume_approx
    

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    # d is the number of dimensions of the sphere 
    # 1. Skapa n st punkter, där varje punkt har d dimensioner i [-1, 1] (List comprehension)
    points = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]

    # 2. Beräkna avståndet i kvadrat (summan av x_i^2) för varje punkt med map och lambda
    squared_distances = list(map(lambda p: sum(x**2 for x in p), points))
  
    return (m.pi ** (d / 2)) / m.gamma(d / 2 + 1)

#Exc3: numba version
from numba import njit
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes
    inside = 0

    for i in range(n):
        distance = 0.0

        for j in range(d):
            x = random.uniform(-1.0, 1.0)
            distance += x**2

        if distance <= 1.0:
            inside += 1

    return (2**d) * (inside / n)
  

#Exc4: parallel code - parallelize actual computations by splitting data

def count_inside_points(args):
    n, d = args
    inside = 0

    for _ in range(n):
        distance = 0.0

        for _ in range(d):
            x = random.uniform(-1, 1)
            distance += x**2

        if distance <= 1.0:
            inside += 1

    return inside


def sphere_volume_parallel(n, d, np=10):
    chunk_sizes = [n // np] * np

    for i in range(n % np):
        chunk_sizes[i] += 1

    tasks = [(size, d) for size in chunk_sizes if size > 0]

    with future.ProcessPoolExecutor(max_workers=np) as executor:
        inside_counts = list(executor.map(count_inside_points, tasks))

    inside = sum(inside_counts)

    return (2**d) * (inside / n)

    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    # n = 1000000
    # d = 11
    # start = pc()
    # sphere_volume(n, d)
    # stop = pc()
    # print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    # print("What is numba time?")

    n = 1000000
    d = 11

    print("Ordinary Python:")

    for i in range(3):
        start = pc()
        result = sphere_volume(n, d)
        stop = pc()

        print(f"Call {i + 1}: {stop - start:.4f} seconds")

    print("Numba:")

    for i in range(3):
        start = pc()
        result = sphere_volume_numba(n, d)
        stop = pc()

        print(f"Call {i + 1}: {stop - start:.4f} seconds")  

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    
    start = pc()
    sphere_volume_parallel(n, d)
    stop = pc()
    print(f"Parallel time: {stop - start:.4f} seconds")
    

if __name__ == '__main__':
	main()
