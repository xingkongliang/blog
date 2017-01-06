---
title: Algorithms - Island Perimeter
layout: post
tags: [Algorithms]
---


### Island Perimeter

You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water. Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells). The island doesn't have "lakes" (water inside that isn't connected to the water around the island). One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

> [[0,1,0,0],
>
> [1,1,1,0],
>
> [0,1,0,0],
>
> [1,1,0,0]]
>
> Answer: 16
>
> Explanation: The perimeter is the 16 yellow stripes in the image below:

![Picture1](https://leetcode.com/static/images/problemset/island.png)

**我的代码：**
```python
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        Perimeter = 0
        a = sum(grid[0])
        b = sum(grid[-1])
        Perimeter = a+b
        for i in range(len(grid)):
            for j in range(len(grid[0])-1):
                if grid[i][j] != grid[i][j+1]:
                    Perimeter += 1
        for i in range(len(grid)-1):
            for j in range(len(grid[0])):
                if grid[i][j] != grid[i+1][j]:
                    Perimeter += 1
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if j == 0  and grid[i][j] == 1:
                    Perimeter += 1
                if j == len(grid[0])-1 and grid[i][j] == 1:
                    Perimeter += 1

        return Perimeter
```

**别人的代码：**
```python
def islandPerimeter(self, grid):
    def water_around(y, x):
        return ((x == 0              or grid[y][x-1] == 0) +
                (x == len(grid[0])-1 or grid[y][x+1] == 0) +
                (y == 0              or grid[y-1][x] == 0) +
                (y == len(grid)-1    or grid[y+1][x] == 0) )
    return sum(water_around(y, x) for y in xrange(len(grid)) for x in xrange(len(grid[0])) if grid[y][x])
```