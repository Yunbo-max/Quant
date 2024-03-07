# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2023-12-31 09:14:40
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-03-07 10:11:13
import cProfile


def example_function():
    result = 0
    for i in range(1000000):
        result += i
    return result


def main():
    for _ in range(10):
        example_function()


if __name__ == "__main__":
    main()
