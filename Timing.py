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
