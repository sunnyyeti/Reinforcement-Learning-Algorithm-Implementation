def newton(func, func_de,inix):
    x = inix
    delta = 10
    while delta>1e-12:
        next_x = x-func(x)/(func_de(x)+1e-15)
        delta = abs(next_x-x)
        x = next_x
    return x


if __name__ == '__main__':
    func = lambda x: x ** 2 - 4
    func_de = lambda x: 2 * x
    print(newton(func,func_de,-3))
