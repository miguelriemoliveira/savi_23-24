#!/usr/bin/env python3


def somar(a, b=4):
    return a + b


def main():

    x = 3
    y = 5

    resultado = somar(x)

    print('somar ' + str(x) + ' + ' + str(y) + ' resulta em ' + str(resultado))


if __name__ == "__main__":
    main()
