
def ncr(n, r):
    total = 1
    for i in range(r):
        total *= (n - i)
        total /= (i + 1)

    return total