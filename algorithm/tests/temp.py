def check(x0, y0, x2, y2):
    x1 = (x0+x2+y2-y0)/2
    y1 = (y0 + y2 + x0 - x2) / 2
    x3 = (x0 + x2 - y2 + y0) / 2
    y3 = (y0 + y2 - x0 + x2) / 2
    return (x1, y1), (x3, y3)


print(check(0, 0, 0, 2))