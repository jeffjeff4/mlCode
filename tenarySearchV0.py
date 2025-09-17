"""
rippling
You are given a one-dimensional strictly convex function F on real interval [a,b], where a < b. You can evaluate the function at a point, but you do not know how F is implemented (ie, you do not have the derivative of F). How would you find the value (a<= x <= b) in which f(x) minimum?
"""

EPSILON = 0.001


def tenarySearchMin(F, a, b):
    # sanity check
    if a > b:
        raise ValueError("the value of a and b are wrong")

    if a == b:
        return a

    while b - a >= EPSILON:
        len0 = (b - a) * 1.0
        mid_left = a + len0 / 3.0
        mid_right = b - len0 / 3.0

        f_mid_left = F(mid_left)
        f_mid_right = F(mid_right)

        if f_mid_left < f_mid_right:
            b = mid_right
        else:
            a = mid_left

    rst = (a + b) * 1.0 / 2.0
    return rst


def F(x):
    return x ** 2 + 1


# test case 0:
a = 0
b = -1
# f0_rst = tenarySearchMin(F, a, b)
# print("f0_rst = ", f0_rst)

# test case 1:
a = 0
b = 0
f1_rst = tenarySearchMin(F, a, b)
print("f1_rst = ", f1_rst)

# test case 2:
a = 0
b = 10
f2_rst = tenarySearchMin(F, a, b)
print("f2_rst = ", f2_rst)

# test case 3:
a = -float("Infinity")
b = float("Infinity")
f3_rst = tenarySearchMin(F, a, b)
print("f3_rst = ", f3_rst)
