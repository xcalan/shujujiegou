# coding:utf-8
# 希尔排序(可以根据插入排序改进)


def shell_sort(alist):
    n = len(alist)
    rap = n//2

    while rap > 0:
        for i in range(rap, n):
            for j in range(i, 0, -rap):
                if alist[j] < alist[j-rap]:
                    alist[j], alist[j-rap] = alist[j-rap], alist[j]
                else:
                    break
        rap //= 2
    return alist


if __name__ == '__main__':
    l = [54, 26, 93, 16, 77, 31, 44, 55, 20]
    print(shell_sort(l))