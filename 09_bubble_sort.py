# coding:utf-8
# 冒泡排序


def bubble_sort(alist):
    n = len(alist)
    for i in range(0, n-1):
        count = 0
        for j in range(i+1, n):
            if alist[i] > alist[j]:
                alist[i], alist[j] = alist[j], alist[i]
                count += 1
        if count == 0:
            break
    return alist


if __name__ == '__main__':
    list = [1, 2, 3, 4, 5, 6]
    print(bubble_sort(list))
