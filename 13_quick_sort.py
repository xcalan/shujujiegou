# coding:utf-8
# 快速排序


def quick_sort(alist, first, end):
    if first >= end:
        return

    n = len(alist)
    low = first
    high = end
    mid_val = alist[first]

    while low < high:
        while low < high and alist[high] > mid_val:
            high -= 1
        alist[low] = alist[high]

        while low < high and alist[low] < mid_val:
            low += 1
        alist[high] = alist[low]

    alist[low] = mid_val

    quick_sort(alist, first, low-1)
    quick_sort(alist, low+1, end)


if __name__ == '__main__':
    l = [54, 26, 93, 16, 77, 31, 44, 55, 20]
    quick_sort(l, 0, len(l)-1)
    print(l)
