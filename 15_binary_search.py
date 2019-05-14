# coding:utf-8
# 二分查找


def binary_search(alist, item):
    """
    递归的方式
    """

    n = len(alist)
    if n > 0:
        mid = n//2

        if alist[mid] == item:
            return True
        elif alist[mid] > item:
            return binary_search(alist[:mid], item)
        else:
            return binary_search(alist[mid+1:], item)

    return False


def binary_search_2(alist, item):
    """
    非递归的方式
    """
    start, end = 0, len(alist)-1

    while start <= end:
        if alist is []:
            return False
        else:
            n = len(alist)
            mid = n//2
            if alist[mid] == item:
                return True
            elif alist[mid] > item:
                end = mid
                alist = alist[start:end]
            else:
                start = mid
                alist = alist[start+1:end]


if __name__ == '__main__':
    l = [16, 20, 26, 31, 44, 54, 55, 77, 93]
    res = binary_search_2(l, 16)
    print(res)
