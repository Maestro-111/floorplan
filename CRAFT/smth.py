import heapq

def next_smaller(n):

    n = list(map(int,list(str(n))))

    first_inc = None
    for i in range(len(n)-1,0,-1):
        if n[i-1]>n[i]:
            first_inc = i-1
            break

    if first_inc is None:
        return -1

    m = (n[first_inc+1],first_inc+1)

    for j in range(first_inc+2, len(n)):
        if n[j]>m[0] and n[j]!=n[first_inc]:
            m = (n[j],j)


    if m[0]==0:
        return -1

    n[first_inc],n[m[1]] = n[m[1]],n[first_inc]
    n[first_inc+1:] = n[first_inc+1:][::-1]

    res = 0
    deg = 1
    k = len(n)
    print(n)
    while k:
        num=n[k-1]
        res += (num*deg)
        deg = deg*10
        k -= 1

    return res


print(next_smaller(901))