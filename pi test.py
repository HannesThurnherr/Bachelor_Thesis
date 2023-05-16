try:
    # import version included with old SymPy
    from sympy.mpmath import mp
except ImportError:
    # import newer version
    from mpmath import mp

mp.dps = 300000  # set number of digits
p = str(mp.pi) # print pi to a thousand places

def test(index, string):
    if string[0]==index[0]:
        if len(index)>1:
            next = test(index[1:],string[1:])
            if len(next)>0:
                return index[0]+next
            else:
                return ""
        else:
            return index
    else:
        return ""

def test_2(index, string):
    if string[:len(index)] == index:
        return index
    else:
        return ""

nonefound = True
for c in range(len(p)):
    res = test_2(str(c), p[c:])
    if len(res)>0:
        print(c,":", test(str(c), p[c:]))
        nonefound = False

if nonefound:
    print("none found")
    


print(p[242424:242433])