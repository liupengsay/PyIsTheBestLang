

def round(f, n):
    st = f":.{n}f"
    return st.format(f)


print(round(4.25, 1))