def stub_element(ele, n):
    """Divides ele into 'n' equal length elements and returns
    a list of these short elements.
    **NOTE**: only use with Drifts and Quads as of now
    """
    short_L = ele.L / n
    short_ele = ele._replace(L=short_L)
    lattice = [short_ele] * n
    
    return lattice

def stub_lattice(lattice, n):
    """Divides every element in the lattice into 'n' elements
    each and returns divided lattice. 
    **NOTE**: only use with Drifts and Quads as of now
    """
    stubbed_lattice = []
    
    for ele in lattice:
        stubbed_lattice.extend(stub_element(ele, n))
        
    return stubbed_lattice