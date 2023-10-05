def get_subscript(L):
    """ 
    This function returns the correct subscript for the opt_einsum contraction
    from an MPS structure to a plain wavefunction:
    O--O--O--O  ---> (_______)
    |  |  |  |        | | | |
    Real ugly workaround, forget you are reading this
    """ 

    allchars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' 

    subscript = '' 
    for l in range(L):
        subscript += allchars[2*l:2*l+3]+','

    return subscript[:-1]