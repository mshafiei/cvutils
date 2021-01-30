#reflection function
#https://www.fabrizioduroni.it/2017/08/25/how-to-calculate-reflection-vector.html
def reflect(n,I):
    return 2*(n * I).sum(-1,keepdim=True) * n - I