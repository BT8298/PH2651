from sympy.assumptions.handlers.calculus import _
from collections.abc import Iterable

alice_basis = ['×', '+', '×', '+', '×', '+', '×', '×', '×', '×', '×', '×', '+', '×', '+', '+', '×', '+', '+', '×', '×', '×', '×', '+', '+', '+', '×', '×', '×', '+', '×', '×', '×', '×', '+', '+', '+', '×', '+', '+', '×', '+', '×', '+', '×', '+', '+', '×', '×', '+', '×', '+']
bob_basis = ['×', '+', '+', '×', '×', '×', '+', '+', '×', '+', '×', '×', '+', '×', '+', '+', '+', '+', '×', '+', '×', '×', '+', '×', '×', '+', '+', '×', '×', '+', '×', '×', '×', '×', '+', '×', '+', '+', '×', '+', '×', '+', '×', '×', '+', '+', '+', '×', '×', '+', '+', '×']

def pprint_basis_choices(choices: Iterable):
    for i, basis in enumerate(choices):
        print(i+1, basis, sep=': ')

if __name__ == "__main__":
    print("Alice basis choices")
    print("-------------------")
    pprint_basis_choices(alice_basis)
    print("Bob basis choices")
    print("-----------------")
    pprint_basis_choices(bob_basis)
