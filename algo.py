import numpy as np

from enum import Enum


class State(Enum):
    NO_ZERO = 1
    ZERO_COL = 2
    ZERO_ROW = 3
    ZERO_EL = 4

    START = 50
    END = 51

    UNDETERMINED = 100


class Permutation():
    def __init__(self, matrix: np.matrix, targets: list) -> None:
        self.matrix = matrix
        targets = targets

    matrix: np.matrix
    targets = []

    def view(self):
        print(f'{self.matrix=}')
        print(f'{self.targets=}')


class Data():
    def __init__(
            self,
            mat: np.matrix, row: np.matrix, col: np.matrix, 
            CnRm: np.matrix, remainder: np.matrix,
            permutation: Permutation,
            m: int, n: int, element: float,
            mat_idx: int, remainder_idx: int
    ) -> None:
        self.mat = mat
        self.row = row
        self.col = col
        self.CnRm = CnRm
        self.remainder = remainder
        self.permutation = permutation
        self.m = m
        self.n = n
        self.element = element
        self.mat_idx = mat_idx
        self.remainder_idx = remainder_idx
        
    mat: np.matrix
    row: np.matrix
    col: np.matrix
    CnRm: np.matrix
    remainder: np.matrix
    permutation: Permutation
    m: int
    n: int
    element: float 
    mat_idx: int
    remainder_idx: int

    def view(self):
        print(f'{self.mat_idx=}')
        print(f'{self.remainder_idx=}')
        print(f'{self.mat=}')
        print(f'{self.row=}')
        print(f'{self.col=}')
        print(f'{self.CnRm=}')
        print(f'{self.remainder=}')
        if self.permutation is not None:
            self.permutation.view()
        print(f'{self.m=}')
        print(f'{self.n=}')
        print(f'{self.element=}')


class Step():
    def __init__(
            self, number: int, state: State, data: Data
        ) -> None:
        self.number = number
        self.state = state
        self.data = data

    # Discription
    number: int
    state: int = State.UNDETERMINED

    # Data 
    data: Data

    # Viewing function
    def view(self):
        print('===================================')
        print(f'{self.number=}')
        print(f'{self.state=}')
        self.data.view()


class Decomposition():
    original_matrix: np.matrix
    steps = []

    rows = []
    cols = []
    indices = []

    # Permutation matrix (for columns)
    permutations = []
    P: np.matrix

    # A = LU
    L: np.matrix
    U: np.matrix

    def calculate_P(self):
        num_of_col = self.original_matrix.shape[1]
        self.P = np.asmatrix(np.eye(num_of_col, num_of_col, dtype=np.float_), dtype=np.float_)
        for permutation in self.permutations:
            self.P = np.matmul(self.P, permutation.matrix)


# Helper functions
def format(x): # Replace all -0.0 with 0.0
    if x == 0.0:
        return 0.0

    if abs(x) < 1e-16:
        x = 0.0
    return x


def vectorized_format(x):
    return np.vectorize(format)(x)


def format(decomposition: Decomposition):
    decomposition.L = vectorized_format(decomposition.L)
    decomposition.U = vectorized_format(decomposition.U)
    for i, row in enumerate(decomposition.rows):
        decomposition.rows[i] = vectorized_format(row)
    for i, col in enumerate(decomposition.cols):
        decomposition.cols[i] = vectorized_format(col)

    for step in decomposition.steps:
        if step.data.mat is not None:
            step.data.mat = vectorized_format(step.data.mat)
        if step.data.row is not None:
            step.data.row = vectorized_format(step.data.row)
        if step.data.col is not None:
            step.data.col = vectorized_format(step.data.col)
        if step.data.CnRm is not None:
            step.data.CnRm = vectorized_format(step.data.CnRm)
        if step.data.remainder is not None:
            step.data.remainder = vectorized_format(step.data.remainder)


# The decomposition algorithm
# A = CR
def decom(mat: np.matrix, m: int, n: int, saves: Decomposition) -> Decomposition:
    num_of_row = mat.shape[0]
    num_of_col = mat.shape[1]
    col_n = mat[:, n]

    # If col_n is a ZERO vector
    if np.array_equal(col_n, np.zeros((num_of_row, 1), dtype=np.float_)):
        row_m = np.asmatrix(np.zeros((1, num_of_col), dtype=np.float_), dtype=np.float_)
        row_m[0, n] = 1.0

        # Save row and col
        format(saves)
        saves.rows.append(row_m)
        saves.cols.append(col_n)
        saves.indices.append([m, n])

        # Record step
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.ZERO_COL,
            Data(
                mat=mat, row=row_m, col=col_n,
                CnRm=None, remainder=mat, permutation=None,
                m=m, n=n, element=0.0,
                mat_idx = saves.steps[-1].data.remainder_idx,
                remainder_idx = saves.steps[-1].data.remainder_idx,
            )
        ))
        return decom(mat=mat, m=m, n=n+1, saves=saves)

    # If col_n is a NON-ZERO vector
    element_mn = mat[m, n]
    row_m = mat[m, :]

    if element_mn != 0.0:
        row_m = row_m / element_mn
        CnRm = np.matmul(col_n, row_m, dtype=np.float_)
        remainder = mat - CnRm

        # If we reach this point and the remainder is a zero matrix
        # -> The matrix has been decomposed
        remainder = vectorized_format(remainder)
        if np.array_equal(remainder, np.zeros(remainder.shape, dtype=np.float_)): 
            print("DONE")
            # Save row and col
            format(saves)
            saves.rows.append(row_m)
            saves.cols.append(col_n)
            saves.indices.append([m, n])

            # Record step
            saves.steps.append(Step(
                saves.steps[-1].number + 1,
                State.END,
                Data(
                    mat=mat, row=row_m, col=col_n,
                    CnRm=CnRm, remainder=remainder, permutation=None,
                    m=m, n=n, element=element_mn,
                    mat_idx = saves.steps[-1].data.remainder_idx,
                    remainder_idx = saves.steps[-1].data.remainder_idx + 1,
                )
            ))
            return saves # End of recursion
        else:
            # Reaching the last row and column without decomposing the matrix
            if m >= num_of_row - 1 and n >= num_of_col - 1: 
                print("Something went wrong?")
                return

        # Linearly independent
        # Save row and col
        format(saves)
        saves.rows.append(row_m)
        saves.cols.append(col_n)
        saves.indices.append([m, n])

        # Record step
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.NO_ZERO,
            Data(
                mat=mat, row=row_m, col=col_n,
                CnRm=CnRm, remainder=remainder, permutation=None,
                m=m, n=n, element=element_mn,
                mat_idx = saves.steps[-1].data.remainder_idx,
                remainder_idx = saves.steps[-1].data.remainder_idx + 1,
            )
        ))
        return decom(mat=remainder, m=m+1, n=n+1, saves=saves)
        
    # Else, element_mn == 0.0
    temp = 1
    while temp < num_of_col:
        if mat[m, temp] == 0.0:
            temp += 1
        else:
            break

    ### Row m is a zero vector
    if temp == num_of_col:
        # We don't take this row

        format(saves)
        # Record step
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.ZERO_ROW,
            Data(
                mat=mat, row=row_m, col=col_n,
                CnRm=None, remainder=mat, permutation=None,
                m=m, n=n, element=0.0,
                mat_idx = saves.steps[-1].data.remainder_idx,
                remainder_idx = saves.steps[-1].data.remainder_idx,
            )
        ))
        return decom(mat=mat, m=m+1, n=n, saves=saves)

    ### Else, permute
    permutation = np.asmatrix(np.zeros((num_of_col, num_of_col), dtype=np.float_), dtype=np.float_)
    for i in range(num_of_col):
        if i == m:
            permutation[m, temp] = 1.0
            continue
        if i == temp:
            permutation[temp, m] = 1.0
            continue
        permutation[i, i] = 1.0

    # We don't take any row or col while permutating
    # Record step
    format(saves)
    permutation_object = Permutation(permutation, [n, temp])
    saves.steps.append(Step(
        saves.steps[-1].number + 1,
        State.ZERO_EL,
        Data(
            mat=mat, row=row_m, col=col_n,
            CnRm=None, remainder=mat,
            permutation=permutation_object,
            m=m, n=n, element=0.0,
            mat_idx = saves.steps[-1].data.remainder_idx,
            remainder_idx = saves.steps[-1].data.remainder_idx,
        )
    ))
    saves.permutations.append(permutation_object)
    return decom(
        mat = np.matmul(mat, permutation, dtype=np.float_), 
        m=m, n=n, saves=saves,
    )
