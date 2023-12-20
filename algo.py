import numpy as np

from enum import Enum


class State(Enum):
    NO_ZERO = 1
    ZERO_ROW_COL = 2
    ZERO_ROW = 3
    ZERO_COL = 4
    ZERO_EL = 5
    TEMPORARY_SKIPPING_COL = 6

    START = 50
    END = 51

    UNDETERMINED = 100


class Action(Enum):
    INCREMENT_ROW_COL = 1
    INCREMENT_ROW = 2
    INCREMENT_COL = 3
    FIND_NONZERO_EL = 4
    INCREMENT_ROW_AND_RECALL_COL = 5

    START = 50
    END = 51

    UNDETERMINED = 100


class Data():
    def __init__(
            self,
            mat: np.matrix, row: np.matrix, col: np.matrix, CnRm: np.matrix, remainder: np.matrix,
            m: int, n: int, element: float,
            mat_idx: int, remainder_idx: int
    ) -> None:
        self.mat = mat
        self.row = row
        self.col = col
        self.CnRm = CnRm
        self.remainder = remainder
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
        print(f'{self.m=}')
        print(f'{self.n=}')
        print(f'{self.element=}')


class Step():
    def __init__(
            self, number: int, state: State, action: Action, data: Data
        ) -> None:
        self.number = number
        self.state = state
        self.action = action
        self.data = data

    # Discription
    number: int
    state: int = State.UNDETERMINED
    action: int = Action.UNDETERMINED

    # Data 
    data: Data

    # Viewing function
    def view(self):
        print('===================================')
        print(f'{self.number=}')
        print(f'{self.state=}')
        print(f'{self.action=}')
        self.data.view()


class Decomposition():
    original_matrix: np.matrix
    steps = []

    # A = CR
    rows = []
    cols = []
    indices = []

    # A = LU
    L = np.matrix
    U = np.matrix


# The decomposition algorithm
# A = CR
def decom(mat: np.matrix, m: int, n: int, save_n: int, goto_save: bool, saves: Decomposition) -> Decomposition:
    # Get the matrix's dimensions
    num_of_row = mat.shape[0]
    num_of_col = mat.shape[1]

    row_m = mat[m, :]
    col_n = mat[:, n]
    element_mn = mat[m, n]

    if element_mn != 0:
        row_m = row_m / element_mn
    else:
        # A[m][n] = 0
        # Find element A[m][n + a] != 0
        # If cannot find -> This row is a zero-vector -> Do nothing
        temp = n
        while mat[m, temp] == 0:
            temp += 1
            if temp == num_of_col: # Exceeds number of columns -> This row is a zero-vector
                break

        # Found
        # -> Pick out C(n + a) and Rm for the decomposition,
        # instead of Cn and Rm,
        # and remember Cn
        # Afterward, continue with the next row - Cn and R(m + 1)
        if temp != num_of_col:
            saves.steps.append(Step(
                saves.steps[-1].number + 1,
                State.ZERO_EL,
                Action.FIND_NONZERO_EL,
                Data(
                    mat, row_m, col_n, None, None,
                    m, n, element_mn,
                    saves.steps[-1].data.remainder_idx,
                    saves.steps[-1].data.remainder_idx,
                )
            ))
            return decom(mat=mat, m=m, n=temp, save_n=n, goto_save=True, saves=saves)
        
    CnRm = np.dot(col_n, row_m)
    remainder = mat - CnRm

    # Input zero matrix
    if not mat.any(): 
        # Save answer
        saves.rows.append(row_m)
        saves.cols.append(col_n)
        saves.indices.append([m, n])

        # Save step
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.END,
            Action.END,
            Data(
                mat, row_m, col_n, CnRm, remainder,
                m, n, element_mn,
                saves.steps[-1].data.remainder_idx,
                saves.steps[-1].data.remainder_idx + 1,
            )
        ))
        return saves # End of recursion

    # If row_m AND col_n are zero-vectors
    if (
        np.array_equal(row_m, np.zeros((1, num_of_col)))
        and np.array_equal(col_n, np.zeros((num_of_row, 1)))
    ):
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.ZERO_ROW_COL,
            Action.INCREMENT_ROW_COL,
            Data(
                mat, row_m, col_n, CnRm, remainder,
                m, n, element_mn,
                saves.steps[-1].data.remainder_idx,
                saves.steps[-1].data.remainder_idx,
            )
        ))
        return decom(mat=mat, m=m + 1, n=n + 1, save_n=0, goto_save=False, saves=saves)

    # If only row_m is a zero-vector
    if np.array_equal(row_m, np.zeros((1, num_of_col))):
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.ZERO_ROW,
            Action.INCREMENT_ROW,
            Data(
                mat, row_m, col_n, CnRm, remainder,
                m, n, element_mn,
                saves.steps[-1].data.remainder_idx,
                saves.steps[-1].data.remainder_idx,
            )
        ))
        return decom(mat=mat, m=m + 1, n=n, save_n=0, goto_save=False, saves=saves)

    # If only col_n is a zero-vector
    if np.array_equal(col_n, np.zeros((num_of_row, 1))):
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.ZERO_COL,
            Action.INCREMENT_COL,
            Data(
                mat, row_m, col_n, CnRm, remainder,
                m, n, element_mn,
                saves.steps[-1].data.remainder_idx,
                saves.steps[-1].data.remainder_idx,
            )
        ))
        return decom(mat=mat, m=m, n=n + 1, save_n=0, goto_save=False, saves=saves)


    # If we reach this point and the remainder is a zero matrix
    # -> The matrix has been decomposed
    if np.array_equal(mat, CnRm): 
        # Save answer
        saves.rows.append(row_m)
        saves.cols.append(col_n)
        saves.indices.append([m, n])

        # Save step
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.END,
            Action.END,
            Data(
                mat, row_m, col_n, CnRm, remainder,
                m, n, element_mn,
                saves.steps[-1].data.remainder_idx,
                saves.steps[-1].data.remainder_idx + 1,
            )
        ))
        return saves # End of recursion
    else:
        # Reaching the last row and column without decomposing the matrix
        if m == num_of_row - 1 and n == num_of_col - 1: 
            print("Something went wrong?")
            return

    # Save answer
    saves.rows.append(row_m)
    saves.cols.append(col_n)
    saves.indices.append([m, n])

    # Continue decomposing
    if goto_save:
        # Save step
        saves.steps.append(Step(
            saves.steps[-1].number + 1,
            State.TEMPORARY_SKIPPING_COL,
            Action.INCREMENT_ROW_AND_RECALL_COL,
            Data(
                mat, row_m, col_n, CnRm, remainder,
                m, n, element_mn,
                saves.steps[-1].data.remainder_idx,
                saves.steps[-1].data.remainder_idx + 1,
            )
        ))
        return decom(mat=remainder, m=m + 1, n=save_n, save_n=0, goto_save=False, saves=saves)

    # Save step
    saves.steps.append(Step(
        saves.steps[-1].number + 1,
        State.NO_ZERO,
        Action.INCREMENT_ROW_COL,
        Data(
            mat, row_m, col_n, CnRm, remainder,
            m, n, element_mn,
            saves.steps[-1].data.remainder_idx,
            saves.steps[-1].data.remainder_idx + 1,
        )
    ))
    return decom(mat=remainder, m=m + 1, n=n + 1, save_n=0, goto_save=False, saves=saves)