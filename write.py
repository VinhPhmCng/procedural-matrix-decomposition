import numpy as np
from algo import *

from pylatex.base_classes import Environment
from pylatex.package import Package
from pylatex import Document, Math, Matrix, Alignat, Command, Section, Subsection
from pylatex.utils import NoEscape, bold


class bNiceMatrix(Environment):
    _latex_name = 'bNiceMatrix'
    packages = [Package('nicematrix')]


    def __init__(self, matrix, *, color=None, rows=None, cols=None):
        self.matrix = matrix

        if color is not None:
            self.color = color
        if rows is not None:
            self.rows = rows
        if cols is not None:
            self.cols = cols

        super().__init__()

    def dumps_content(self):
        string = ''
        matrix_string = ''
        rows_string = ''
        cols_string = ''

        #
        shape = self.matrix.shape
        for (y, x), value in np.ndenumerate(self.matrix):
            if x:
                matrix_string += '&'
            matrix_string += str(value)

            if x == shape[1] - 1 and y != shape[0] - 1:
                matrix_string += r'\\' + '%\n'

        #
        if self.color is not None and self.rows is not None:
            rows_string = r'\rowcolor{' + self.color + r'}{' + self.rows + r'}'

        #
        if self.color is not None and self.cols is not None:
            cols_string = r'\columncolor{' + self.color + r'}{' + self.cols + r'}'

        #
        string += r'\CodeBefore' + '%\n'
        string += rows_string + '%\n'
        string += cols_string + '%\n'
        string += r'\Body' + '%\n'
        string += matrix_string
        
        super().dumps_content()

        return string


def write_answer(doc: Document, decom: Decomposition):
    doc.append('The matrix can be decomposed into the following combinations:')

    # Write A = CR
    with doc.create(Alignat(numbering=False, escape=False)) as agn:
        agn.append(Matrix(decom.original_matrix, mtype='b'))
        agn.append(r' &= ')

        for i in range(len(decom.rows)):
            agn.append(Matrix(decom.cols[i], mtype='b'))
            agn.append('\cdot')
            agn.append(Matrix(decom.rows[i], mtype='b'))
            if i != len(decom.rows) - 1:
                #agn.append(r'\\ &+ ')
                agn.append(r' + ')

    # Write A = LU
    doc.append('Another representation is: ')
    #doc.append(Math(data=[r'A = LU'], inline=True, escape=False))

    if len(decom.permutations) > 0:
        doc.append(Math(
            data=[
                Matrix(decom.original_matrix, mtype='b'),
                r' \cdot ',
                Matrix(decom.P, mtype='b'),
                r' = ',
                Matrix(decom.L, mtype='b'),
                r' \cdot ',
                Matrix(decom.U, mtype='b'),
            ],
            inline=False,
            escape=False,
        ))
    else:
        doc.append(Math(
            data=[
                Matrix(decom.original_matrix, mtype='b'),
                r' = ',
                Matrix(decom.L, mtype='b'),
                r' \cdot ',
                Matrix(decom.U, mtype='b'),
            ],
            inline=False,
            escape=False,
        ))
    return


def write_detailed_solution(doc: Document, decom: Decomposition):
    def write_row(doc: Document, data: Data):
        doc.append(Math(data=[r'R_{', data.m + 1, r'} \ '], inline=True, escape=False))
        return


    def write_col(doc: Document, data: Data):
        doc.append(Math(data=[r'C_{', data.n + 1, r'} \ '], inline=True, escape=False))
        return


    def write_element(doc: Document, data: Data):
        doc.append(Math(
            data=[
                r'e_{', data.m + 1, data.n + 1,  r'} = ', data.element, r'\ '
            ], 
            inline=True, 
            escape=False,
        ))
        return


    def write_step_state(doc: Document, step: Step):
        data = step.data
        match step.state:
            case State.START:
                doc.append('We prepare ourselves mentally.')
                return

            case State.END:
                doc.append('Finally, we have the matrix')
                doc.append(Math(
                    data=[
                        r'A_{', data.mat_idx, r'} = ',
                        Matrix(data.mat, mtype='b'),
                        #bNiceMatrix(
                            #data.mat, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                        #)
                    ], 
                    inline=False, 
                    escape=False,
                ))
                doc.append('We can see that the row ')
                write_row(doc, data)
                doc.append('and the column ')
                write_col(doc, data)
                doc.append('are both ')
                doc.append(bold('non-zero vectors'))
                doc.append(', meaning their multiplication will be a non-zero matrix. ')
                doc.append('Therefore, they are ')
                doc.append(bold('linearly independent'))
                doc.append(', and thus we can pick them out.')
                doc.append('\nAlso, we have to divide either one of them by the element ')
                write_element(doc, data)
                doc.append('to ensure it is not duplicated in the matrix multiplication.')
                doc.append('\nAnd then, we have')
                with doc.create(Alignat(numbering=False, escape=False)) as agn:
                    agn.append(r'C_{')
                    agn.append(data.n  + 1)
                    agn.append(r'} \cdot')
                    agn.append(r'R_{')
                    agn.append(data.m + 1)
                    agn.append(r'} &= ')
                    agn.append(Matrix(data.col, mtype='b'))
                    agn.append(r'\cdot')
                    agn.append(Matrix(data.row, mtype='b'))
                    agn.append(r'\\ &= ')
                    agn.append(Matrix(data.CnRm, mtype='b'))

                doc.append(', which will always be exactly equal to ')
                doc.append(Math(data=[r'A_{', data.mat_idx, r'} \ '], inline=True, escape=False))
                doc.append('.\nThus, we are left with the last matrix ')
                doc.append(Math(data=[r'A_{', data.remainder_idx, r'} \ '], inline=True, escape=False)) 
                doc.append(r'with a ')
                doc.append(bold('rank of zero'))
                doc.append(r', or in other words, a ')
                doc.append(bold('zero-matrix'))
                doc.append(r'.')
                with doc.create(Alignat(numbering=False, escape=False)) as agn:
                    agn.append(r'A_{')
                    agn.append(data.mat_idx)
                    agn.append(r'} - ')
                    agn.append(r'C_{')
                    agn.append(data.n  + 1)
                    agn.append(r'} \cdot')
                    agn.append(r'R_{')
                    agn.append(data.m + 1)
                    agn.append(r'} &= ')
                    agn.append(Matrix(data.mat, mtype='b'))
                    agn.append(r' - ')
                    agn.append(Matrix(data.CnRm, mtype='b'))
                    agn.append(r'\\ = A_{')
                    agn.append(data.remainder_idx)
                    agn.append(r'} &= ')
                    agn.append(Matrix(data.remainder, mtype='b'))
                    #agn.append(bNiceMatrix(
                            #data.remainder, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                    #))
                doc.append('Bravo! We have successfully decomposed the matrix.')
                return

            case State.NO_ZERO:
                doc.append('We have the matrix')
                doc.append(Math(
                    data=[
                        r'A_{', data.mat_idx, r'} = ',
                        Matrix(data.mat, mtype='b'),
                        #bNiceMatrix(
                            #data.mat, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                        #)
                    ], 
                    inline=False, 
                    escape=False,
                ))
                doc.append('We can see that the row ')
                write_row(doc, data)
                doc.append('and the column ')
                write_col(doc, data)
                doc.append('are both ')
                doc.append(bold('non-zero vectors'))
                doc.append(', meaning their multiplication will be a non-zero matrix. ')
                doc.append('Therefore, they are ')
                doc.append(bold('linearly independent'))
                doc.append(', and thus we can pick them out.')
                doc.append('\nAlso, we have to divide either one of them by the element ')
                write_element(doc, data)
                doc.append('to ensure it is not duplicated in the matrix multiplication.')
                doc.append('\nAnd then, we have')
                with doc.create(Alignat(numbering=False, escape=False)) as agn:
                    agn.append(r'C_{')
                    agn.append(data.n  + 1)
                    agn.append(r'} \cdot')
                    agn.append(r'R_{')
                    agn.append(data.m + 1)
                    agn.append(r'} &= ')
                    agn.append(Matrix(data.col, mtype='b'))
                    agn.append(r'\cdot')
                    agn.append(Matrix(data.row, mtype='b'))
                    agn.append(r'\\ &= ')
                    agn.append(Matrix(data.CnRm, mtype='b'))

                doc.append(', which when substracted from ')
                doc.append(Math(data=[r'A_{', data.mat_idx, r'} \ '], inline=True, escape=False))
                doc.append(r'will result in a new matrix ')
                doc.append(Math(data=[r'A_{', data.remainder_idx, r'} \ '], inline=True, escape=False))
                doc.append(r'with a ')
                doc.append(bold('lower rank'))
                with doc.create(Alignat(numbering=False, escape=False)) as agn:
                    agn.append(r'A_{')
                    agn.append(data.mat_idx)
                    agn.append(r'} - ')
                    agn.append(r'C_{')
                    agn.append(data.n  + 1)
                    agn.append(r'} \cdot')
                    agn.append(r'R_{')
                    agn.append(data.m + 1)
                    agn.append(r'} &= ')
                    agn.append(Matrix(data.mat, mtype='b'))
                    agn.append(r' - ')
                    agn.append(Matrix(data.CnRm, mtype='b'))
                    agn.append(r'\\ = A_{')
                    agn.append(data.remainder_idx)
                    agn.append(r'} &= ')
                    agn.append(Matrix(data.remainder, mtype='b'))
                    #agn.append(bNiceMatrix(
                            #data.remainder, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                    #))

                doc.append('because ')
                write_row(doc, data)
                doc.append('and ')
                write_col(doc, data)
                doc.append('have been reduced to 0.')
                return

            case State.ZERO_COL:
                doc.append('We have the matrix')
                doc.append(Math(
                    data=[
                        r'A_{', data.mat_idx, r'} = ',
                        Matrix(data.mat, mtype='b'),
                        #bNiceMatrix(
                            #data.mat, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                        #)
                    ], 
                    inline=False, 
                    escape=False,
                ))
                doc.append('We can see that the column ')
                write_col(doc, data)
                doc.append('is a ')
                doc.append(bold('zero vector'))
                doc.append(', meaning validly multiplying it with any vector/matrix ')
                doc.append('will always give a zero-matrix. ')
                doc.append('We will still pick a ')
                doc.append(bold('basis vector'))
                doc.append(Math(
                    data=[
                        r'I_{', data.n, r'} = ',
                        Matrix(data.row, mtype='b'),
                        #bNiceMatrix(
                            #data.mat, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                        #)
                    ], 
                    inline=False, 
                    escape=False,
                ))
                doc.append('to match with the column ')
                write_col(doc, data)
                doc.append('\nSubstracting ')
                doc.append(Math(
                    data=[
                        r'C_{', data.n + 1 ,r'} \cdot I_{', data.mat_idx, r'} \ ',
                    ], 
                    inline=True, 
                    escape=False,
                ))
                doc.append(' from ')
                doc.append(Math(
                    data=[
                        r'A_{', data.mat_idx, r'} \ ',
                    ], 
                    inline=True, 
                    escape=False,
                ))
                doc.append(' does nothing. So we increment the column but keep the same row.')
                return

            case State.ZERO_ROW:
                doc.append('We have the matrix')
                doc.append(Math(
                    data=[
                        r'A_{', data.mat_idx, r'} = ',
                        Matrix(data.mat, mtype='b'),
                        #bNiceMatrix(
                            #data.mat, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                        #)
                    ], 
                    inline=False, 
                    escape=False,
                ))
                doc.append('We can see that the row ')
                write_row(doc, data)
                doc.append('is a ')
                doc.append(bold('zero vector'))
                doc.append(', meaning validly multiplying it with any vector/matrix ')
                doc.append('will always give a zero-matrix.')
                doc.append('\nTherefore, we will keep the column but increment the row.')
                return

            case State.ZERO_EL:
                doc.append('We have the matrix')
                doc.append(Math(
                    data=[
                        r'A_{', data.mat_idx, r'} = ',
                        Matrix(data.mat, mtype='b'),
                        #bNiceMatrix(
                            #data.mat, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                        #)
                    ], 
                    inline=False, 
                    escape=False,
                ))
                doc.append('Though either the row ')
                write_row(doc, data)
                doc.append('or the column ')
                write_col(doc, data)
                doc.append('is a ')
                doc.append(bold('non-zero vector'))
                doc.append(', we can see that the element ')
                write_element(doc, data)
                doc.append('\nThis means that the row ')
                write_col(doc, data)
                doc.append(' is a linear combination of some other columns.')
                doc.append('To reduce the rank, we have to switch column ')
                doc.append(Math(data=[data.permutation.a + 1, r' \ '], inline=True, escape=False))
                doc.append(' with column ')
                doc.append(Math(data=[data.permutation.b + 1, r' \ '], inline=True, escape=False))
                doc.append('.\nTo do that, we have to multiply the matrix with a permutation matrix')
                doc.append(Math(
                    data=[
                        r'A_{', data.mat_idx, r'} = ',
                        Matrix(data.mat, mtype='b'),
                        #bNiceMatrix(
                            #data.mat, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                        #)
                        r'\cdot ',
                        Matrix(data.permutation.matrix, mtype='b'),
                        #bNiceMatrix(
                            #data.permutation.matrix, color='red!15',
                            #rows=str(data.m + 1),
                            #cols=str(data.n + 1),
                        #)
                        r' = ',
                        Matrix(np.matmul(data.mat, data.permutation.matrix), mtype='b'),
                    ], 
                    inline=False, 
                    escape=False,
                ))
                doc.append('Then, we continue with the same row and column, but with the new matrix.')
                return

            case _:
                doc.append('\nSomething went wrong?')
                return

    # Write
    for step in decom.steps:
        with doc.create(Subsection('Step ' + str(step.number))):
            write_step_state(doc, step)
    return


# Helper functions
def format(x): # Replace all -0.0 with 0.0
    if x == 0.0:
        return 0.0

    if abs(x) < 1e-16:
        x = 0.0
    return x

def vectorized_format(x):
    return np.vectorize(format)(x)


# Main
def write_pdf(matrix: np.matrix):
    pre = Decomposition()
    # Maybe these clear()s fixed the bug? (repeating data of previous compositions)
    pre.steps.clear()
    pre.rows.clear()
    pre.cols.clear()
    pre.indices.clear()
    pre.permutations.clear()

    pre.steps.append(Step(
        0, State.START,
        Data(None, None, None, None, None, None, None, None, None, 0, 0)
    ))
    pre = decom(matrix, 0, 0, pre)
    pre.original_matrix = matrix

    # Post processing - 
    for i, col in enumerate(pre.cols):
        if not col.any(): # Column is zero
            temp = len(pre.cols) - 1
            while temp > i:
                if not pre.cols[temp].any():
                    temp -= 1
                else:
                    break

            if temp == i:
                continue
            else:
                num_of_col = pre.original_matrix.shape[1]
                permutation = np.asmatrix(np.zeros((num_of_col, num_of_col), dtype=np.float_), dtype=np.float_)
                for j in range(num_of_col):
                    if j == i:
                        permutation[i, temp] = 1.0
                        continue
                    if j == temp:
                        permutation[temp, i] = 1.0
                        continue
                    permutation[j, j] = 1.0
                permutation_object = Permutation(permutation, i, temp)
                pre.permutations.append(permutation_object)
        else:
            continue


    decomposition = Decomposition()
    if len(pre.permutations) == 0:
        decomposition = pre
    else:
        decomposition.steps.clear()
        decomposition.rows.clear()
        decomposition.cols.clear()
        decomposition.indices.clear()
        decomposition.permutations.clear()
        # Get matrix P
        num_of_col = pre.original_matrix.shape[1]
        decomposition.P = np.asmatrix(np.eye(num_of_col, num_of_col, dtype=np.float_), dtype=np.float_)
        for permutation in decomposition.permutations:
            decomposition.P = np.matmul(decomposition.P, permutation.matrix)
        
        AP = np.matmul(matrix, decomposition.P)
        print(f'{AP=}')

        decomposition.steps.append(Step(
            0, State.START,
            Data(None, None, None, None, None, None, None, None, None, 0, 0)
        ))
        decomposition = decom(AP, 0, 0, decomposition)
        decomposition.original_matrix = matrix


    # A = L
    ## Get matrix L
    cols_as_arrays = [
        col.A1 for col in decomposition.cols
    ]
    decomposition.L = np.matrix(cols_as_arrays).T # Tranpose
    ## Get matrix U
    rows_as_arrays = [
        row.A1 for row in decomposition.rows
    ]
    decomposition.U = np.matrix(rows_as_arrays)

    decomposition.L = vectorized_format(decomposition.L)
    decomposition.U = vectorized_format(decomposition.U)
    for step in decomposition.steps:
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
        #step.view()


    #########################################
    # Set up LaTex Document 
    doc = Document()
    doc.documentclass = Command(
        'documentclass',
        options=['preview=true', NoEscape(r'border={10pt 10pt 300pt 10pt}')],
        arguments=['standalone'],
    )
    # Incompatible with Streamlit
    #doc.packages.append(Package('nicematrix')) 

    # Write answer
    with doc.create(Section('Answer')):
        write_answer(doc, decomposition)
    
    # Write details
    with doc.create(Section('Detailed Solution')):
        write_detailed_solution(doc, decomposition)

    # Generate PDF
    doc.generate_pdf('decomposition', clean_tex=True, clean=True, compiler='latexmk')
    return