import numpy as np
from algo import *

from pylatex.base_classes import Environment, Arguments, CommandBase
from pylatex.base_classes.containers import Container
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


def write_result_CR(doc: Document, decom: Decomposition, oneline: bool):
    doc.append('The matrix can be decomposed into the following combination:')

    # Write A = CR
    with doc.create(Alignat(numbering=False, escape=False)) as agn:
        agn.append(Matrix(decom.original_matrix, mtype='b'))
        agn.append(r' &= ')

        for i in range(len(decom.rows)):
            agn.append(Matrix(decom.cols[i], mtype='b'))
            agn.append('\cdot')
            agn.append(Matrix(decom.rows[i], mtype='b'))
            if i != len(decom.rows) - 1:
                if oneline:
                    agn.append(r'\\ &+ ')
                else:
                    agn.append(r' + ')

    # Write short answer
    short = 'A = '
    for i in range(len(decom.rows)):
        short += 'C_{'
        short += str(decom.indices[i][1] + 1)
        short += '}R_{'
        short += str(decom.indices[i][0] + 1)
        short += '}'
        if i != len(decom.rows) - 1:
            short += ' + '

    doc.append('This means that:')
    doc.append(Math(data=[short], inline=False, escape=False))
    doc.append('Look at the detailed solution to see where ')
    doc.append(Math(data=[r'R_{i}\ '], inline=True, escape=False))
    doc.append('and ')
    doc.append(Math(data=[r'C_{j}\ '], inline=True, escape=False))
    doc.append('come from.')
    return


def write_result_LU(doc: Document, decom: Decomposition):
    doc.append('The matrix can also be decomposed into two matrices ')
    doc.append(Math(data=[r'A = LU'], inline=True, escape=False))

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
    doc.append('We will now go into the details of ')
    doc.append(Math(data=[r'A = \sum_{}^{}C_{i}R_{j}'], inline=True, escape=False))

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

            case State.ZERO_ROW_COL:
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
                doc.append(bold('zero vectors'))
                doc.append(', meaning they are linearly dependent and choosing them will do nothing.')
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
                doc.append('will always give a zero-matrix.')
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
                doc.append('\nThis means ... ')
                doc.append('and they cannot be picked.')
                return

            case State.TEMPORARY_SKIPPING_COL:
                doc.append('We have found a non-zero element, which is ')
                write_element(doc, data)
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
                doc.append('Now the row ')
                write_row(doc, data)
                doc.append('and the column ')
                write_col(doc, data)
                doc.append('are both ')
                doc.append(bold('linearly independent '))
                doc.append(r' and thus we can pick them out.')
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

            case _:
                doc.append('\nSomething went wrong?')
                return

    def write_step_action(doc: Document, step: Step):
        data = step.data
        match step.action:
            case Action.START:
                return

            case Action.END:
                doc.append('\nBravo! We have successfully decomposed the matrix.')
                return

            case Action.INCREMENT_ROW_COL:
                doc.append('\nWe then increment our row and column indices and continue searching.')
                return

            case Action.INCREMENT_ROW:
                doc.append('\nTherefore, we increment our row index while keeping the same column.')
                return

            case Action.INCREMENT_COL:
                doc.append('\nThus, we increment our column index but keep the same row.')
                return

            case Action.FIND_NONZERO_EL:
                doc.append('\nIn this case, we increment our column index while keeping the same row ')
                doc.append('until we find a non-zero element. (We will definitely find it because ')
                doc.append('otherwise it will be a zero-vector row case.)')
                return

            case Action.INCREMENT_ROW_AND_RECALL_COL:
                doc.append('\nHowever, now we have to go back to the old column index ')
                doc.append('while incrementing our row, in search for the next combination.')
                return

            case _:
                doc.append('\nSomething went wrong?')
                return

    for step in decom.steps:
        with doc.create(Subsection('Step ' + str(step.number))):
            write_step_state(doc, step)
            write_step_action(doc, step)
    return


# Helper functions
def format(x): # Replace all -0.0 with 0.0
    if x == 0.0:
        return 0.0

    #float_precision = 2 # Set = -1 to ignore rounding
    #if float_precision != -1:
        #return np.round(x, float_precision)
    return x

def vectorized_format(x):
    return np.vectorize(format)(x)


# Main
def write_pdf(matrix: np.matrix):
    decomposition = Decomposition()
    # Maybe these clear()s fixed the bug? (repeating data of previous compositions)
    decomposition.steps.clear()
    decomposition.rows.clear()
    decomposition.cols.clear()
    decomposition.indices.clear()

    decomposition.steps.append(Step(
        0, State.START, Action.START,
        Data(None, None, None, None, None, None, None, None, 0, 0)
    ))

    decomposition = decom(matrix, 0, 0, 0, False, decomposition)
    decomposition.original_matrix = matrix

    # This block is incompatible with Streamlit environment for some reason
    ## A = L
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

    # Format every matrix in the Decomposition
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
        with doc.create(Subsection('Solution')):
            write_result_CR(doc, decomposition, False)
        with doc.create(Subsection('Solution 2')):
            write_result_LU(doc, decomposition)
    
    # Write details
    with doc.create(Section('Detailed Solution')):
        write_detailed_solution(doc, decomposition)

    # Generate PDF
    doc.generate_pdf('decomposition', clean_tex=True, clean=True, compiler='latexmk')
    return