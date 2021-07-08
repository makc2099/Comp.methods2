import numpy as np
from tabulate import tabulate


def spectral(matrix):
    matrix = np.matrix(matrix)
    return np.linalg.norm(matrix) * np.linalg.norm(matrix.I)


def square_root(matrix, b):
    num_rows, num_cols = np.shape(matrix)
    L = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        sum = 0
        for k in range(2):
            sum += L[k][i]*L[k][i]
        L[i][i] = np.sqrt(matrix[i][i]-sum)
        for j in range(i+1, num_rows):
            sum = 0
            for k in range(i):
                sum += L[k][i]*L[k][j]
            L[i][j]=(matrix[i][j]-sum)/L[i][i]
    y = np.linalg.solve(L.T, b)
    x = np.linalg.solve(L, y)
    return x


def main():
    A = np.array([[1, 1 / 2, 1 / 3, 1/4,1/5],
                  [1 / 2, 1 / 3, 1 / 4, 1/5,1/6],
                  [1 / 3, 1 / 4, 1 / 5, 1/6,1/7],
                  [1/4,1/5,1/6,1/7,1/8],
                  [1/5,1/6,1/7,1/8,1/9]])
    b = np.array([[1, 1 / 2, 1 / 3, 1/4,1/5],
                  [1 / 2, 1 / 3, 1 / 4, 1/5,1/6],
                  [1 / 3, 1 / 4, 1 / 5, 1/6,1/7],
                  [1/4,1/5,1/6,1/7,1/8],
                  [1/5,1/6,1/7,1/8,1/9]])
    print('Матрица A:')
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in A]))
    num_rows, num_cols = np.shape(A)
    print(tabulate([[10**(-2),spectral(A),spectral(A + np.identity(num_rows)*10**(-2)),np.linalg.norm(square_root(A,b) - square_root(A + np.identity(num_rows)*10**(-2),b))],
                    [10**(-3),spectral(A),spectral(A + np.identity(num_rows)*10**(-3)),np.linalg.norm(square_root(A,b) - square_root(A + np.identity(num_rows)*10**(-3),b))],
                    [10**(-5),spectral(A),spectral(A + np.identity(num_rows)*10**(-5)),np.linalg.norm(square_root(A,b) - square_root(A + np.identity(num_rows)*10**(-5),b))],
                    [10**(-7),spectral(A),spectral(A + np.identity(num_rows)*10**(-7)),np.linalg.norm(square_root(A,b) - square_root(A + np.identity(num_rows)*10**(-7),b))],
                    [10**(-9),spectral(A),spectral(A + np.identity(num_rows)*10**(-9)),np.linalg.norm(square_root(A,b) - square_root(A + np.identity(num_rows)*10**(-9),b))],
                    [10**(-12),spectral(A),spectral(A + np.identity(num_rows)*10**(-12)),np.linalg.norm(square_root(A,b) - square_root(A + np.identity(num_rows)*10**(-12),b))]],headers=['alpha','cond(A)','cond(A + alpha*E)','||x - x_a||'],tablefmt='orgtbl'))

    print("Выбираем alpha = 10^(-12)")
    alpha = 10**(-12)
    print("||x - x_a||:")
    print(tabulate([[np.linalg.norm(square_root(A,b) - square_root(A + np.identity(num_rows)*alpha,b)),
                     np.linalg.norm(square_root(A+np.identity(num_rows)*alpha,b) - square_root(A + 2*np.identity(num_rows)*alpha,b)),
                     np.linalg.norm(square_root(A+10*np.identity(num_rows)*alpha,b) - square_root(A + 11*np.identity(num_rows)*alpha,b)),
                     np.linalg.norm(square_root(A+ 0.1*alpha*np.identity(num_rows),b) - square_root(A + 0.1*np.identity(num_rows)*alpha+np.identity(num_rows)*alpha,b))]],headers=['Ax = b','(A + alpha*E)x = B','(A + 10*alpha*E)x = B','(A + 0.1*alpha*E)x = B'],tablefmt='orgtbl'))
if __name__ == '__main__':
    main()
