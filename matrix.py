import random
import pandas as pd


class Matrix:

    def __init__(self, rows, cols, randomized=True):
        self.rows = rows
        self.cols = cols
        self.data = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        if randomized:
            self.generate()

    @classmethod
    def from_list(cls, l):
        result = cls(len(l), 1, False)
        for i in range(len(l)):
            result.data[i][0] = l[i]
        return result

    @staticmethod
    def dot(a, b):
        # A . B
        if type(a) is not Matrix:
            raise AttributeError("A must be a matrix.")

        if type(b) is not Matrix or a.cols != b.rows:
            raise AttributeError("Not a matrix or wrong shape match")
        result = Matrix(a.rows, b.cols, False)

        for i in range(result.rows):
            for j in range(result.cols):
                tot = 0
                for k in range(a.cols):
                    tot += a.data[i][k] * b.data[k][j]
                result.data[i][j] = tot
        return result

    @staticmethod
    def transpose(a):
        result = Matrix(a.cols, a.rows, False)
        for i in range(a.rows):
            for j in range(a.cols):
                result.data[j][i] = a.data[i][j]
        return result

    @staticmethod
    def subtract(a, b):
        if a.cols != b.cols or a.rows != b.rows:
            raise AttributeError("Shape must match!")
        result = Matrix(a.rows, a.cols, False)
        for i in range(a.rows):
            for j in range(a.cols):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        return result

    @staticmethod
    def mmap(m, fn):
        result = Matrix(m.rows, m.cols, False)
        for i in range(m.rows):
            for j in range(m.cols):
                result.data[i][j] = fn(m.data[i][j])
        return result

    def generate(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = random.uniform(-1, 1)

    def add(self, n):

        if type(n) is Matrix:
            if self.rows != n.rows or self.cols != n.cols:
                raise ValueError("Different shape!")

            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]
            return

        if type(n) is not int:
            raise AttributeError('Int or matrix is needed to be added.')

        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] += n

    def scale(self, n):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] *= n

    def multiply(self, m):
        if type(m) is not Matrix or m.rows != self.rows or m.cols != self.cols:
            raise AttributeError("Shape must match!")

        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = self.data[i][j] * m.data[i][j]

    def T(self):
        data = [[0 for _ in range(self.rows)] for _ in range(self.cols)]
        for i in range(self.rows):
            for j in range(self.cols):
                data[j][i] = self.data[i][j]
        self.cols, self.rows = self.rows, self.cols
        self.data = data

    def map(self, fn):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = fn(self.data[i][j])

    def to_list(self):
        result = []
        for row in self.data:
            for d in row:
                result.append(d)
        return result

    def __str__(self):
        df = pd.DataFrame(data=self.data)
        result = str(df)
        return result

    def print(self):
        df = pd.DataFrame(data=self.data)
        print(str(df))

def main():
    m1 = Matrix(2, 3)
    print(m1)
    m2 = Matrix(3, 4)
    print(m2)
    m3 = Matrix.dot(m1, m2)
    print(m3)
    m3.T()
    print(m3)
    m3.map(lambda x: x*10)
    print(m3)

    l = [1, 2, 0 , -9]
    m4 = Matrix.from_list(l)
    print(m4)

    m4.print()

if __name__ == '__main__':
    main()