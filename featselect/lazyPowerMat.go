package featselect

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// LazyPowerMatrix is a matrix that can have arbitrary additional columns
// formed by taking powers of the existing columns
type LazyPowerMatrix struct {
	X      *mat.Dense
	powers []map[int]int
}

func (m *LazyPowerMatrix) numOrigCols() int {
	_, ncols := m.X.Dims()
	return ncols
}

// NumCols return the total number of columns in the matrix
func (m *LazyPowerMatrix) NumCols() int {
	orig := m.numOrigCols()
	return orig + len(m.powers)
}

func (m *LazyPowerMatrix) numRows() int {
	nrows, _ := m.X.Dims()
	return nrows
}

// GetCol returns a column of the matrix
func (m *LazyPowerMatrix) GetCol(col int) []float64 {
	res := make([]float64, m.numRows())
	if col < m.numOrigCols() {
		v := m.X.ColView(col)
		for i := 0; i < v.Len(); i++ {
			res[i] = v.AtVec(i)
		}
	} else {
		for row := 0; row < m.numRows(); row++ {
			res[row] = 1.0
			for col, power := range m.powers[col-m.numOrigCols()] {
				res[row] *= math.Pow(m.X.At(row, col), float64(power))
			}
		}
	}
	return res
}

// Add a set of power to the matrix
func (m *LazyPowerMatrix) AddPower(power map[int]int) {
	m.powers = append(m.powers, power)
}

// NewLazyMatrix creates a new matrix with all powers up to maxPower
// if maxPower is 0, it is equivalent to the original matrix X
func NewLazyMatrix(X *mat.Dense, maxPower int) *LazyPowerMatrix {
	var lazy LazyPowerMatrix
	lazy.X = X
	_, ncols := X.Dims()
	cols := make([]int, ncols)
	for i := 0; i < len(cols); i++ {
		cols[i] = i
	}
	lazy.AddPowerSequence(cols, maxPower)
	return &lazy
}

// AddPowerSequence adds all powers and cross-terms of the columns listed in
// cols to the matrix
func (m *LazyPowerMatrix) AddPowerSequence(cols []int, maxPower int) {
	powers := make([]int, maxPower+1)
	for p := 0; p < maxPower+1; p++ {
		powers[p] = p
	}

	for _, pow := range IterProduct(powers, len(cols)) {
		if Sum(pow) <= 1 || Sum(pow) > maxPower {
			continue
		}
		p := make(map[int]int)
		for i := 0; i < len(cols); i++ {
			if pow[i] > 0 {
				p[cols[i]] = pow[i]
			}
		}
		m.AddPower(p)
	}
}

func (m *LazyPowerMatrix) FullMatrix() *mat.Dense {
	X := mat.NewDense(m.numRows(), m.NumCols(), nil)
	for c := 0; c < m.NumCols(); c++ {
		X.SetCol(c, m.GetCol(c))
	}
	return X
}
