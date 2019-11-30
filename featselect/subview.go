package featselect

import "gonum.org/v1/gonum/mat"

// It implements the Matrix interface
type IndexedView struct {
	Matrix     mat.Matrix
	RowIndices []int
	ColIndices []int
}

// At returns the value of the element at row i and column j of the indexed
// matrix, that is, RowIndices[i] and ColumnIndices[j] of the Matrix field.
func (v *IndexedView) At(i, j int) float64 {
	return v.Matrix.At(v.RowIndices[i], v.ColIndices[j])
}

// Dims returns the dimensions of the indexed matrix and the number of columns.
func (v *IndexedView) Dims() (int, int) {
	return len(v.RowIndices), len(v.ColIndices)
}

// T returns an implicit transpose of the matrix field
func (v *IndexedView) T() mat.Matrix {
	return mat.Transpose{v}
}

// NewIndexedView returns a view of the matrix containing only the rows specified
// by rowInd and columns given by colInd. If rowInd/colInd is nil, all rows/columns
// are included. Thus, NewIndexedView(m, nil, nil) is the same as the original matrix
func NewIndexedView(m mat.Matrix, rowInd, colInd []int) IndexedView {
	nr, nc := m.Dims()

	if rowInd == nil {
		rowInd = make([]int, nr)
		for i := 0; i < nr; i++ {
			rowInd[i] = i
		}
	} else {
		for _, v := range rowInd {
			if v < 0 || v >= nr {
				panic("Row index out of bounds")
			}
		}
	}

	if colInd == nil {
		colInd = make([]int, nc)
		for i := 0; i < nc; i++ {
			colInd[i] = i
		}
	} else {
		for _, v := range colInd {
			if v < 0 || v >= nc {
				panic("Column index out of bounds")
			}
		}
	}

	return IndexedView{m, rowInd, colInd}
}
