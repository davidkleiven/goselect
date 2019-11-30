package featselect

import (
	"gonum.org/v1/gonum/mat"
)

// IndexedColView implements the Matrix interface
type IndexedColView struct {
	Matrix     mat.Matrix
	ColIndices []int
}

// At returns the value of the element at row i and column j of the indexed
// matrix, that is, RowIndices[i] and ColumnIndices[j] of the Matrix field.
func (v *IndexedColView) At(i, j int) float64 {
	return v.Matrix.At(i, v.ColIndices[j])
}

// Dims returns the dimensions of the indexed matrix and the number of columns.
func (v *IndexedColView) Dims() (int, int) {
	nr, _ := v.Matrix.Dims()
	return nr, len(v.ColIndices)
}

// T returns an implicit transpose of the matrix field
func (v *IndexedColView) T() mat.Matrix {
	return mat.Transpose{v}
}

// NewIndexedColView returns a view of the matrix containing only the rows specified
// by rowInd and columns given by colInd. If rowInd/colInd is nil, all rows/columns
// are included. Thus, NewIndexedView(m, nil, nil) is the same as the original matrix
func NewIndexedColView(m mat.Matrix, colInd []int) IndexedColView {
	return IndexedColView{m, colInd}
}

// IndexedColVecView is a view for column vectors
type IndexedColVecView struct {
	Vector mat.Vector
	Rows   []int
}

// NewIndexedColVecView creates a new column vector view
func NewIndexedColVecView(v mat.Vector, rows []int) *IndexedColVecView {
	var view IndexedColVecView
	view.Vector = v
	view.Rows = rows
	return &view
}

// At returns the (i, 0) element
func (v *IndexedColVecView) At(i, j int) float64 {
	return v.Vector.AtVec(v.Rows[i])
}

// AtVec returns the element at
func (v *IndexedColVecView) AtVec(i int) float64 {
	return v.At(i, 0)
}

// Dims returns the dimensions
func (v *IndexedColVecView) Dims() (int, int) {
	return len(v.Rows), 1
}

// T returns an implicit transpose of the matrix field
func (v *IndexedColVecView) T() mat.Matrix {
	return mat.Transpose{v}
}

// Len returns the length of the vector
func (v *IndexedColVecView) Len() int {
	return len(v.Rows)
}
