package featselect

import "gonum.org/v1/gonum/mat"

// CDParam interface provides a generic interface to calculate the c and d paremeter
// in the LASSO algorithm defined
// Tibshirani, R.J., 2013. The lasso problem and uniqueness. Electronic Journal of Statistics, 7, pp.1456-1490.
// In short C = (X^TX)^{-1}X^Ty and D = (X^TX)^{-1}s, where s is the sign vector. The column in X should only
// corresponds to the columns in the active set.
type CDParam interface {
	// C calculates the C parameter. The length of the return value is
	// equal to the number of items in the active set
	C(y mat.Vector) *mat.VecDense

	// D calculates the D parameter. The length of the return value is
	// equal to the number of items in the active set. The sign parameter
	// is a view of the signs in the active view. The corresponding column
	// in the full design matrix is given by activeSet
	D(signs mat.Vector) *mat.VecDense

	// SetActiveSet sets a new value for the active set
	SetActiveSet(active []int)

	// Set X sets the full design matrix
	SetX(X mat.Matrix)
}

// MorsePenroseCD implements the CDParem interface and will
// lead to the exact same algorithm as described in
// Tibshirani, R.J., 2013. The lasso problem and uniqueness. Electronic Journal of Statistics, 7, pp.1456-1490.
type MorsePenroseCD struct {
	X         mat.Matrix
	active    []int
	svd       mat.SVD
	invDesign *mat.Dense
}

// C calculates the c-parameter in Tibshirani 2013
func (m *MorsePenroseCD) C(y mat.Vector) *mat.VecDense {
	s := m.svd.Values(nil)
	var v mat.Dense
	var u mat.Dense
	m.svd.VTo(&v)
	m.svd.UTo(&u)

	for i := 0; i < len(s); i++ {
		if s[i]*s[i] > 1e-6 {
			s[i] = 1.0 / s[i]
		}
	}

	diag := mat.NewDiagDense(len(s), s)
	nr, _ := v.Dims()
	cMat := mat.NewDense(nr, 1, nil)
	cMat.Product(&v, diag, u.T(), y)
	res := mat.NewVecDense(nr, nil)
	for i := 0; i < nr; i++ {
		res.SetVec(i, cMat.At(i, 0))
	}
	return res
}

// D calculates the d-parameter in Tibshirani 2013
func (m *MorsePenroseCD) D(signs mat.Vector) *mat.VecDense {
	nr, _ := m.invDesign.Dims()
	res := mat.NewVecDense(nr, nil)
	res.MulVec(m.invDesign, signs)
	return res
}

// SetX sets the full design matrix
func (m *MorsePenroseCD) SetX(X mat.Matrix) {
	m.X = X
}

// SetActiveSet sets the active set
func (m *MorsePenroseCD) SetActiveSet(active []int) {
	m.active = active

	Xe := NewIndexedColView(m.X, m.active)
	m.svd.Factorize(&Xe, mat.SVDThin)
	m.invDesign = PseudoInverse(&m.svd, 1e-6)
}
