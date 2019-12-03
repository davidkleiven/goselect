package featselect

import "gonum.org/v1/gonum/mat"

// NormalizedData is a structure that is used to normalise
// columns to zero mean and unit variance
type NormalizedData struct {
	X       *mat.Dense
	y       []float64
	std     []float64
	mu      []float64
	stdY    float64
	muY     float64
	HasBias bool
}

// NewNormalizedData initializes a new structure with normalised data
// Note that both X and y will be altered by this method.
func NewNormalizedData(X *mat.Dense, y []float64) *NormalizedData {
	var normD NormalizedData
	normD.stdY = Std(y)
	normD.muY = Mean(y)
	normD.X = X
	normD.y = y

	for i := 0; i < len(y); i++ {
		y[i] = (y[i] - normD.muY) / normD.stdY
	}

	nr, nc := X.Dims()
	tmp := make([]float64, nr)

	normD.mu = make([]float64, nc)
	normD.std = make([]float64, nc)

	for c := 0; c < nc; c++ {
		for r := 0; r < nr; r++ {
			tmp[r] = X.At(r, c)
		}

		normD.std[c] = Std(tmp)
		normD.mu[c] = Mean(tmp)

		if normD.std[c] < 1e-10 && c != 0 {
			panic("normdata: Only the first column can be a constant!")
		}

		stdtmp := normD.std[c]
		if stdtmp < 1e-10 {
			normD.HasBias = true
			stdtmp = 1.0
		}

		for r := 0; r < nr; r++ {
			X.Set(r, c, (tmp[r]-normD.mu[c])/stdtmp)
		}
	}
	return &normD
}

// LinearNormalizationTransformation computes the difference in coefficient in the expansion
// y1 = c_0 + c_1*x_1 + c_2*x_2+ ... and
// y2 = c_0' + c_1'*x_1' + c_2'*x_2', where primed x and y are normalized values
func (n *NormalizedData) LinearNormalizationTransformation(idx int, value float64) float64 {
	if idx == 0 && n.HasBias {
		panic("normdata: Does not work for the linear bias term")
	}
	return value * n.stdY / n.std[idx]
}

// LinearTransformationBias calculates the bias coefficient
func (n *NormalizedData) LinearTransformationBias(selected []int, coeff []float64) float64 {
	sumShift := 0.0
	for i, v := range selected {
		sumShift += coeff[i] * n.mu[v] / n.std[v]
	}
	return n.muY - n.stdY*sumShift
}
