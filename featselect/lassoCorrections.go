package featselect

import "math"

// LassoCorrection is an interface to types that can be added to the
// lasso solver
type LassoCorrection interface {
	// Deriv returns the derivative with respect to beta
	Deriv(beta []float64, featNo int) float64

	// Update is a callback that is called every iteration
	Update(beta []float64)
}

// PureLasso is a type that does not alter the original lasso method
type PureLasso struct{}

// Deriv returns 0.0
func (p *PureLasso) Deriv(bet []float64, featNo int) float64 {
	return 0.0
}

// Update does not do anything
func (p *PureLasso) Update(beta []float64) {}

// CLasso implements the LassoCorrection interface and tends to
// promote selection of groups of correlated feature
type CLasso struct {
	beta []float64
	eta  float64
}

// NewCLasso returns a pointer to a new instance of CLasso
func NewCLasso(numFeat int, eta float64) *CLasso {
	var cl CLasso
	cl.eta = eta
	cl.beta = make([]float64, numFeat)
	return &cl
}

// Update updates the feature covariance matrix
func (cl *CLasso) Update(beta []float64) {
	copy(cl.beta, beta)
}

// Deriv calculates derivative of the correction with respect to the
// new beta values
func (cl *CLasso) Deriv(beta []float64, featNo int) float64 {
	normSq := 0.0
	inner := 0.0
	for i := range cl.beta {
		normSq += cl.beta[i] * cl.beta[i]
		inner += cl.beta[i] * beta[i]
	}

	if math.Abs(normSq) < 1e-10 {
		return 0.0
	}
	return cl.eta * cl.beta[featNo] * inner / normSq
}
