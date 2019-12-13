package featselect

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

const lassoCrdDescZero = 1e-16

// CorrectableLasso is a function that implements the lasso method with corrections
type CorrectableLasso = func(dset *NormalizedData, lamb float64, cov CovMat, x0 []float64, maxIter int, tol float64, corr LassoCorrection) []float64

// LassoCrdDesc solves the lasso problem via coordinate descent
func LassoCrdDesc(dset *NormalizedData, lamb float64, cov CovMat, x0 []float64, maxIter int, tol float64, corr LassoCorrection) []float64 {
	nr, nFeat := dset.X.Dims()
	if x0 == nil {
		x0 = make([]float64, nFeat)
	}

	covMat := cov.Get(dset.X)

	// Precalcuations
	yVec := mat.NewVecDense(len(dset.y), dset.y)
	XTy := mat.NewVecDense(nFeat, nil)
	XTy.MulVec(dset.X.T(), yVec)

	iterIndices := make([]int, nFeat-1)
	for i := 1; i < nFeat; i++ {
		iterIndices[i-1] = i
	}

	betaOld := make([]float64, nFeat)
	copy(x0, betaOld)
	beta := make([]float64, nFeat)
	covDotBeta := MulSlice(covMat, betaOld)
	converged := false
	for iter := 0; iter < maxIter; iter++ {
		for _, j := range iterIndices {
			covDiag := covMat.At(j, j)
			oldCoeff := betaOld[j]
			covDotBetaNoDiag := covDotBeta[j] - betaOld[j]*covDiag
			newCoeff := XTy.AtVec(j)/float64(nr) - covDotBetaNoDiag - corr.Deriv(betaOld, j)
			newCoeff = SoftThreshold(newCoeff, lamb) / covDiag

			UpdateCovDotBeta(covMat, covDotBeta, j, oldCoeff, newCoeff)
			beta[j] = newCoeff
		}
		corr.Update(beta)

		maxChange := 0.0
		maxCoeff := 0.0
		for _, j := range iterIndices {
			diff := math.Abs(beta[j] - betaOld[j])
			if diff > maxChange {
				maxChange = diff
			}

			if math.Abs(beta[j]) > maxCoeff {
				maxCoeff = math.Abs(beta[j])
			}
		}

		converged = maxChange/maxCoeff < tol

		numActive := 0
		for _, j := range iterIndices {
			if math.Abs(beta[j]) > lassoCrdDescZero {
				iterIndices[numActive] = j
				numActive++
			}
		}
		iterIndices = iterIndices[:numActive]

		copy(betaOld, beta)

		if converged {
			unsatisfied := UnsatisfiedKKTConditions(XTy, covDotBeta, beta, lamb, corr)
			if len(unsatisfied) == 0 {
				break
			} else {
				iterIndices = UnionInt(iterIndices, unsatisfied)
			}
		}
	}

	if !converged {
		fmt.Printf("Warning! Lasso coordinate descent did not converge within the given number of iterations\n")
	}
	return beta
}

// SoftThreshold applyes a soft threshold to the value
func SoftThreshold(x float64, threshold float64) float64 {
	if x < -threshold {
		return x + threshold
	} else if x > threshold {
		return x - threshold
	}
	return 0.0
}

// UnsatisfiedKKTConditions returns the indices where KKT conditions are not met
func UnsatisfiedKKTConditions(Xy mat.Vector, covDotBeta []float64, coeff []float64, lamb float64, correction LassoCorrection) []int {
	unsatisfied := []int{}
	nr := Xy.Len()
	for i := 1; i < len(covDotBeta); i++ {
		value := math.Abs(Xy.AtVec(i) - covDotBeta[i]*float64(nr) - correction.Deriv(coeff, i))
		if value < lamb && math.Abs(coeff[i]) > lassoCrdDescZero {
			unsatisfied = append(unsatisfied, i)
		}
	}
	return unsatisfied
}

// MulSlice multiplies together a matrix and a vector
func MulSlice(X mat.Matrix, v []float64) []float64 {
	nr, nc := X.Dims()
	if len(v) != nc {
		panic("Inconsistent length of passed vector")
	}

	res := make([]float64, nr)
	for i := 0; i < nr; i++ {
		for j := 0; j < nc; j++ {
			res[i] += X.At(i, j) * v[j]
		}
	}
	return res
}

// UpdateCovDotBeta updates dot product between a matrix and an vector when one item changes
func UpdateCovDotBeta(cov mat.Matrix, covDotBeta []float64, coeffNo int, oldCoeff float64, newCoeff float64) []float64 {
	for i := 0; i < len(covDotBeta); i++ {
		covDotBeta[i] += cov.At(i, coeffNo) * (newCoeff - oldCoeff)
	}
	return covDotBeta
}

// LassoCrdWorkload is a struct holder information to carry out a lasso coordinate descent path
type LassoCrdWorkload struct {
	x0      []float64
	lamb    float64
	dset    *NormalizedData
	cov     CovMat
	maxIter int
	lambIdx int
	tol     float64
	corr    LassoCorrection
}

// LassoRes is a structure used to return the result
type LassoRes struct {
	node    *LassoLarsNode
	lambIdx int
}

// PerformLassoCrd listens to the workload channel and passes its result to res
func PerformLassoCrd(workload <-chan LassoCrdWorkload, res chan<- LassoRes) {
	for wrk := range workload {
		coeff := LassoCrdDesc(wrk.dset, wrk.lamb, wrk.cov, wrk.x0, wrk.maxIter, wrk.tol, wrk.corr)
		selection := []int{}
		selectedCoeff := []float64{}
		for j := range coeff {
			if math.Abs(coeff[j]) > lassoCrdDescZero {
				selection = append(selection, j)
				selectedCoeff = append(selectedCoeff, coeff[j])
			}
		}
		node := NewLassoLarsNode(selectedCoeff, wrk.lamb, selection)
		var resStruct LassoRes
		resStruct.node = node
		resStruct.lambIdx = wrk.lambIdx
		res <- resStruct
	}
}

// LassoCrdDescPath calculates a set of lasso solutions along equi-logspaced set of lambda values
func LassoCrdDescPath(dset *NormalizedData, cov CovMat, lambs []float64, maxIter int, tol float64, correction LassoCorrection) []*LassoLarsNode {
	_, nFeat := dset.X.Dims()
	x0 := make([]float64, nFeat)

	nodes := make([]*LassoLarsNode, len(lambs))
	availableLambs := make([]int, len(lambs))
	for i := 0; i < len(availableLambs); i++ {
		availableLambs[i] = len(lambs) - i - 1
	}

	numWorkers := 10

	if len(lambs) < numWorkers {
		numWorkers = len(lambs)
	}

	workChan := make(chan LassoCrdWorkload)
	resChan := make(chan LassoRes)
	for i := 0; i < numWorkers; i++ {
		go PerformLassoCrd(workChan, resChan)

		var wrk LassoCrdWorkload
		wrk.x0 = x0
		wrk.lamb = lambs[availableLambs[0]]
		wrk.lambIdx = availableLambs[0]
		availableLambs = availableLambs[1:]
		wrk.dset = dset
		wrk.cov = cov
		wrk.maxIter = maxIter
		wrk.tol = tol
		wrk.corr = correction
		workChan <- wrk
	}

	numReceive := 0
	highestInserted := 0
	for numReceive < len(lambs) {
		result := <-resChan
		node := result.node
		pos := len(lambs) - result.lambIdx - 1
		nodes[pos] = node

		if result.lambIdx > highestInserted {
			highestInserted = pos
		}

		fmt.Printf("Lamb: %6.1e Num coeff. %5d\n", node.Lamb, len(node.Selection))
		numReceive++

		if len(availableLambs) > 0 {
			var wrk LassoCrdWorkload
			wrk.x0 = make([]float64, nFeat)
			hnode := nodes[highestInserted]
			for i := range hnode.Selection {
				wrk.x0[hnode.Selection[i]] = hnode.Coeff[i]
			}
			wrk.lamb = lambs[availableLambs[0]]
			wrk.lambIdx = availableLambs[0]
			availableLambs = availableLambs[1:]
			wrk.dset = dset
			wrk.cov = cov
			wrk.maxIter = maxIter
			wrk.tol = tol
			wrk.corr = correction
			workChan <- wrk
		}
	}

	firstModelWithFeatures := 0
	for i := range nodes {
		if len(nodes[i].Selection) > 0 {
			firstModelWithFeatures = i
			break
		}
	}
	return nodes[firstModelWithFeatures:]
}

// PureLassoCohen is a type that is used to calculate the Cohen's kappa value
type pureLassoCohen struct {
	Dset       *NormalizedData
	Lamb       float64
	Cov        CovMat
	MaxIter    int
	Tol        float64
	Correction LassoCorrection
	x0         []float64
	lasso      CorrectableLasso
}

// NewPureLassoCohen returns a new instance of the
// lasso cohen
func NewPureLassoCohen() *pureLassoCohen {
	return &pureLassoCohen{
		lasso: LassoCrdDesc,
	}
}

// GetX returns the full design matrix
func (p *pureLassoCohen) GetX() mat.Matrix {
	return p.Dset.X
}

// GetSelection returns the selected variables when only data corresponding
// to indices is used
func (p *pureLassoCohen) GetSelection(indices []int) []int {
	_, nc := p.Dset.X.Dims()
	X := mat.NewDense(len(indices), nc, nil)
	y := make([]float64, len(indices))

	if len(p.x0) != nc {
		p.x0 = make([]float64, nc)
	}

	for i, idx := range indices {
		for j := 0; j < nc; j++ {
			X.Set(i, j, p.Dset.X.At(idx, j))
		}
		y[i] = p.Dset.y[idx]
	}

	normD := NewNormalizedData(X, y)
	coeff := p.lasso(normD, p.Lamb, p.Cov, p.x0, p.MaxIter, p.Tol, p.Correction)
	p.x0 = coeff
	selection := []int{}

	for i := range coeff {
		if math.Abs(coeff[i]) > lassoCrdDescZero {
			selection = append(selection, i)
		}
	}
	return selection
}

// HyperParameters returns a map with the hyper parameters
func (p *pureLassoCohen) HyperParameters() map[string]float64 {
	hyper := make(map[string]float64)
	hyper["lamb"] = p.Lamb
	return hyper
}

// StringRep returns a string representation used for logging
func (p *pureLassoCohen) StringRep() string {
	return fmt.Sprintf("Lambda: %5.3e", p.Lamb)
}

// ClassoCohen is a type used to runs Cohen's kappa calculations using the covariance lasso
type cLassoCohen struct {
	PLasso *pureLassoCohen
	Eta    float64
}

// NewCLassoCohen returns a new instance of cLassoCohen
func NewCLassoCohen() *cLassoCohen {
	return &cLassoCohen{
		PLasso: NewPureLassoCohen(),
	}
}

func (c *cLassoCohen) GetX() mat.Matrix {
	return c.PLasso.GetX()
}

func (c *cLassoCohen) GetSelection(indices []int) []int {
	return c.PLasso.GetSelection(indices)
}

func (c *cLassoCohen) HyperParameters() map[string]float64 {
	hyper := make(map[string]float64)
	hyper["lamb"] = c.PLasso.Lamb
	hyper["eta"] = c.Eta
	return hyper
}

func (c *cLassoCohen) StringRep() string {
	return fmt.Sprintf("Lambda: %5.3e Eta: %5.3e", c.PLasso.Lamb, c.Eta)
}
