package featselect

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

const lassoCrdDescZero = 1e-16

// LassoCrdDesc solves the lasso problem via coordinate descent
func LassoCrdDesc(dset *NormalizedData, lamb float64, cov CovMat, x0 []float64, maxIter int) []float64 {
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
	tol := 1e-10
	converged := false
	for iter := 0; iter < maxIter; iter++ {
		for _, j := range iterIndices {
			covDiag := covMat.At(j, j)
			oldCoeff := betaOld[j]
			covDotBetaNoDiag := covDotBeta[j] - betaOld[j]*covDiag
			newCoeff := XTy.AtVec(j)/float64(nr) - covDotBetaNoDiag
			newCoeff = SoftThreshold(newCoeff, lamb) / covDiag

			UpdateCovDotBeta(covMat, covDotBeta, j, oldCoeff, newCoeff)
			beta[j] = newCoeff
		}

		converged = true
		for _, j := range iterIndices {
			if math.Abs(beta[j]-betaOld[j]) > tol {
				converged = false
				break
			}
		}

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
			unsatisfied := UnsatisfiedKKTConditions(XTy, covDotBeta, beta, lamb)
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
func UnsatisfiedKKTConditions(Xy mat.Vector, covDotBeta []float64, coeff []float64, lamb float64) []int {
	unsatisfied := []int{}
	for i := 1; i < len(covDotBeta); i++ {
		if math.Abs(Xy.AtVec(i)-covDotBeta[i]) < lamb && math.Abs(coeff[i]) > lassoCrdDescZero {
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
}

// LassoRes is a structure used to return the result
type LassoRes struct {
	node    *LassoLarsNode
	lambIdx int
}

// PerformLassoCrd listens to the workload channel and passes its result to res
func PerformLassoCrd(workload <-chan LassoCrdWorkload, res chan<- LassoRes) {
	for wrk := range workload {
		coeff := LassoCrdDesc(wrk.dset, wrk.lamb, wrk.cov, wrk.x0, wrk.maxIter)
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
func LassoCrdDescPath(dset *NormalizedData, cov CovMat, lambs []float64, maxIter int) []*LassoLarsNode {
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
			workChan <- wrk
		}
	}
	return nodes
}
