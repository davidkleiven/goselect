package featselect

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

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

	betaOld := x0
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
		copy(betaOld, beta)

		if converged {
			break
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

// LassoCrdDescPath calculates a set of lasso solutions along equi-logspaced set of lambda values
func LassoCrdDescPath(dset *NormalizedData, cov CovMat, lambs []float64, maxIter int) []*LassoLarsNode {
	_, nFeat := dset.X.Dims()
	x0 := make([]float64, nFeat)

	nodes := make([]*LassoLarsNode, len(lambs))
	for i := 0; i < len(lambs); i++ {
		lamb := lambs[len(lambs)-i-1]
		coeff := LassoCrdDesc(dset, lamb, cov, x0, maxIter)
		copy(x0, coeff)

		selection := []int{}
		selectedCoeff := []float64{}
		for j := range coeff {
			if math.Abs(coeff[j]) > lamb/2. {
				selection = append(selection, j)
				selectedCoeff = append(selectedCoeff, coeff[j])
			}
		}

		nodes[i] = NewLassoLarsNode(selectedCoeff, lamb, selection)
	}
	return nodes
}
