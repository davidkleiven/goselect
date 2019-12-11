package featselect

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// ThresholdOperator is a type the is used to threshold matrices
type ThresholdOperator struct {
	threshold  float64
	numRemoved int
}

// Apply sets all elements in X that is smaller than the
// threshold to 0
func (t *ThresholdOperator) Apply(X mat.Mutable) {
	nr, nc := X.Dims()
	t.numRemoved = 0
	for i := 0; i < nr; i++ {
		for j := 0; j < nc; j++ {
			if math.Abs(X.At(i, j)) < t.threshold {
				X.Set(i, j, 0.0)
				t.numRemoved++
			}
		}
	}
}

// ValueGridPt is a type that holds a float value as well as its channel number
type ValueGridPt struct {
	value  float64
	gridPt int
}

// Workload is a struct with information to be passed to a go-worker
type Workload struct {
	cvMat1 *mat.Dense
	cvMat2 *mat.Dense
	gr     int
	step   float64
}

// L2ConsistentCovTO returns a threshold operator that yield a sparse
// approximation to the covariance matrix X^TX when applied. It does so by
// partitioning the data (rows) into to matrix at random a number times (numSamples)
// then it searches on a grid with (numGrid) for the optimal threshold. The grid is
// defined by {j*sqrt((log p)/n): 0 <= j < numGrud}, where p is the number of columns and
// n is the number of rows
func L2ConsistentCovTO(X mat.Matrix, numSamples int, numGrid int) *ThresholdOperator {
	nr, nc := X.Dims()
	step := math.Sqrt(math.Log(float64(nc)) / float64(nr))

	n1 := int(float64(nr) * (1. - 1./math.Log(float64(nr))))

	resChannel := make(chan ValueGridPt)
	workChannel := make(chan Workload)

	numSim := 10

	if numGrid < numSim {
		numSim = numGrid
	}

	gridPt := 0
	mat1, mat2 := RandomRowSplit(X, n1)
	cvMat1 := CovarianceMatrix(mat1)
	cvMat2 := CovarianceMatrix(mat2)
	for i := 0; i < numSim; i++ {
		go PerformNormDiffWork(workChannel, resChannel)
		var work Workload
		work.cvMat1 = cvMat1
		work.cvMat2 = cvMat2
		work.gr = gridPt
		work.step = step
		workChannel <- work
		gridPt++
	}

	meanNormDiff := make([]float64, numGrid)

	numReceives := 0
	ticker := time.Tick(10 * time.Second)
	for numReceives < numSamples*numGrid {
		select {
		case res := <-resChannel:
			meanNormDiff[res.gridPt] += res.value / float64(numSamples)
			var work Workload
			work.cvMat1 = cvMat1
			work.cvMat2 = cvMat2
			work.gr = gridPt
			work.step = step

			gridPt++
			numReceives++

			// When we have received a result from all grid point
			// we create a new partition
			if numReceives%numGrid == 0 {
				mat1, mat2 = RandomRowSplit(X, n1)
				cvMat1 = CovarianceMatrix(mat1)
				cvMat2 = CovarianceMatrix(mat2)
				work.cvMat1 = cvMat1
				work.cvMat2 = cvMat2
				work.gr = 0
				gridPt = 1
			}
			if gridPt <= numGrid {
				workChannel <- work
			}
		case <-ticker:
			fmt.Printf("Sample %10d of %10d\n", numReceives, numSamples*numGrid)
		}
	}
	fmt.Printf("Finished listening...\n")

	minIdx := Argmin(meanNormDiff)
	var op ThresholdOperator
	op.threshold = float64(minIdx) * step
	fmt.Printf("%v\n", meanNormDiff)
	return &op
}

// RandomRowSplit splits the rows of a matrix into two new matrices. The first
// matrix will have num rows, and the second will have N - num rows
func RandomRowSplit(X mat.Matrix, num int) (*mat.Dense, *mat.Dense) {
	nr, nc := X.Dims()
	n2 := nr - num

	mat1 := mat.NewDense(num, nc, nil)
	mat2 := mat.NewDense(n2, nc, nil)

	indices := make([]int, nr)
	for i := 0; i < nr; i++ {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

	for i := 0; i < num; i++ {
		for j := 0; j < nc; j++ {
			mat1.Set(i, j, X.At(indices[i], j))
		}
	}

	for i := num; i < nr; i++ {
		for j := 0; j < nc; j++ {
			mat2.Set(i-num, j, X.At(indices[i], j))
		}
	}
	return mat1, mat2
}

// CalculateFNormDiff between the thresholded version of mat1 and mat2. The method does
// not alter mat1, so it can be re-used
func CalculateFNormDiff(mat1 *mat.Dense, mat2 *mat.Dense, gridPt int, step float64, res chan<- ValueGridPt) {
	var op ThresholdOperator
	op.threshold = float64(gridPt) * step
	mat1Cpy := mat.DenseCopyOf(mat1)
	op.Apply(mat1Cpy)

	nr, nc := mat1Cpy.Dims()
	for i := 0; i < nr; i++ {
		for j := 0; j < nc; j++ {
			mat1Cpy.Set(i, j, mat1Cpy.At(i, j)-mat2.At(i, j))
		}
	}

	var result ValueGridPt
	result.value = mat.Norm(mat1Cpy, 2.0)
	result.gridPt = gridPt
	res <- result
}

// CovarianceMatrix returns the covariance matrix of X with itself
func CovarianceMatrix(X mat.Matrix) *mat.Dense {
	nr, nc := X.Dims()
	cov := mat.NewDense(nc, nc, nil)
	cov.Product(X.T(), X)
	cov.Scale(1./float64(nr), cov)
	return cov
}

// PerformNormDiffWork executes the normalization work
func PerformNormDiffWork(workload <-chan Workload, res chan<- ValueGridPt) {
	for wrk := range workload {
		CalculateFNormDiff(wrk.cvMat1, wrk.cvMat2, wrk.gr, wrk.step, res)
	}
}
