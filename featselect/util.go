package featselect

import (
	"math"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// GetDesignMatrix returns the design matrix corresponding to the passed model
func GetDesignMatrix(model []bool, X DesignMatrix) *mat.Dense {
	n, _ := X.Dims()
	numFeat := NumFeatures(model)

	if numFeat == 0 {
		panic("getDesignMatrix: No features in model")
	}
	design := mat.NewDense(n, numFeat, nil)

	col := 0
	for i := 0; i < len(model); i++ {
		if model[i] {
			colView := X.ColView(i)

			for j := 0; j < n; j++ {
				design.Set(j, col, colView.At(j, 0))
			}
			col++
		}
	}
	return design
}

func All(a []int, value int) bool {
	for i := 0; i < len(a); i++ {
		if a[i] != value {
			return false
		}
	}
	return true
}

func IterProduct(values []int, repeat int) [][]int {
	res := make([][]int, 1)

	for r := 0; r < repeat; r++ {
		nItems := len(res)
		updatedRes := make([][]int, 0)
		for j := 0; j < nItems; j++ {
			for i := 0; i < len(values); i++ {
				newRow := append(res[j], values[i])
				updatedRes = append(updatedRes, newRow)
			}
		}
		res = updatedRes
	}
	return res
}

func Sum(a []int) int {
	s := 0
	for _, v := range a {
		s += v
	}
	return s
}

type CommandLineOptions struct {
	Csvfile   string
	Outfile   string
	TargetCol int
	Cutoff    float64
}

// ParseCommandLineArgs parses options given on the command line
func ParseCommandLineArgs(args []string) *CommandLineOptions {
	var options CommandLineOptions
	options.Outfile = "defaultGoSelectOut.json"
	for _, v := range args {
		if strings.HasPrefix(v, "--csv=") {
			options.Csvfile = strings.SplitAfter(v, "--csv=")[1]
		} else if strings.HasPrefix(v, "--target=") {
			value := strings.SplitAfter(v, "--target=")[1]
			if num, err := strconv.ParseInt(value, 10, 0); err == nil {
				options.TargetCol = int(num)
			}
		} else if strings.HasPrefix(v, "--out=") {
			options.Outfile = strings.SplitAfter(v, "--out=")[1]
		} else if strings.HasPrefix(v, "--cutoff=") {
			value := strings.SplitAfter(v, "--cutoff=")[1]
			if num, err := strconv.ParseFloat(value, 64); err == nil {
				options.Cutoff = num
			}
		}
	}
	return &options
}

// NewLog2Pruned updates the number of pruned solutions.
// current is log2 of the current number of pruned solutions
// numPruned is log2 of the new number of pruned solutions
func NewLog2Pruned(current float64, numPruned int) float64 {
	diff := float64(numPruned) - current
	return current + math.Log2(1+math.Pow(2, diff))
}

// RearrangeDense changes the order of the columns in the matrix X such that they appear
// in the order dictated by colOrder. If colOrder = [2, 0, 4, ...], the first column
// in the new matrix is the third column in the original matrix, the second column in
// the new matrix is the first column in the original etc.
func RearrangeDense(X *mat.Dense, colOrder []int) *mat.Dense {
	nr, nc := X.Dims()

	newMat := mat.NewDense(nr, nc, nil)

	inserted := make([]bool, nc)
	numInserted := 0
	for i := 0; i < len(colOrder); i++ {
		if colOrder[i] != -1 {
			for j := 0; j < nr; j++ {
				newMat.Set(j, i, X.At(j, colOrder[i]))
			}
			inserted[colOrder[i]] = true
			numInserted++
		}
	}

	// Transfer the remaining columns in the original order
	for i := 0; i < len(inserted); i++ {
		if !inserted[i] {
			for j := 0; j < nr; j++ {
				newMat.Set(j, numInserted, X.At(j, i))
			}
			numInserted++
		}
	}
	return newMat
}

// Selected2Model converts a list of selected features into a boolean
// array of true/false indicating whether the feature is selected or not
func Selected2Model(selected []int, numFeatures int) []bool {
	model := make([]bool, numFeatures)

	for _, v := range selected {
		model[v] = true
	}
	return model
}

// Mean calculates the mean of an array
func Mean(v []float64) float64 {
	mu := 0.0
	for i := 0; i < len(v); i++ {
		mu += v[i]
	}
	return mu / float64(len(v))
}

// Std calculates the standard deviation of an array
func Std(v []float64) float64 {
	if len(v) <= 1 {
		return 0.0
	}

	sigmaSq := 0.0
	mu := Mean(v)

	for i := 0; i < len(v); i++ {
		sigmaSq += (v[i] - mu) * (v[i] - mu)
	}
	return math.Sqrt(sigmaSq / float64(len(v)-1))
}

// NormalizeArray normalizes an array to unit variance and zero mean
func NormalizeArray(v []float64) {
	mu := Mean(v)
	std := Std(v)

	for i := 0; i < len(v); i++ {
		v[i] = (v[i] - mu) / std
	}
}

// NormalizeRows normalizes all rows to unit variance and zero mean
func NormalizeRows(X *mat.Dense) {
	nrows, _ := X.Dims()
	for i := 0; i < nrows; i++ {
		NormalizeArray(X.RawRowView(i))
	}
}
