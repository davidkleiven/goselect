package featselect

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"

	"gonum.org/v1/gonum/floats"

	"gonum.org/v1/gonum/mat"
)

// Dataset is a structure that holds fitting data for linear fitting
type Dataset struct {
	X         *mat.Dense
	Y         []float64
	Names     []string
	TargetCol int
}

// GetFeatName gets the name of the feature corresponding to the i-th column in X
func (d *Dataset) GetFeatName(i int) string {
	if i < d.TargetCol {
		return d.Names[i]
	}
	return d.Names[i+1]
}

// FeatNoByName returns the features number corresponding to the
// passed name
func (d *Dataset) FeatNoByName(name string) int {
	for i, n := range d.Names {
		if n == name {
			if i > d.TargetCol {
				return i - 1
			}
			return i
		}
	}
	panic("FeatNoByName: Did not find any names that matched")
}

// GetSubset returns a new dataset consisting only of the selected
// features
func (d *Dataset) GetSubset(features []int) *Dataset {
	var subset Dataset
	nr, _ := d.X.Dims()

	subset.X = mat.NewDense(nr, len(features), nil)
	subset.Y = make([]float64, nr)
	subset.Names = make([]string, len(features)+1)

	for r := 0; r < nr; r++ {
		for col, v := range features {
			subset.X.Set(r, col, d.X.At(r, v))
		}
	}

	copy(subset.Y, d.Y)

	for i, v := range features {
		subset.Names[i] = d.GetFeatName(v)
	}
	subset.TargetCol = len(features)
	subset.Names[len(features)] = d.Names[d.TargetCol]
	return &subset
}

// IsEqual returns true if the two datasets are equal
func (d *Dataset) IsEqual(o *Dataset) bool {
	nr, nc := d.X.Dims()
	nr1, nc1 := o.X.Dims()

	if nr != nr1 || nc != nc1 || len(d.Y) != len(o.Y) || len(d.Names) != len(o.Names) || d.TargetCol != o.TargetCol {
		return false
	}

	tol := 1e-10
	if !floats.EqualApprox(d.Y, o.Y, tol) {
		return false
	}

	if !mat.EqualApprox(d.X, o.X, tol) {
		return false
	}
	return true
}

// Copy returns a deepcopy of the dataset
func (d *Dataset) Copy() *Dataset {
	var cpy Dataset
	cpy.X = mat.DenseCopyOf(d.X)
	cpy.Y = make([]float64, len(d.Y))
	copy(cpy.Y, d.Y)
	cpy.TargetCol = d.TargetCol
	cpy.Names = make([]string, len(d.Names))
	copy(cpy.Names, d.Names)
	return &cpy
}

// ReadCSV reads a dataset from a csv file
func ReadCSV(fname string, targetCol int) *Dataset {
	csvFile, err := os.Open(fname)
	if err != nil {
		fmt.Printf("Could not open file %s\n", fname)
		return nil
	}
	defer csvFile.Close()
	return ParseCSV(csvFile, targetCol)
}

// ParseCSV parses data from CSV file. It is assumed that the file starts with a header
// The values in the column targetCol is placed in y of the returned struct and the rest
// of the columns are placed in a matrix
func ParseCSV(handle io.Reader, targetCol int) *Dataset {
	reader := csv.NewReader(bufio.NewReader(handle))
	var dset Dataset

	lineNo := 0
	data := make([][]float64, 0)
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}

		if lineNo == 0 {
			dset.Names = line
		} else {
			row := make([]float64, len(line))
			for i, v := range line {
				if fl, err := strconv.ParseFloat(v, 64); err == nil {
					row[i] = fl
				}
			}
			data = append(data, row)
		}
		lineNo++
	}

	numCol := len(dset.Names)
	if targetCol < 0 {
		targetCol += numCol
	}

	dset.TargetCol = targetCol
	dset.X = mat.NewDense(len(data), len(data[0])-1, nil)
	dset.Y = make([]float64, len(data))
	for row := 0; row < len(data); row++ {
		xcol := 0
		for col := 0; col < len(data[row]); col++ {
			if col == targetCol {
				dset.Y[row] = data[row][col]
			} else {
				dset.X.Set(row, xcol, data[row][col])
				xcol++
			}
		}
	}
	return &dset
}

// Save dataset to a csv file
func (dset *Dataset) Save(fname string) {
	csvFile, err := os.Create(fname)
	if err != nil {
		return
	}
	defer csvFile.Close()
	dset.SaveHandle(csvFile)
}

// SaveHandle writes the output to a writer
func (dset *Dataset) SaveHandle(handle io.Writer) {
	writer := csv.NewWriter(bufio.NewWriter(handle))
	nrows, ncols := dset.X.Dims()
	values := make([]string, ncols+1)

	writer.Write(dset.Names)

	for i := 0; i < nrows; i++ {
		shift := 0
		for j := 0; j < ncols+1; j++ {
			if j == dset.TargetCol {
				values[j] = fmt.Sprintf("%f", dset.Y[i])
				shift = 1
			} else {
				values[j] = fmt.Sprintf("%f", dset.X.At(i, j-shift))
			}
		}
		writer.Write(values)
	}
	writer.Flush()
}

// JSONDataset is a type defined to be able to read/write dataset in a simple way from JSON files
type JSONDataset struct {
	X         []float64
	Y         []float64
	TargetCol int
	Names     []string
	Nr, Nc    int
}

// MarshalJSON is implemented to add the Dataset type to a JSON file
func (dset *Dataset) MarshalJSON() ([]byte, error) {
	nr, nc := dset.X.Dims()
	var jData JSONDataset
	jData.Nr = nr
	jData.Nc = nc
	jData.Y = dset.Y
	jData.TargetCol = dset.TargetCol
	jData.Names = dset.Names

	jData.X = make([]float64, nr*nc)
	for i := 0; i < nr; i++ {
		for j := 0; j < nc; j++ {
			jData.X[i*nc+j] = dset.X.At(i, j)
		}
	}
	return json.Marshal(jData)
}

// UnmarshalJSON returns a datasetom from JSON
func (dset *Dataset) UnmarshalJSON(data []byte) error {
	var jData JSONDataset
	if err := json.Unmarshal(data, &jData); err != nil {
		return err
	}

	dset.Y = jData.Y
	dset.Names = jData.Names
	dset.TargetCol = jData.TargetCol
	fmt.Printf("%v\n", jData.Nr)
	dset.X = mat.NewDense(jData.Nr, jData.Nc, jData.X)
	return nil
}
