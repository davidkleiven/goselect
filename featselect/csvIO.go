package featselect

import (
	"bufio"
	"encoding/csv"
	"io"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type Dataset struct {
	X     *mat.Dense
	Y     []float64
	names []string
}

// ReadCSV reads a dataset from a csv file
func ReadCSV(fname string, targetCol int) *Dataset {
	csvFile, err := os.Open(fname)
	if err != nil {
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
			dset.names = line
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
