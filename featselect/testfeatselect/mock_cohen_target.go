package testfeatselect

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// MockCohenTarget implements the CohenTarget interface
type MockCohenTarget struct {
	Mode int
}

// GetX returns an empety matrix
func (m *MockCohenTarget) GetX() mat.Matrix {
	X := mat.NewDense(10, 10, nil)
	return X
}

// GetSelection returns a selection
func (m *MockCohenTarget) GetSelection(indices []int) []int {
	if m.Mode == 0 {
		selection := []int{0, 4, 8}
		return selection
	}
	numSelect := rand.Intn(10)
	selection := make([]int, numSelect)
	for i := 0; i < numSelect; i++ {
		selection[i] = rand.Intn(10)
	}
	return selection
}

// HyperParameters returns a map with the hyper parameters
func (m *MockCohenTarget) HyperParameters() map[string]float64 {
	res := make(map[string]float64)
	res["mode"] = float64(m.Mode)
	return res
}

// StringRep returns a string representation of the hyper parameters
func (m *MockCohenTarget) StringRep() string {
	return fmt.Sprintf("Mode: %2d", m.Mode)
}
