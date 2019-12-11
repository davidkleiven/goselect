package featselect

import (
	"fmt"
	"testing"

	"github.com/davidkleiven/goselect/featselect/testfeatselect"
)

func TestNestedLasso(t *testing.T) {
	X, y := testfeatselect.GetExampleXY()
	_, nc := X.Dims()
	var dset Dataset
	dset.X = X
	dset.Y = y
	dset.TargetCol = nc
	dset.Names = make([]string, nc+1)
	for i := 0; i < nc+1; i++ {
		dset.Names[i] = fmt.Sprintf("n%d", i)
	}

	var estimator MorsePenroseCD
	res := NestedLasso(&dset, 1e-10, 0.8, &estimator)
	expectLen := 2

	if len(res.Paths) != expectLen {
		t.Errorf("unexpected number of paths in nested lasso. Expected %d got %d", expectLen, len(res.Paths))
	}
}
