package featselect

// Gcs returns the greatest common model (gcs).
// GCS is equal to the model with the largest
// amount of features, given that all "bits"
// up to start is not altered
func Gcs(model []bool, start int) []bool {
	gcsMod := make([]bool, len(model))
	copy(gcsMod, model[:start])
	for i := start; i < len(model); i++ {
		gcsMod[i] = true
	}
	return gcsMod
}

// Lcs returns the least common model. The lcs is the model with
// as few features as possible given that all "bits"
// up to start are not altered
func Lcs(model []bool, start int) []bool {
	lcsMod := make([]bool, len(model))
	copy(lcsMod, model[:start])

	for i := start; i < len(model); i++ {
		lcsMod[i] = false
	}
	return lcsMod
}

// NumFeatures returns the number of features in a model
func NumFeatures(model []bool) int {
	num := 0
	for _, flag := range model {
		if flag {
			num++
		}
	}
	return num
}
