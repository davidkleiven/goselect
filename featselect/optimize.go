package featselect

// Return the greatest common model (gcs).
// GCS is equal to the model with the largest
// amount of features, given that all "bits"
// up to start is not altered
func GCS(model []bool, start int) []bool {
	gcsMod := make([]bool, len(model))
	copy(gcsMod, model[:start])
	for i := start; i < len(model); i++ {
		gcsMod[i] = true
	}
	return gcsMod
}

// Return the least common model. The lcs is the model with
// as few features as possible given that all "bits"
// up to start are not altered
func LCS(model []bool, start int) []bool {
	lcsMod := make([]bool, len(model))
	copy(lcsMod, model[:start])

	for i := start; i < len(model); i++ {
		lcsMod[i] = false
	}
	return lcsMod
}

// Return the number of features in a model
func numFeatures(model []bool) int {
	num := 0
	for _, flag := range model {
		if flag {
			num += 1
		}
	}
	return num
}
