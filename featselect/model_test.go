package featselect

import "testing"

func TestSetGetModel(t *testing.T) {
	for j, test := range []struct {
		target []bool
	}{
		{
			target: []bool{true, false, false},
		},
		{
			target: []bool{false, true, true, false},
		},
		{
			target: []bool{true, true},
		},
		{
			target: []bool{false, false, false, true, false, false, true, true, false, false, true},
		},
		{
			target: []bool{false, false, false, false, true, true, false, true, false, true, true,
				false, true, false, true, true, false, true, false, false, true, true, true, false},
		},
	} {
		mod := NewModel(len(test.target))
		for i, v := range test.target {
			if v {
				mod.Set(i)
			}
		}

		got := mod.ToBools()
		for i := 0; i < len(test.target); i++ {
			if got[i] != test.target[i] {
				t.Errorf("Test #%v fails. Expected %v Got %v", j, test.target, got)
				break
			}
		}
	}
}

func TestFlipModel(t *testing.T) {
	for i, test := range []struct {
		expect []bool
		flips  []int
	}{
		{
			expect: []bool{false, true},
			flips:  []int{1},
		},
		{
			expect: []bool{false, false, true, false, true},
			flips:  []int{2, 4},
		},
		{
			expect: []bool{true, true, true, true, false, false, true, false, true},
			flips:  []int{0, 1, 2, 3, 6, 8},
		},
	} {
		mod := NewModel(len(test.expect))
		for _, v := range test.flips {
			mod.Flip(v)
		}

		got := mod.ToBools()

		for j := 0; j < len(got); j++ {
			if got[j] != test.expect[j] {
				t.Errorf("Test #%v fails. Expected %v Got %v", i, test.expect, got)
			}
		}
	}
}
