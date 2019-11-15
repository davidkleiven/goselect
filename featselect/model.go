package featselect

// Model is a wrapper around a bit array
type Model struct {
	bits []uint8
	size int
}

// NewModel create a new model of a given size
func NewModel(size int) *Model {
	var m Model
	m.bits = make([]uint8, size/8+1)
	m.size = size
	return &m
}

// Get return true if feature at position index is 1
func (m *Model) Get(index int) bool {
	arrayIndex := index / 8
	reminder := index - 8*arrayIndex
	return m.bits[arrayIndex]&uint8(0x01<<uint(reminder)) == uint8(0x01<<uint(reminder))
}

// Set sets the bit at position index
func (m *Model) Set(index int) {
	arrayIndex := index / 8
	reminder := index - 8*arrayIndex
	m.bits[arrayIndex] |= 0x01 << uint(reminder)
}

// Flip flips the bit at position
func (m *Model) Flip(index int) {
	arrayIndex := index / 8
	reminder := index - 8*arrayIndex
	m.bits[arrayIndex] ^= 0x01 << uint(reminder)
}

// ToBools converts the model into an array of booleans
func (m *Model) ToBools() []bool {
	res := make([]bool, m.size)
	for i := 0; i < m.size; i++ {
		res[i] = m.Get(i)
	}
	return res
}
