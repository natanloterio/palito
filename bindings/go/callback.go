package palito

// #include <stddef.h>
// #include <stdint.h>
import "C"

import "runtime/cgo"

// textSink receives streamed text; it returns false to stop generation.
type textSink func(string) bool

//export palitoGoText
func palitoGoText(text *C.char, length C.size_t, handle C.uintptr_t) C.int {
	sink := cgo.Handle(handle).Value().(textSink)
	if sink(C.GoStringN(text, C.int(length))) {
		return 0
	}
	return 1
}
