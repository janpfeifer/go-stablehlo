package shapes

import (
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
)

func TestToStableHLO(t *testing.T) {
	shape := Make(dtypes.Float32, 1, 10)
	if shape.ToStableHLO() != "tensor<1x10xf32>" {
		t.Errorf("expected tensor<1x10xf32>, got %s", shape.ToStableHLO())
	}

	// Test scalar.
	shape = Make(dtypes.Int32)
	if shape.ToStableHLO() != "tensor<i32>" {
		t.Errorf("expected tensor<i32>, got %s", shape.ToStableHLO())
	}
}
