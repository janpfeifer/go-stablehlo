/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package shapes

import (
	"reflect"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
)

func TestCastAsDType(t *testing.T) {
	value := [][]int{{1, 2}, {3, 4}, {5, 6}}
	{
		want := [][]float32{{1, 2}, {3, 4}, {5, 6}}
		got := CastAsDType(value, dtypes.Float32)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("CastAsDType(value, dtypes.Float32) = %v, want %v", got, want)
		}
	}
	{
		want := [][]complex64{{1, 2}, {3, 4}, {5, 6}}
		got := CastAsDType(value, dtypes.Complex64)
		if !reflect.DeepEqual(want, got) {
			t.Errorf("CastAsDType(value, dtypes.Complex64) = %v, want %v", got, want)
		}
	}
}

func TestShape(t *testing.T) {
	invalidShape := Invalid()
	if invalidShape.Ok() {
		t.Error("expected invalidShape.Ok() to be false")
	}

	shape0 := Make(dtypes.Float64)
	if !shape0.Ok() {
		t.Error("expected shape0.Ok() to be true")
	}
	if !shape0.IsScalar() {
		t.Error("expected shape0.IsScalar() to be true")
	}
	if shape0.IsTuple() {
		t.Error("expected shape0.IsTuple() to be false")
	}
	if shape0.Rank() != 0 {
		t.Errorf("expected shape0.Rank() to be 0, got %d", shape0.Rank())
	}
	if len(shape0.Dimensions) != 0 {
		t.Errorf("expected len(shape0.Dimensions) to be 0, got %d", len(shape0.Dimensions))
	}
	if shape0.Size() != 1 {
		t.Errorf("expected shape0.Size() to be 1, got %d", shape0.Size())
	}
	if int(shape0.Memory()) != 8 {
		t.Errorf("expected shape0.Memory() to be 8, got %d", int(shape0.Memory()))
	}

	shape1 := Make(dtypes.Float32, 4, 3, 2)
	if !shape1.Ok() {
		t.Error("expected shape1.Ok() to be true")
	}
	if shape1.IsScalar() {
		t.Error("expected shape1.IsScalar() to be false")
	}
	if shape1.IsTuple() {
		t.Error("expected shape1.IsTuple() to be false")
	}
	if shape1.Rank() != 3 {
		t.Errorf("expected shape1.Rank() to be 3, got %d", shape1.Rank())
	}
	if len(shape1.Dimensions) != 3 {
		t.Errorf("expected len(shape1.Dimensions) to be 3, got %d", len(shape1.Dimensions))
	}
	if shape1.Size() != 4*3*2 {
		t.Errorf("expected shape1.Size() to be %d, got %d", 4*3*2, shape1.Size())
	}
	if int(shape1.Memory()) != 4*4*3*2 {
		t.Errorf("expected shape1.Memory() to be %d, got %d", 4*4*3*2, int(shape1.Memory()))
	}
}

func checkPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	f()
}

func checkNotPanic(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("The code panicked with: %v", r)
		}
	}()
	f()
}

func TestDim(t *testing.T) {
	shape := Make(dtypes.Float32, 4, 3, 2)
	if shape.Dim(0) != 4 {
		t.Errorf("expected shape.Dim(0) == 4, got %d", shape.Dim(0))
	}
	if shape.Dim(1) != 3 {
		t.Errorf("expected shape.Dim(1) == 3, got %d", shape.Dim(1))
	}
	if shape.Dim(2) != 2 {
		t.Errorf("expected shape.Dim(2) == 2, got %d", shape.Dim(2))
	}
	if shape.Dim(-3) != 4 {
		t.Errorf("expected shape.Dim(-3) == 4, got %d", shape.Dim(-3))
	}
	if shape.Dim(-2) != 3 {
		t.Errorf("expected shape.Dim(-2) == 3, got %d", shape.Dim(-2))
	}
	if shape.Dim(-1) != 2 {
		t.Errorf("expected shape.Dim(-1) == 2, got %d", shape.Dim(-1))
	}
	checkPanic(t, func() { _ = shape.Dim(3) })
	checkPanic(t, func() { _ = shape.Dim(-4) })
}

func TestFromAnyValue(t *testing.T) {
	shape, err := FromAnyValue([]int32{1, 2, 3})
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	checkNotPanic(t, func() { shape.Assert(dtypes.Int32, 3) })

	shape, err = FromAnyValue([][][]complex64{{{1, 2, -3}, {3, 4 + 2i, -7 - 1i}}})
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	checkNotPanic(t, func() { shape.Assert(dtypes.Complex64, 1, 2, 3) })

	// Irregular shape is not accepted:
	shape, err = FromAnyValue([][]float32{{1, 2, 3}, {4, 5}})
	if err == nil {
		t.Errorf("irregular shape should have returned an error, instead got shape %s", shape)
	}
}
