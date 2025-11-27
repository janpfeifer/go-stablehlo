package main

import "k8s.io/klog/v2"

func must(err error) {
	if err != nil {
		klog.Fatalf("Failed with error: %+v", err)
	}
}

func must1[T any](value T, err error) T {
	must(err)
	return value
}

func main() {
	GenerateBinaryOps()
	GenerateUnaryOps()
}
