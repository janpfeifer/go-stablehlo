# [XLA](https://openxla.org/)'s [StableHLO](https://openxla.org/stablehlo) Builder API for Go

[![GoDev](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white)](https://pkg.go.dev/github.com/gx-org/go-stablehlo?tab=doc)
[![GitHub](https://img.shields.io/github/license/gx-org/go-stablehlo)](https://github.com/gx-org/go-stablehlo/blob/main/LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/gx-org/go-stablehlo)](https://goreportcard.com/report/github.com/gx-org/go-stablehlo)
[![TestStatus](https://github.com/gx-org/go-stablehlo/actions/workflows/tests.yaml/badge.svg)](https://github.com/gx-org/go-stablehlo/actions/workflows/tests.yaml)
[![Slack](https://img.shields.io/badge/Slack-GoMLX-purple.svg?logo=slack)](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
[![Sponsor gomlx](https://img.shields.io/badge/Sponsor-gomlx-white?logo=github&style=flat-square)](https://github.com/sponsors/gomlx)

> [!Note]
> - This project started in [github.com/gomlx/stablehlo](https://github.com/gomlx/stablehlo), and it was moved under
>   the [gx-org](https://github.com/gx-org) organization on v0.2.0, in an effort to create a common generic backend
>   engine and concrete backends (XLA being the primary target) for both projects, GX and GoMLX.
> - Discussion in the [Slack channel #gomlx](https://app.slack.com/client/T029RQSE6/C08TX33BX6U)
>   (you can [join the slack server here](https://invite.slack.golangbridge.org/))

<img align="right" src="docs/gomlx_stablehlo_gopher.png" alt="GoMLX Gopher" width="220px"/>

[StableHLO](https://openxla.org/stablehlo) is an operation set for high-level operations (HLO) in machine learning (ML) models. 

It's the portability layer between ML frameworks (targeted for [GX](https://github.com/gx-org/gx) and 
[GoMLX](https://github.com/gomlx/gomlx), but could be used for others) and ML compilers. 
It allows for easy support for different vendors, by coupling with **XLA's PJRT** (*) API for executing
StableHLO programs. So many different GPUs and TPUs are supported.

(*) **PJRT**, which stands for Pluggable JIT Runtime, is an API in the context of XLA (Accelerated Linear Algebra)
that provides a unified, cross-platform interface for interacting with different hardware accelerators. 
StableHLO is the device-independent language to specify the computation, and it also includes APIs to handle
buffer (the data) management and optionally distributed execution.

See:

* [StableHLO specification](https://openxla.org/stablehlo/spec)
* [GoPJRT](https://github.com/gomlx/gopjrt): a Go wrapper for PJRT C API, capable of executing StableHLO programs,
  for a lower level API.
* [GoMLX](https://github.com/gomlx/gomlx): a Go ML framework that supports an XLA (StableHLO+PJRT) backend to
  efficiently run (or train) ML programs.
* [GX](https://github.com/gx-org/gx): **Experimental** portable (across various programming languages) 
  domain-specific language (DSL) for ML, based on the Go syntax, and having Go as a primary target.

## Examples

The tests in [`tests/gopjrt/gopjrt_test.go`](https://github.com/gx-org/go-stablehlo/blob/main/tests/gopjrt/gopjrt_test.go) 
should serve as simple examples of each operation.

Notice that `stablehlo` is a low-level API, usually used to build higher-level frameworks (an ML framework like GoMLX, 
maybe an image manipulation library that uses accelerators like GPUs, some scientific library, etc.), so it's deliberately 
verbose and requires boilerplate (error handling) everywhere. 
It sacrifices ergonomics for performance, consistency and stability. 

See another example of `go-stablehlo` and GoPJRT (to execute the generated StableHLO program) in 
[Mandelbrot mandelbrot.ipynb notebook](https://github.com/gomlx/gopjrt/blob/main/examples/mandelbrot.ipynb).
It includes some sample StableHLO code if you are curious.

<a href="https://github.com/gomlx/gopjrt/blob/main/examples/mandelbrot.ipynb">
<img src="https://github.com/gomlx/gopjrt/assets/7460115/d7100980-e731-438d-961e-711f04d4425e" style="width:400px; height:240px"/>
</a>

## Status of Operations

Most operations are already implemented. See the 
[list of supported operations](https://github.com/gx-org/go-stablehlo/blob/main/internal/optypes/optypes.go#L91)
(the ones not implemented are in the bottom of the list).

If you need a specific operation, please open an issue.

See also the [CHANGELOG](https://github.com/gx-org/go-stablehlo/blob/main/docs/CHANGELOG.md).

## Dynamic Shapes Support

* [Reference StableHLO documentation here](https://openxla.org/stablehlo/dynamism).
* [RFC: Dynamism 101](https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md)

For now, XLA's PJRT only supports _unbounded dynamism_ using shape polymorphism**:
where axes dimensions are not defined and have no bounds, and where PJRT will be able to dynamically
re-instantiate and re-compile the program to a new shape (or re-use a cache).

We expect this brings little benefit for GoMLX or GX, since recompilation to a new shape is quick and convenient, 
and the time will be dominated by the PJRT compilation time anyway. 
Hence, there is little priority for supporting this in the short term – please, reach out and open an issue if 
you need it.

We hope to support other backends (ONNX, ggml, Vulkan, WebML), some of which support dynamic shapes in 
some form or another, which we hope to support in the future.

Other types of dynamism:

* _Unranked dynamism_: rank unknown and compile time. **Not supported**.
* _Data-dependent dynamism_: for data-dependent dynamic ops. For instance, if a function returns the indices of all 
  non-zero elements. **There is little support for this, so we do not support it yet.**

## The `shapeinference` sub-package

The `shapeinference` sub-package provides a way to infer the shapes of StableHLO programs, 
given the shapes of the inputs. We had to reimplement this in Go from the specification to avoid any C++ dependency.
If you find any discrepancies or errors due to wrong output shapes, please open an issue.

