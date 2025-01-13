# Tiny CBOR

[![](https://img.shields.io/github/actions/workflow/status/levischuck/tiny-cbor/build.yaml?branch=main)](https://github.com/LeviSchuck/tiny-cbor/actions)
[![](https://img.shields.io/codecov/c/gh/levischuck/tiny-cbor?style=flat-square)](https://codecov.io/gh/levischuck/tiny-cbor)
[![](https://img.shields.io/github/v/tag/levischuck/tiny-cbor?label=npm&logo=npm&style=flat-square)](https://www.npmjs.com/package/@levischuck/tiny-cbor)
[![](https://img.shields.io/github/v/tag/levischuck/tiny-cbor?label=deno&logo=deno&style=flat-square)](https://deno.land/x/tiny_cbor)
[![](https://img.shields.io/github/license/levischuck/tiny-cbor)](https://github.com/LeviSchuck/tiny-cbor/blob/main/LICENSE.txt)
![](https://img.shields.io/bundlephobia/min/%40levischuck/tiny-cbor)

This minimal generic library decodes and encodes most useful CBOR structures
into simple JavaScript structures:

- Maps with keys as `string`s or `number`s with `CBORType` values as a `Map`
- Arrays of `CBORType` values
- integers as `number`s
- float32 and float64 as `number`s
- float16 `NaN`, `Infinity`, `-Infinity`
- `string`s
- byte strings as `Uint8Array`
- booleans
- `null` and `undefined`
- tags as `CBORTag(tag, value)`

## Limitations

This implementation does not support:

- indefinite length maps, arrays, text strings, or byte strings.
- half precision floating point numbers
- integers outside the range of `[-9007199254740991, 9007199254740991]`, see
  [Number.MAX_SAFE_INTEGER](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER)
- native output to JSON
- does not support generic objects, only `Map`s

This implementation has the following constraints:

- Map keys may only be strings or numbers
- Tags are not interpreted

## Behavior

Maps that have duplicate keys will throw an error during decoding. Decoding data
that is incomplete will throw an error during decoding.
