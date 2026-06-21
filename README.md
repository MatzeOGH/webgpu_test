# webgpu_test

WebGPU renderer with a clustered-mesh ("nanite") pipeline and a per-frame RenderGraph. Builds two
ways: native Windows (Dawn + SDL3) and WebAssembly (Emscripten / emdawnwebgpu).

Dawn (prebuilt) and SDL3 are downloaded on first configure by CMake FetchContent — no manual install.

## Prerequisites

- **Native:** Visual Studio 2022 or newer (MSVC + "Desktop development with C++"), Windows 10/11 SDK,
  CMake ≥ 3.22.
- **Web:** emsdk (Emscripten). Set `EMSDK` in `build_web.bat` to your install path.
- First configure downloads deps into `build-win\_deps\` / `build-web\_deps\`.

## Native Windows

Release (default), or pass a config:

```
build_win.bat
build_win.bat Debug
```

Configures `build-win\` and builds every target (`app`, `nanite_builder`, `rg`). Output:
`build-win\<Config>\app.exe`. Runtime DLLs (SDL3, dxil/dxcompiler/d3dcompiler) are copied next to each
exe automatically.

Raw CMake equivalent:

```
cmake -B build-win -S .
cmake --build build-win --config Release
```

## Web (WebAssembly)

Release (default), or pass a config:

```
build_web.bat
build_web.bat Debug
```

Output: `build-web\index.html` (renamed from `app.html`). Serve it:

```
python -m http.server 8080 --directory build-web
```

Raw CMake equivalent:

```
emcmake cmake -B build-web -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build-web
```

## RenderGraph standalone

`src/RenderGraph.cpp` is a single TU — the SDL smoke-test driver `RenderGraph_main.cpp` is `#include`d
at its end. Two ways to build it.

**Fast loop** — direct `cl`, no CMake, compiles and runs:

```
build_rg.bat
```

Writes `build-win\Release\rg.exe` and runs it. If `cl` isn't on PATH it calls `msvcsetup.bat` (edit
`VSINSTALLDIR` there for your VS edition).

**Visual Studio with Edit & Continue** — the CMake `rg` target:

1. `build_win.bat` (configures the solution).
2. Open `build-win\webgpu_app.slnx`.
3. Set `rg` as the startup project, pick **Debug**, press F5.

Debug uses `/ZI` + incremental linking, so you can edit `RenderGraph.cpp` / `RenderGraph_main.cpp`
mid-session and continue without restarting. (`.slnx` is CMake 4.x's solution format; VS 2022 17.10+ /
VS 18 open it natively.)

## Mesh tool

`nanite_builder` converts a glTF into the clustered `.mesh` format. Build it with `build_win.bat`, then:

```
build-mesh.bat
```

Runs `nanite_builder.exe assets\Untitled.gltf assets\n.mesh --dot assets\n.dot`. Edit the paths in the
script for other inputs.

## Reference

### Scripts

| Script | Does |
|--------|------|
| `build_win.bat [Config]` | Configure + build native (`build-win\`), all targets. Default Release. |
| `build_web.bat [Config]` | Configure + build web (`build-web\`) via Emscripten. Default Release. |
| `build_rg.bat` | Compile + run the RenderGraph smoke test standalone (no CMake). |
| `build-mesh.bat` | Run `nanite_builder` on the asset glTF. |
| `msvcsetup.bat` | Put MSVC `cl` on PATH (called by `build_rg.bat`). |

### CMake targets

| Target | Source | Output | Platform |
|--------|--------|--------|----------|
| `app` | `main_SDL.cpp` / `main_web.cpp` + Framework / Renderer / ClusteredMesh | `app.exe` / `index.html` | native + web |
| `nanite_builder` | `mainNaniteBuilder.cpp` | `nanite_builder.exe` | native |
| `rg` | `RenderGraph.cpp` (+ included driver) | `rg.exe` | native |
