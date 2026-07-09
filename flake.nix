{
  description = "trd - tile/relational wgpu rendering library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        # Runtime libraries the wgpu/Vulkan backend loads via dlopen.
        runtimeLibs = with pkgs; [
          vulkan-loader
          libGL
          wayland
          libxkbcommon
          libx11
          libxcursor
          libxi
          libxrandr
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            rustToolchain
            bun
            wasm-bindgen-cli
            vulkan-loader
            vulkan-tools
            pkg-config
          ] ++ runtimeLibs;

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"
            # Mesa (nixpkgs) bundles the Dozen "dzn" ICD (Vulkan-on-D3D12). It is
            # never useful on a Linux GPU machine and crashes during Vulkan
            # adapter enumeration when no D3D12 backend exists, so disable it.
            # The loader still selects a real GPU adapter when one is present.
            export VK_LOADER_DRIVERS_DISABLE="*dzn*"

            # WSL2: the GPU is exposed only via /dev/dxg (D3D12), and NVIDIA ships
            # no native Linux Vulkan ICD for WSL. Expose the Windows GPU userspace
            # libs so Mesa's d3d12 driver can reach the real GPU, and select it for
            # OpenGL. Then `WGPU_BACKEND=gl` renders on the GPU (the default Vulkan
            # backend falls back to software llvmpipe on WSL).
            if [ -e /dev/dxg ] && [ -d /usr/lib/wsl/lib ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
              export GALLIUM_DRIVER=d3d12
              echo "trd: WSL2 GPU detected - use WGPU_BACKEND=gl for OpenGL-on-D3D12 GPU rendering"
            fi

            echo "trd devShell: $(rustc --version), bun $(bun --version)"
          '';
        };
      });
}
