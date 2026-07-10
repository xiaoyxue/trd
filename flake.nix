{
  description = "trd - tile/relational wgpu rendering library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      crane,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        lib = pkgs.lib;

        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        craneLib = (crane.mkLib pkgs).overrideToolchain (_: rustToolchain);

        # Keep `.wgsl` shaders in the build source (trd-core embeds them via
        # `include_wgsl!`); crane's default filter would drop non-Rust files.
        src = lib.cleanSourceWith {
          src = ./.;
          filter = path: type: (lib.hasSuffix ".wgsl" path) || (craneLib.filterCargoSources path type);
          name = "source";
        };

        # wasm-bindgen requires the CLI to exactly match the `wasm-bindgen`
        # crate. Derive the CLI from Cargo.lock (rather than the unversioned
        # `pkgs.wasm-bindgen-cli` alias, which can drift on `nix flake update`)
        # so the two can never silently diverge.
        wasmBindgenVersion =
          (lib.findFirst (p: p.name == "wasm-bindgen") (throw "wasm-bindgen not found in Cargo.lock")
            (builtins.fromTOML (builtins.readFile ./Cargo.lock)).package
          ).version;
        wasmBindgenCliAttr = "wasm-bindgen-cli_${lib.replaceStrings [ "." ] [ "_" ] wasmBindgenVersion}";
        wasmBindgenCli =
          pkgs.${wasmBindgenCliAttr}
            or (throw "nixpkgs lacks ${wasmBindgenCliAttr} (to match Cargo.lock wasm-bindgen ${wasmBindgenVersion})");

        # The nix web build bundles offline: it provides the nix-built trd-wasm
        # as node_modules and skips `bun install`, and the tsc check supplies
        # typescript via nixpkgs. That shortcut is only valid while these are the
        # sole dependencies, so make the invariant executable and fail loudly if
        # a real npm dependency is ever added.
        webPackageJson = builtins.fromJSON (builtins.readFile ./web/package.json);
        checkedWebSrc =
          assert lib.assertMsg (lib.attrNames (webPackageJson.dependencies or { }) == [ "trd-wasm" ])
            "web/package.json runtime dependencies must be exactly {trd-wasm}: the nix web build skips `bun install`. Add npm-dependency support (e.g. bun2nix) before adding runtime deps.";
          assert lib.assertMsg (lib.attrNames (webPackageJson.devDependencies or { }) == [ "typescript" ])
            "web/package.json devDependencies must be exactly {typescript}: the tsc check provides it via nixpkgs. Add npm-dependency support before adding devDeps.";
          webSrc;

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

        commonArgs = {
          inherit src;
          strictDeps = true;
          nativeBuildInputs = [ pkgs.pkg-config ];
        };

        # --- Native (host) build ---------------------------------------------
        # Shared dependency artifacts for the whole workspace on the host target.
        # trd-wasm compiles to an empty crate off wasm32, so this is cheap.
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        trd-cli = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            pname = "trd-cli";
            cargoExtraArgs = "--package trd-cli";
            doCheck = false;
            nativeBuildInputs = commonArgs.nativeBuildInputs ++ [ pkgs.makeWrapper ];
            # The wgpu backend dlopens Vulkan/GL/X11 at run time; expose them.
            postInstall = ''
              wrapProgram $out/bin/trd \
                --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath runtimeLibs}
            '';
          }
        );

        # --- wasm build ------------------------------------------------------
        wasmArgs = commonArgs // {
          CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
          cargoExtraArgs = "--package trd-wasm";
          doCheck = false;
        };
        cargoArtifactsWasm = craneLib.buildDepsOnly wasmArgs;

        # Replicates the wasm-pack `web` package (JS glue + wasm + d.ts) using
        # wasm-bindgen-cli + binaryen directly, so the build stays sandbox-pure.
        trd-wasm = craneLib.buildPackage (
          wasmArgs
          // {
            pname = "trd-wasm";
            cargoArtifacts = cargoArtifactsWasm;
            nativeBuildInputs = commonArgs.nativeBuildInputs ++ [
              wasmBindgenCli
              pkgs.binaryen
            ];
            doNotPostBuildInstallCargoBinaries = true;
            installPhaseCommand = ''
              mkdir -p $out
              wasm-bindgen \
                --target web \
                --out-dir $out \
                --out-name trd_wasm \
                target/wasm32-unknown-unknown/release/trd_wasm.wasm
              wasm-opt -Oz -o $out/trd_wasm_bg.wasm $out/trd_wasm_bg.wasm
              # Drop the wasm-bindgen `*_bg.wasm.d.ts`: it types the raw wasm
              # module (no default export), which shadows the `*.wasm` ambient
              # URL-import declaration under bundler resolution. wasm-pack omits
              # it from its package `files` for the same reason.
              rm -f $out/trd_wasm_bg.wasm.d.ts
              cat > $out/package.json <<'EOF'
              {
                "name": "trd-wasm",
                "type": "module",
                "version": "0.1.0",
                "main": "trd_wasm.js",
                "types": "trd_wasm.d.ts",
                "files": [
                  "trd_wasm_bg.wasm",
                  "trd_wasm.js",
                  "trd_wasm.d.ts"
                ]
              }
              EOF
            '';
          }
        );

        # --- Web bundle ------------------------------------------------------
        webSrc = lib.cleanSourceWith {
          src = ./web;
          filter =
            path: type:
            let
              b = baseNameOf path;
            in
            b != "node_modules" && b != "dist";
          name = "web-src";
        };

        # Provide the sole runtime dependency (the nix-built wasm package) as
        # node_modules/trd-wasm; bun then bundles offline with no `bun install`.
        provideNodeModules = ''
          mkdir -p node_modules
          cp -r ${trd-wasm} node_modules/trd-wasm
          chmod -R u+w node_modules
        '';

        web = pkgs.stdenv.mkDerivation {
          pname = "trd-web";
          version = "0.1.0";
          src = checkedWebSrc;
          nativeBuildInputs = [ pkgs.bun ];
          buildPhase = ''
            runHook preBuild
            export HOME=$TMPDIR
            export DO_NOT_TRACK=1
            ${provideNodeModules}
            bun build ./index.html --outdir dist
            runHook postBuild
          '';
          installPhase = ''
            runHook preInstall
            cp -r dist $out
            runHook postInstall
          '';
          dontFixup = true;
        };

        webServe = pkgs.writeShellApplication {
          name = "trd-web-serve";
          runtimeInputs = [ pkgs.static-web-server ];
          text = ''
            exec static-web-server --root ${web} --port "''${PORT:-8080}"
          '';
        };
      in
      {
        packages = {
          inherit trd-cli trd-wasm web;
          default = web;
        };

        apps = {
          trd = {
            type = "app";
            program = "${trd-cli}/bin/trd";
          };
          web = {
            type = "app";
            program = "${webServe}/bin/trd-web-serve";
          };
          default = self.apps.${system}.web;
        };

        checks = {
          inherit
            trd-cli
            trd-wasm
            web
            ;

          fmt = craneLib.cargoFmt { inherit src; };

          clippy-native = craneLib.cargoClippy (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--workspace --all-targets -- -D warnings";
            }
          );

          clippy-wasm = craneLib.cargoClippy (
            wasmArgs
            // {
              cargoArtifacts = cargoArtifactsWasm;
              cargoClippyExtraArgs = "-- -D warnings";
            }
          );

          test = craneLib.cargoTest (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoTestExtraArgs = "--workspace";
            }
          );

          # TS type-check using nixpkgs' typescript; the wasm package supplies
          # the `trd-wasm` types via node_modules.
          tsc =
            pkgs.runCommand "check-tsc"
              {
                nativeBuildInputs = [ pkgs.typescript ];
              }
              ''
                cp -r ${checkedWebSrc} web && chmod -R u+w web && cd web
                ${provideNodeModules}
                tsc --noEmit
                touch $out
              '';

          # Biome format-check + lint for the web wrapper.
          biome =
            pkgs.runCommand "check-biome"
              {
                nativeBuildInputs = [ pkgs.biome ];
              }
              ''
                cp -r ${checkedWebSrc} web && chmod -R u+w web && cd web
                biome ci .
                touch $out
              '';
        };

        formatter = pkgs.nixfmt;

        devShells.default = pkgs.mkShell {
          buildInputs =
            with pkgs;
            [
              rustToolchain
              bun
              wasmBindgenCli
              wasm-pack
              binaryen
              biome
              typescript
              static-web-server
              vulkan-loader
              vulkan-tools
              pkg-config
            ]
            ++ runtimeLibs;

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
      }
    );
}
