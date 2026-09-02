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
    bun2nix = {
      url = "github:nix-community/bun2nix/2.1.1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      crane,
      bun2nix,
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
        # `include_wgsl!`), the subset UI fonts (`trd-gui` embeds them via
        # `include_bytes!`, #359) and the golden e2e fixtures/PNGs
        # (`crates/trd-core/tests/golden/`, read at test time via
        # `CARGO_MANIFEST_DIR`); crane's default filter would drop non-Rust files.
        #
        # A file missing here fails *only* under nix: a plain `cargo build` reads
        # the working tree and never notices.
        src = lib.cleanSourceWith {
          src = ./.;
          filter =
            path: type:
            (lib.hasSuffix ".wgsl" path)
            || (lib.hasInfix "/crates/trd-core/tests/golden/" path)
            || (lib.hasInfix "/assets/fonts/" path)
            || (craneLib.filterCargoSources path type);
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

        # Single source of truth for the project version: the Cargo workspace.
        workspaceVersion = (builtins.fromTOML (builtins.readFile ./Cargo.toml)).workspace.package.version;

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
              wrapProgram $out/bin/trd-cli \
                --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath runtimeLibs}
            '';
          }
        );

        # --- Native interactive GUI (eframe/egui) ---------------------------
        # native/trd-gui-app is the interactive front-end (issue #97), built as
        # the public `trd-gui` Nix output. Like trd-cli it dlopens the GPU/window
        # libs (GL/Vulkan/X11/Wayland), so expose the same runtime libraries.
        trd-gui = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            pname = "trd-gui";
            cargoExtraArgs = "--package trd-gui-app";
            doCheck = false;
            # X11/Wayland/xkbcommon dev libs for the eframe/winit build.
            buildInputs = runtimeLibs;
            nativeBuildInputs = commonArgs.nativeBuildInputs ++ [ pkgs.makeWrapper ];
            postInstall = ''
              wrapProgram $out/bin/trd-gui-app \
                --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath runtimeLibs}
            '';
          }
        );

        # Native video-editing eframe shell. ffmpeg/ffprobe provide the native
        # demux/decode adapter; all editor document/state remains in Rust.
        trd-gui-video-editing = craneLib.buildPackage (
          commonArgs
          // {
            inherit cargoArtifacts;
            pname = "trd-gui-video-editing";
            cargoExtraArgs = "--package trd-gui-video-editing";
            doCheck = false;
            buildInputs = runtimeLibs;
            nativeBuildInputs = commonArgs.nativeBuildInputs ++ [ pkgs.makeWrapper ];
            postInstall = ''
              wrapProgram $out/bin/trd-gui-video-editing \
                --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath runtimeLibs} \
                --prefix PATH : ${lib.makeBinPath [ pkgs.ffmpeg ]}
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
                "version": "${workspaceVersion}",
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
        # bun2nix materializes the web workspace's npm dependencies
        # reproducibly from web/bun.nix, so the nix
        # web build and tsc check can run an offline `bun install` in the sandbox
        # instead of the old shortcut of injecting only trd-wasm. (The biome lint
        # runs from nixpkgs' biome instead, see the `biome` check below.)
        bun2nixPkg = bun2nix.packages.${system}.default;

        webBunDeps = bun2nixPkg.fetchBunDeps {
          bunNix = ./web/bun.nix;
        };

        # The viewer resolves `trd-wasm` via
        # `file:../../crates/trd-wasm/pkg`, so the source must retain the repo
        # layout. Keep it lean by dropping build outputs and VCS metadata.
        webSrc = lib.cleanSourceWith {
          src = ./.;
          filter =
            path: type:
            let
              b = baseNameOf path;
            in
            b != "node_modules"
            && b != "dist"
            && b != "pkg"
            && b != "target"
            && b != ".worktree"
            && b != ".git";
          name = "trd-web-src";
        };

        # Shared builder for the bun-driven web derivations (bundle + checks).
        # The bun2nix hook installs node_modules offline in `bunRoot`; the
        # pre-install hook materializes the nix-built wasm artifacts at the
        # package-owned paths referenced by the three web workspace packages.
        mkWebDerivation =
          {
            pname,
            buildCommand,
            installCommand,
          }:
          pkgs.stdenv.mkDerivation {
            inherit pname;
            version = workspaceVersion;
            src = webSrc;
            nativeBuildInputs = [ bun2nixPkg.hook ];
            bunDeps = webBunDeps;
            bunRoot = "web";
            dontUseBunCheck = true;
            # Use bun's standard hoisted node_modules layout instead of the
            # bun2nix hook default `--linker=isolated`, so bundler/tsc module
            # resolution matches a normal `bun install`.
            bunInstallFlags = "--linker=hoisted";

            preBunNodeModulesInstallPhase = ''
              mkdir -p ../crates/trd-wasm/pkg
              cp -r ${trd-wasm}/. ../crates/trd-wasm/pkg/
              chmod -R u+w ../crates/trd-wasm/pkg
              mkdir -p gui-viewer/pkg
              cp -r ${trd-wasm}/. gui-viewer/pkg/
              chmod -R u+w gui-viewer/pkg
              mkdir -p gui-video-editing/pkg
              cp -r ${trd-wasm}/. gui-video-editing/pkg/
              chmod -R u+w gui-video-editing/pkg
            '';

            buildPhase = ''
              runHook preBuild
              cd web/viewer
              ${buildCommand}
              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              ${installCommand}
              runHook postInstall
            '';

            dontFixup = true;
          };

        web = mkWebDerivation {
          pname = "trd-web";
          buildCommand = ''
            bun run build:web
          '';
          installCommand = ''
            mkdir -p $out
            cp -r dist/. $out
          '';
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
          inherit
            trd-cli
            trd-gui
            trd-gui-video-editing
            trd-wasm
            web
            ;
          default = web;
        };

        apps = {
          trd-cli = {
            type = "app";
            program = "${trd-cli}/bin/trd-cli";
          };
          trd-gui = {
            type = "app";
            program = "${trd-gui}/bin/trd-gui-app";
          };
          trd-gui-video-editing = {
            type = "app";
            program = "${trd-gui-video-editing}/bin/trd-gui-video-editing";
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
            trd-gui
            trd-gui-video-editing
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

          # Lints **every** crate that is compiled to wasm, not just the
          # `trd-wasm` delivery surface: `trd-gui` is a plain rlib but is compiled
          # *for* wasm through it, and used to be built for wasm without ever being
          # linted — see #181, where `std::time::Instant` (which panics on
          # `wasm32-unknown-unknown`) reached the browser unnoticed.
          clippy-wasm = craneLib.cargoClippy (
            wasmArgs
            // {
              cargoArtifacts = cargoArtifactsWasm;
              cargoExtraArgs = "--package trd-wasm --package trd-gui --lib";
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

          # Doc links are documentation, and they rot silently: renaming an item
          # turns every `[`Item`]` naming it into plain text, with no warning at
          # any normal gate. Nineteen had accumulated across four crates before
          # this check existed — `RenderMode`, `FrameParams`, `OutputSession`,
          # `crate::app` and the rest — so the fix is only worth as much as the
          # gate that keeps it fixed.
          rustdoc = craneLib.cargoDoc (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoDocExtraArgs = "--workspace --no-deps";
              RUSTDOCFLAGS = "-D rustdoc::broken_intra_doc_links";
            }
          );

          # TS type-check using the project's own typescript (installed offline
          # via bun2nix); the nix-built wasm package supplies the trd-wasm types.
          tsc = mkWebDerivation {
            pname = "check-tsc";
            buildCommand = ''
              cd ..
              bun run typecheck
            '';
            installCommand = ''
              mkdir -p $out
              touch $out/success
            '';
          };

          # Biome format-check + lint for every web package. Biome only parses the
          # source (it never resolves npm imports), so it needs no node_modules
          # and runs straight from nixpkgs' biome. nixpkgs pins the same version
          # as web/viewer/package.json (2.4.16), so the gate matches
          # `bun run check`.
          # (bun2nix can't materialize biome's large optional platform binary
          # @biomejs/cli-linux-x64 into the sandbox node_modules, so we avoid it.)
          biome =
            pkgs.runCommand "check-biome"
              {
                nativeBuildInputs = [ pkgs.biome ];
              }
              ''
                cp -r ${webSrc}/web web
                chmod -R u+w web
                (cd web/viewer && biome ci .)
                (cd web/gui-viewer && biome ci .)
                (cd web/gui-video-editing && biome ci .)
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
              ffmpeg
              uv
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
