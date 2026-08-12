{
  description = "whatisit: natural language to shell command, fully local";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Keep this in sync with whatisit_pkg/pyproject.toml.
      version = "0.2.1";

      eachSystem = f: nixpkgs.lib.genAttrs
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ]
        (system: f system (nixpkgs.legacyPackages.${system}));
    in
    {
      packages = eachSystem (system: pkgs: {
        whatisit = pkgs.python3.pkgs.buildPythonApplication {
          pname = "whatisit";
          inherit version;
          src = ./whatisit_pkg;
          format = "pyproject";

          nativeBuildInputs = [ pkgs.makeWrapper pkgs.python3.pkgs.setuptools ];
          nativeCheckInputs = [ pkgs.python3.pkgs.pytestCheckHook ];

          # The model (941 MB) is deliberately NOT bundled: fetch it once with
          # `whatisit setup --model /path/to/model.gguf`. The runtime comes
          # from Nix instead of the tool's own downloader.
          postInstall = ''
            wrapProgram $out/bin/whatisit \
              --set WHATISIT_LLAMA_SERVER ${pkgs.llama-cpp}/bin/llama-server \
              --set WHATISIT_LLAMA_CLI ${pkgs.llama-cpp}/bin/llama-cli
          '';
        };
        default = self.packages.${system}.whatisit;
      });

      devShells = eachSystem (system: pkgs: {
        default = pkgs.mkShell {
          packages = [
            self.packages.${system}.whatisit
            (pkgs.python3.withPackages (ps: [ ps.pytest ps.pytest-cov ps.ruff ps.build ]))
            pkgs.llama-cpp
          ];
        };
      });
    };
}
