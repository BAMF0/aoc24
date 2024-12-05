{
  description = "Nix flake for CUDA development";

  inputs = {
    nixpkgs.url = "nixpkgs/unstable";
  };

  outputs = { self, nixpkgs }: {
    devShells.default = nixpkgs.lib.mkShell {
      packages = with nixpkgs.pkgs; [
        cuda                        # CUDA toolkit
        gcc                         # GNU C Compiler
        gdb                         # Debugger
        ninja                       # Optional build system
        cmake                       # CMake for building projects
        nvtop                       # Monitor GPU usage
      ];

      # Optional: Add environment variables needed for CUDA development
      shellHook = ''
        export PATH=${nixpkgs.pkgs.cuda}/bin:$PATH
        export LD_LIBRARY_PATH=${nixpkgs.pkgs.cuda}/lib64:$LD_LIBRARY_PATH
        echo "CUDA environment is ready!"
      '';
    };
  };
}

