{
  description = "Nix flake for CUDA development";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShell = with pkgs; mkShell {
          allowUnfree = true;
          packages = [
             autoconf curl
             procps gnumake util-linux m4 gperf unzip
             cudaPackages.cudatoolkit cudaPackages.cudnn cudaPackages.cuda_cudart linuxPackages.nvidia_x11
             libGLU libGL
             xorg.libXi xorg.libXmu freeglut
             xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
             ncurses5 stdenv.cc
         ];
         shellHook = ''
            export CUDA_PATH=${pkgs.cudatoolkit}
            export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib:$CUDA_PATH/lib:$LD_LIBRARY_PATH
            export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib -L/${pkgs.cudatoolkit}/lib"
            export EXTRA_CCFLAGS="-I/usr/include"
         '';
        };
      }
    );
}

