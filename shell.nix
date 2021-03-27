with import <nixpkgs> {};
# packageOverrides = super: let self = super.pkgs; in
let myPythonEnv = python38.withPackages
                     (ps: with ps; [
                       # libraries
                       ipython
                       scikitimage
                       opencv4
                       scipy
                     ]);
in
stdenv.mkDerivation rec{
  name="python-webcam";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
    myPythonEnv
    v4l-utils
    black
  ];
}
