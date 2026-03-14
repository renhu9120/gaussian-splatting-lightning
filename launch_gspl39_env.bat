@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.31
set DISTUTILS_USE_SDK=1
set MSSdk=1
conda activate gspl39
cd /d D:\Programming\Python\3DGS\platform_new\gaussian-splatting-lightning
cmd
