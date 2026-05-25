$ErrorActionPreference = "Stop"

$exe = ".\\lab-4\\build\\Release\\path_tracer.exe"
$scene = "lab-4\\scene_simple.obj"
$common = @("--width", "512", "--height", "512", "--spp", "64", "--max-depth", "64")

& $exe --scene $scene --out image_view1.ppm @common --seed 11 --cam-pos 0 1.2 3.5 --cam-look 0.2 0.6 -0.6
& $exe --scene $scene --out image_view2.ppm @common --seed 12 --cam-pos -2.5 1.5 1.5 --cam-look 0.2 0.6 -0.6
& $exe --scene $scene --out image_view3.ppm @common --seed 13 --cam-pos 2.5 1.5 1.5 --cam-look 0.2 0.6 -0.6
& $exe --scene $scene --out image_view4.ppm @common --seed 14 --cam-pos 0 2.5 2.0 --cam-look 0.2 0.6 -0.6
& $exe --scene $scene --out image_view5.ppm @common --seed 15 --cam-pos -2.95 1.7 -2.95 --cam-look 2.8 2.0 -0.333
& $exe --scene $scene --out image_view6.ppm @common --seed 16 --cam-pos 2.95 1.7 -2.95 --cam-look -2.0 1.5 -0.333
