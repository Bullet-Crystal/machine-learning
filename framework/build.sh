#!/bin/sh
set -xe
CXXFLAGS="-std=c++17 -O3 -march=native -DNDEBUG -Iimgui $(pkg-config --cflags sdl3)"

mkdir -p build
for f in imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp \
         imgui/backends/imgui_impl_sdl3.cpp imgui/backends/imgui_impl_opengl3.cpp; do
    o=build/$(basename $f .cpp).o
    [ "$o" -nt "$f" ] || g++ $CXXFLAGS -c $f -o $o
done

g++ $CXXFLAGS nn.cpp build/*.o $(pkg-config --libs sdl3) -lGL -o app
