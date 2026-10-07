#!/bin/sh

set -xe
g++ -std=c++17 nn_xor.cpp imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_tables.cpp imgui/imgui_widgets.cpp imgui/backends/imgui_impl_sdl3.cpp imgui/backends/imgui_impl_opengl3.cpp -Iimgui \
	$(pkg-config --cflags --libs sdl3) -lGL -o app_xor
./app_xor
