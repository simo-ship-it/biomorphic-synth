cmake_minimum_required(VERSION 3.16)
project(BiomorphicHandSynth LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD 20)
add_compile_options(-fobjc-arc)

add_executable(biomorphic_hand_synth
    main.mm
    mediapipe_hand_tracker.mm
)

target_include_directories(biomorphic_hand_synth PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(biomorphic_hand_synth PRIVATE
    "-framework Cocoa"
    "-framework QuartzCore"
    "-framework AudioToolbox"
    "-framework AVFoundation"
    "-framework Vision"
    "-framework CoreMedia"
    "-framework CoreVideo"
    "-framework Foundation"
    "-Wl,-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_SOURCE_DIR}/../Info.plist"
)
