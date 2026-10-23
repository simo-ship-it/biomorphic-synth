#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <utility>

namespace mediapipe_hands {

static constexpr int CAM_PREVIEW_W = 320;
static constexpr int CAM_PREVIEW_H = 240;

enum LandmarkId : int {
    WRIST = 0,
    THUMB_CMC = 1,
    THUMB_MCP = 2,
    THUMB_IP = 3,
    THUMB_TIP = 4,
    INDEX_FINGER_MCP = 5,
    INDEX_FINGER_PIP = 6,
    INDEX_FINGER_DIP = 7,
    INDEX_FINGER_TIP = 8,
    MIDDLE_FINGER_MCP = 9,
    MIDDLE_FINGER_PIP = 10,
    MIDDLE_FINGER_DIP = 11,
    MIDDLE_FINGER_TIP = 12,
    RING_FINGER_MCP = 13,
    RING_FINGER_PIP = 14,
    RING_FINGER_DIP = 15,
    RING_FINGER_TIP = 16,
    PINKY_MCP = 17,
    PINKY_PIP = 18,
    PINKY_DIP = 19,
    PINKY_TIP = 20,
    NUM_LANDMARKS = 21
};

inline constexpr std::pair<int, int> HAND_CONNECTIONS[21] = {
    {0, 1},   {1, 2},   {2, 3},   {3, 4},     // Thumb
    {0, 5},   {5, 6},   {6, 7},   {7, 8},     // Index finger
    {5, 9},   {9, 10},  {10, 11}, {11, 12},   // Middle finger
    {9, 13},  {13, 14}, {14, 15}, {15, 16},   // Ring finger
    {13, 17}, {0, 17},  {17, 18}, {18, 19}, {19, 20} // Pinky & palm base
};

struct NormalizedLandmark {
    float x = 0.5f;
    float y = 0.5f;
    float confidence = 0.0f;
};

struct SingleHandState {
    bool detected = false;
    std::array<NormalizedLandmark, NUM_LANDMARKS> landmarks{};
    int open_fingers = 0;
    bool is_open = false;
    bool is_pinching = false; // True when THUMB_TIP (4) and INDEX_FINGER_TIP (8) touch
    float norm_x = 0.5f;
    float norm_y = 0.5f;
};

struct HandTrackingState {
    bool camera_ready = false;
    SingleHandState left_hand;
    SingleHandState right_hand;
};

class HandTracker {
public:
    HandTracker();
    ~HandTracker();

    bool start();
    void stop();

    // Copies only the lightweight struct (few hundred bytes) and copies pixels into caller's preallocated buffer
    HandTrackingState getStateAndPixels(uint32_t *out_rgba_320x240);

private:
    struct Impl;
    Impl *impl_ = nullptr;
};

} // namespace mediapipe_hands
