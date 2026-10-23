#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <Vision/Vision.h>

#include "mediapipe_hand_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

using namespace mediapipe_hands;

namespace {

struct SharedHandState {
    std::mutex mtx;
    HandTrackingState state;
    std::array<uint32_t, CAM_PREVIEW_W * CAM_PREVIEW_H> pixels;
    float smooth_rx = 0.5f;
    float smooth_ry = 0.5f;
    bool prev_pinch = false;

    SharedHandState() {
        pixels.fill(0xEAE7E0FFu);
    }
};

static float dist2D(const NormalizedLandmark &a, const NormalizedLandmark &b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static int countExtendedFingers(const std::array<NormalizedLandmark, NUM_LANDMARKS> &pts) {
    const int tips[4] = {INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP};
    const int pips[4] = {INDEX_FINGER_PIP, MIDDLE_FINGER_PIP, RING_FINGER_PIP, PINKY_PIP};
    const int mcps[4] = {INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP};

    int count = 0;
    const auto &wrist = pts[WRIST];
    for (int f = 0; f < 4; ++f) {
        const auto &tip = pts[tips[f]];
        const auto &pip = pts[pips[f]];
        const auto &mcp = pts[mcps[f]];
        if (tip.confidence > 0.2f && pip.confidence > 0.2f) {
            float dTip = dist2D(wrist, tip);
            float dPip = dist2D(wrist, pip);
            float dMcp = dist2D(wrist, mcp);
            if (dTip > dPip * 1.14f && dTip > dMcp * 1.32f && tip.y < mcp.y + 0.05f) {
                count++;
            }
        }
    }
    return count;
}

} // namespace

@interface MPHandCaptureDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate> {
@public
    SharedHandState *shared_;
    VNDetectHumanHandPoseRequest *handRequest_;
    double lastInferenceTime_;
}
- (instancetype)initWithShared:(SharedHandState *)shared;
@end

@implementation MPHandCaptureDelegate

- (instancetype)initWithShared:(SharedHandState *)shared {
    self = [super init];
    if (self) {
        shared_ = shared;
        lastInferenceTime_ = 0.0;
        if (@available(macOS 11.0, *)) {
            handRequest_ = [[VNDetectHumanHandPoseRequest alloc] init];
            handRequest_.maximumHandCount = 2;
        }
    }
    return self;
}

- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
    (void)output;
    (void)connection;

    // CRITICAL: Drain all Objective-C / Vision / CoreML / IOSurface objects every frame!
    @autoreleasepool {
        CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        if (!pixelBuffer) return;

        // 1. Update 320x240 preview buffer in-place (zero heap allocations)
        CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        size_t srcW = CVPixelBufferGetWidth(pixelBuffer);
        size_t srcH = CVPixelBufferGetHeight(pixelBuffer);
        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
        const uint8_t *baseAddr = (const uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);

        if (baseAddr && srcW > 0 && srcH > 0) {
            std::lock_guard<std::mutex> lock(shared_->mtx);
            uint32_t *dst = shared_->pixels.data();
            for (int dy = 0; dy < CAM_PREVIEW_H; ++dy) {
                size_t sy = (dy * srcH) / CAM_PREVIEW_H;
                const uint8_t *row = baseAddr + sy * bytesPerRow;
                uint32_t *dstRow = dst + dy * CAM_PREVIEW_W;
                for (int dx = 0; dx < CAM_PREVIEW_W; ++dx) {
                    size_t sx = ((CAM_PREVIEW_W - 1 - dx) * srcW) / CAM_PREVIEW_W;
                    const uint8_t *px = row + sx * 4;
                    uint8_t b = px[0], g = px[1], r = px[2];
                    float luma = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
                    float tone = 38.0f + luma * 202.0f;
                    uint32_t dr = (uint32_t)std::clamp(tone + 4.0f, 0.0f, 255.0f);
                    uint32_t dg = (uint32_t)std::clamp(tone + 1.0f, 0.0f, 255.0f);
                    uint32_t db = (uint32_t)std::clamp(tone - 5.0f, 0.0f, 255.0f);
                    dstRow[dx] = (dr << 24) | (dg << 16) | (db << 8) | 0xFFu;
                }
            }
            shared_->state.camera_ready = true;
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

        // 2. Throttle Vision Hand Pose inference to ~15 FPS (every 66ms) to keep CPU/RAM tiny
        double now = CACurrentMediaTime();
        if (now - lastInferenceTime_ < 0.066) {
            return;
        }
        lastInferenceTime_ = now;

        SingleHandState leftHand{};
        SingleHandState rightHand{};

        if (@available(macOS 11.0, *)) {
            if (handRequest_) {
                @autoreleasepool {
                    VNImageRequestHandler *handler =
                        [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBuffer options:@{}];
                    NSError *err = nil;
                    [handler performRequests:@[ handRequest_ ] error:&err];

                    if (!err && handRequest_.results.count > 0) {
                        struct JointMapping {
                            LandmarkId id;
                            VNHumanHandPoseObservationJointName name;
                        };
                        static const JointMapping map[21] = {
                            {WRIST, VNHumanHandPoseObservationJointNameWrist},
                            {THUMB_CMC, VNHumanHandPoseObservationJointNameThumbCMC},
                            {THUMB_MCP, VNHumanHandPoseObservationJointNameThumbMP},
                            {THUMB_IP, VNHumanHandPoseObservationJointNameThumbIP},
                            {THUMB_TIP, VNHumanHandPoseObservationJointNameThumbTip},
                            {INDEX_FINGER_MCP, VNHumanHandPoseObservationJointNameIndexMCP},
                            {INDEX_FINGER_PIP, VNHumanHandPoseObservationJointNameIndexPIP},
                            {INDEX_FINGER_DIP, VNHumanHandPoseObservationJointNameIndexDIP},
                            {INDEX_FINGER_TIP, VNHumanHandPoseObservationJointNameIndexTip},
                            {MIDDLE_FINGER_MCP, VNHumanHandPoseObservationJointNameMiddleMCP},
                            {MIDDLE_FINGER_PIP, VNHumanHandPoseObservationJointNameMiddlePIP},
                            {MIDDLE_FINGER_DIP, VNHumanHandPoseObservationJointNameMiddleDIP},
                            {MIDDLE_FINGER_TIP, VNHumanHandPoseObservationJointNameMiddleTip},
                            {RING_FINGER_MCP, VNHumanHandPoseObservationJointNameRingMCP},
                            {RING_FINGER_PIP, VNHumanHandPoseObservationJointNameRingPIP},
                            {RING_FINGER_DIP, VNHumanHandPoseObservationJointNameRingDIP},
                            {RING_FINGER_TIP, VNHumanHandPoseObservationJointNameRingTip},
                            {PINKY_MCP, VNHumanHandPoseObservationJointNameLittleMCP},
                            {PINKY_PIP, VNHumanHandPoseObservationJointNameLittlePIP},
                            {PINKY_DIP, VNHumanHandPoseObservationJointNameLittleDIP},
                            {PINKY_TIP, VNHumanHandPoseObservationJointNameLittleTip},
                        };

                        for (VNHumanHandPoseObservation *obs in handRequest_.results) {
                            NSError *ptErr = nil;
                            NSDictionary<VNHumanHandPoseObservationJointName, VNRecognizedPoint *> *allPts =
                                [obs recognizedPointsForJointsGroupName:VNHumanHandPoseObservationJointsGroupNameAll
                                                                  error:&ptErr];
                            if (ptErr || !allPts) continue;

                            SingleHandState hs{};
                            int validCount = 0;
                            for (const auto &m : map) {
                                VNRecognizedPoint *rp = allPts[m.name];
                                if (rp && rp.confidence > 0.2f) {
                                    hs.landmarks[m.id].x = std::clamp(1.0f - (float)rp.location.x, 0.0f, 1.0f);
                                    hs.landmarks[m.id].y = std::clamp(1.0f - (float)rp.location.y, 0.0f, 1.0f);
                                    hs.landmarks[m.id].confidence = (float)rp.confidence;
                                    validCount++;
                                }
                            }
                            if (validCount >= 8) {
                                hs.detected = true;
                                hs.open_fingers = countExtendedFingers(hs.landmarks);
                                hs.is_open = (hs.open_fingers >= 3);

                                float palm_cx = hs.landmarks[MIDDLE_FINGER_MCP].x * 0.7f + hs.landmarks[WRIST].x * 0.3f;
                                float palm_cy = hs.landmarks[MIDDLE_FINGER_MCP].y * 0.7f + hs.landmarks[WRIST].y * 0.3f;

                                // Strictly assign by Left vs Right Quadrant Box
                                if (palm_cx < 0.50f) {
                                    hs.norm_x = std::clamp(palm_cx, 0.0f, 1.0f);
                                    hs.norm_y = std::clamp(1.0f - palm_cy, 0.0f, 1.0f);
                                    if (!leftHand.detected) leftHand = hs;
                                } else {
                                    // For Right Hand: use midpoint between THUMB_TIP (4) and INDEX_FINGER_TIP (8)
                                    const auto &th = hs.landmarks[THUMB_TIP];
                                    const auto &ix = hs.landmarks[INDEX_FINGER_TIP];
                                    float px = (th.confidence > 0.2f && ix.confidence > 0.2f)
                                                   ? 0.5f * (th.x + ix.x)
                                                   : palm_cx;
                                    float py = (th.confidence > 0.2f && ix.confidence > 0.2f)
                                                   ? 0.5f * (th.y + ix.y)
                                                   : palm_cy;
                                    hs.norm_x = std::clamp(px, 0.0f, 1.0f);
                                    hs.norm_y = std::clamp(1.0f - py, 0.0f, 1.0f);

                                    // Scale-normalized pinch distance between thumb tip (4) and index tip (8)
                                    float palm_scale = std::max(0.05f, dist2D(hs.landmarks[WRIST], hs.landmarks[MIDDLE_FINGER_MCP]));
                                    float pinch_dist = dist2D(th, ix);
                                    float pinch_ratio = pinch_dist / palm_scale;
                                    float thresh = shared_->prev_pinch ? 0.56f : 0.38f;
                                    hs.is_pinching = (th.confidence > 0.2f && ix.confidence > 0.2f) &&
                                                     (pinch_ratio < thresh || pinch_dist < (shared_->prev_pinch ? 0.068f : 0.048f));

                                    if (!rightHand.detected) rightHand = hs;
                                }
                            }
                        }
                    }
                }
            }
        }

        std::lock_guard<std::mutex> lock(shared_->mtx);
        shared_->state.left_hand = leftHand;
        if (rightHand.detected) {
            shared_->smooth_rx += 0.42f * (rightHand.norm_x - shared_->smooth_rx);
            shared_->smooth_ry += 0.42f * (rightHand.norm_y - shared_->smooth_ry);
            rightHand.norm_x = std::clamp(shared_->smooth_rx, 0.0f, 1.0f);
            rightHand.norm_y = std::clamp(shared_->smooth_ry, 0.0f, 1.0f);
        }
        shared_->prev_pinch = rightHand.detected ? rightHand.is_pinching : false;
        shared_->state.right_hand = rightHand;
    }
}

@end

namespace mediapipe_hands {

struct HandTracker::Impl {
    SharedHandState shared;
    AVCaptureSession *session = nil;
    MPHandCaptureDelegate *delegate = nil;
    dispatch_queue_t queue = nil;
};

HandTracker::HandTracker() : impl_(new Impl()) {}

HandTracker::~HandTracker() {
    stop();
    delete impl_;
}

bool HandTracker::start() {
    if (impl_->session) return true;

    @autoreleasepool {
        @try {
            AVAuthorizationStatus authStatus =
                [AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo];
            if (authStatus == AVAuthorizationStatusDenied ||
                authStatus == AVAuthorizationStatusRestricted) {
                return false;
            }

            AVCaptureDevice *device = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
            if (!device) return false;

            NSError *err = nil;
            AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&err];
            if (err || !input) return false;

            impl_->session = [[AVCaptureSession alloc] init];
            if ([impl_->session canSetSessionPreset:AVCaptureSessionPreset640x480]) {
                impl_->session.sessionPreset = AVCaptureSessionPreset640x480;
            }

            if ([impl_->session canAddInput:input]) {
                [impl_->session addInput:input];
            } else {
                impl_->session = nil;
                return false;
            }

            AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
            output.alwaysDiscardsLateVideoFrames = YES;
            output.videoSettings = @{
                (NSString *)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
            };

            impl_->delegate = [[MPHandCaptureDelegate alloc] initWithShared:&impl_->shared];
            impl_->queue = dispatch_queue_create("com.biomorphic.mediapipe.handqueue", DISPATCH_QUEUE_SERIAL);
            [output setSampleBufferDelegate:impl_->delegate queue:impl_->queue];

            if ([impl_->session canAddOutput:output]) {
                [impl_->session addOutput:output];
            } else {
                impl_->session = nil;
                return false;
            }

            [impl_->session startRunning];
            return true;
        } @catch (NSException *exception) {
            impl_->session = nil;
            return false;
        }
    }
}

void HandTracker::stop() {
    @autoreleasepool {
        if (impl_->session) {
            @try {
                [impl_->session stopRunning];
            } @catch (NSException *e) {
            }
            impl_->session = nil;
        }
        impl_->delegate = nil;
    }
}

HandTrackingState HandTracker::getStateAndPixels(uint32_t *out_rgba_320x240) {
    std::lock_guard<std::mutex> lock(impl_->shared.mtx);
    if (out_rgba_320x240 && impl_->shared.state.camera_ready) {
        std::memcpy(out_rgba_320x240,
                    impl_->shared.pixels.data(),
                    CAM_PREVIEW_W * CAM_PREVIEW_H * sizeof(uint32_t));
    }
    return impl_->shared.state;
}

} // namespace mediapipe_hands
