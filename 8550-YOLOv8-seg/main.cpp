#include <YOLOv8s.h>
#include <opencv2/opencv.hpp>
#include <random>

#include <chrono>

long long GetMillisecondTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return millis;
}

cv::Scalar getRandomColor() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 180);

    int r = dis(gen);
    int g = dis(gen);
    int b = dis(gen);

    return cv::Scalar(b, g, r);
}

int main() {
    cv::Mat img = cv::imread("../imgs/frisbee.jpg");
    cv::Mat img2;
    cv::cvtColor(img, img2, cv::COLOR_BGR2RGB);
    ObjectDetection detect;
    ObjectDetectionConfig cfg;
    cfg.model_path = std::string("../models/modified_yolov8s-seg_ver2_quantize_cached.dlc");
    cfg.runtime = runtime::DSP;
    cfg.inputLayers = {"images"};
    cfg.outputLayers = {"/model.22/Sigmoid", "/model.22/Mul_2", "/model.22/Concat", "/model.22/proto/cv3/act/Mul"};
    cfg.outputTensors = {"/model.22/Sigmoid_output_0", "/model.22/Mul_2_output_0", "/model.22/Concat_output_0", "output1"};
    detect.Initialize(cfg);
    std::vector<ObjectData> results;
    auto t0 = GetMillisecondTimestamp();
    detect.Detect(img2, results);
    printf("Dtect cost %lld \n", GetMillisecondTimestamp()-t0);
    // cv::Mat blackImage(img.size(), CV_8UC1, cv::Scalar(0, 0, 0));
    // for (auto i:results) {
    //     cv::bitwise_or(blackImage, i.mask, blackImage);
    // }
    // cv::cvtColor(blackImage, blackImage, cv::COLOR_GRAY2BGR);
    // cv::bitwise_and(blackImage, cv::Scalar(0, 0, 255), blackImage);
    // cv::Mat img_mask;
    // cv::addWeighted(img, 1, blackImage, 0.75, 0, img_mask);
    // cv::imwrite("mask.jpg", img_mask);

    cv::Mat img_mask;
    img.copyTo(img_mask);
    for (auto i:results) {
        cv::Mat mask;
        cv::cvtColor(i.mask, mask, cv::COLOR_GRAY2BGR);
        cv::bitwise_and(mask, getRandomColor(), mask);
        cv::addWeighted(img_mask, 1, mask, 1.2, 0, img_mask);
    
    }
    // cv::cvtColor(blackImage, blackImage, cv::COLOR_GRAY2BGR);
    // cv::bitwise_and(blackImage, cv::Scalar(0, 0, 255), blackImage);
    
    cv::imwrite("mask.jpg", img_mask);
    
    for (auto i:results) {
        cv::putText(img_mask, std::to_string(i.label)+std::string(" : ")+std::to_string(i.confidence), cv::Point(i.bbox.x, i.bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        cv::rectangle(img_mask, cv::Rect(i.bbox.x, i.bbox.y, i.bbox.width, i.bbox.height), cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("result.jpg", img_mask);
    printf("I Img saved result.jpg\n");
    return 0;
}

 