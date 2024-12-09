#ifndef __YOLOV8S_H__
#define __YOLOV8S_H__

#include <vector>
#include <string>
#include <unistd.h>
#include <memory>

#include "SNPETask.h"
#include "YOLOv8s.h"

struct ObjectData {
    cv::Rect bbox;
    float confidence = -1.0f;
    int label = -1;
    size_t time_cost = 0;
    int index = -1;
    cv::Mat mask;
};

typedef struct _ObjectDetectionConfig {
    std::string model_path;
    runtime_t runtime;
    int labels = 80;
    int grids = 8400;
    std::vector<std::string> inputLayers;
    std::vector<std::string> outputLayers;
    std::vector<std::string> outputTensors;
} ObjectDetectionConfig;

class ObjectDetection {
public:
    ObjectDetection();
    ~ObjectDetection();
    bool Detect(const cv::Mat& image, std::vector<ObjectData>& results);
    bool Initialize(const ObjectDetectionConfig& config);
    bool DeInitialize();

    bool SetScoreThresh(const float& conf_thresh, const float& nms_thresh = 0.5) noexcept {
        this->m_nmsThresh  = nms_thresh;
        this->m_confThresh = conf_thresh;
        return true;
    }

    bool SetMinBoxBorder(uint32_t border = 16) noexcept {
        this->m_minBoxBorder = border;
        return true;
    }


    bool IsInitialized() const {
        return m_isInit;
    }

    static std::vector<ObjectData> nms(std::vector<ObjectData> winList, const float& nms_thresh) {
        if (winList.empty()) {
            return winList;
        }
        std::sort(winList.begin(), winList.end(), [] (const ObjectData& left, const ObjectData& right) {
            if (left.confidence > right.confidence) {
                return true;
            } else {
                return false;
            }
        });
        std::vector<bool> flag(winList.size(), false);
        for (int i = 0; i < winList.size(); i++) {
            if (flag[i]) {
                continue;
            }
            for (int j = i + 1; j < winList.size(); j++) {
                if (calcIoU(winList[i].bbox, winList[j].bbox) > nms_thresh) {
                    flag[j] = true;
                }
            }
        }
        std::vector<ObjectData> ret;
        for (int i = 0; i < winList.size(); i++) {
            if (!flag[i])
                ret.push_back(winList[i]);
        }
        return ret;
    }

private:
    bool m_isInit = false;
    bool m_isRegisteredPreProcess = false;
    bool m_isRegisteredPostProcess = false;

    bool PreProcess(const cv::Mat& frame);
    bool PostProcess(std::vector<ObjectData> &results, int64_t time);
    void get_mask(const cv::Mat& mask_info, const cv::Mat& mask_data, cv::Rect bound, cv::Mat& mast_out);
    
    std::unique_ptr<snpetask::SNPETask> m_task;
    std::vector<std::string> m_inputLayers;
    std::vector<std::string> m_outputLayers;
    std::vector<std::string> m_outputTensors;

    int m_labels;
    int m_grids;
    float* m_output;

    uint32_t m_minBoxBorder = 16;
    float m_nmsThresh = 0.5f;
    float m_confThresh = 0.5f;
    float m_scale;
    int m_xOffset, m_yOffset;
    int orin_cols, orin_rows;
};


#endif // __YOLOV8S_H__
