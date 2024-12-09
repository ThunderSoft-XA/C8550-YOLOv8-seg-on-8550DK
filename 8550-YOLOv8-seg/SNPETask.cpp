#include "SNPETask.h"

namespace snpetask{

static size_t calcSizeFromDims(const zdl::DlSystem::Dimension* dims, size_t rank, size_t elementSize)
{
    if (rank == 0) return 0;
    size_t size = elementSize;
    while (rank--) {
        size *= *dims;
        dims++;
    }
    return size;
}


static void createUserBuffer(zdl::DlSystem::UserBufferMap& userBufferMap,
                      std::unordered_map<std::string, float*>& applicationBuffers,
                      std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>>& snpeUserBackedBuffers,
                      const zdl::DlSystem::TensorShape& bufferShape,
                      const char* name)
{
    std::vector<size_t> strides(bufferShape.rank());
    strides[strides.size() - 1] = sizeof(float);
    size_t stride = strides[strides.size() - 1];
    for (size_t i = bufferShape.rank() - 1; i > 0; i--)
    {
        stride *= bufferShape[i];
        strides[i - 1] = stride;
    }
    // const size_t bufferElementSize = sizeof(float);
    size_t bufSize = calcSizeFromDims(bufferShape.getDimensions(), bufferShape.rank(), 1);
    float* buffer = new float[bufSize];

    // set the buffer encoding type
    zdl::DlSystem::UserBufferEncodingFloat userBufferEncodingFloat;
    // create user-backed storage to load input data onto it
    applicationBuffers.emplace(name, buffer);
    // create SNPE user buffer from the user-backed buffer
    zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
    snpeUserBackedBuffers.push_back(ubFactory.createUserBuffer(applicationBuffers.at(name),
                                                                bufSize,
                                                                strides,
                                                                &userBufferEncodingFloat));
    // add the user-backed buffer to the inputMap, which is later on fed to the network for execution
    userBufferMap.add(name, snpeUserBackedBuffers.back().get());
}

SNPETask::SNPETask()
{
    static zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    printf("INFO : Using SNPE: %s\n", version.asString().c_str());
}

SNPETask::~SNPETask()
{

}

bool SNPETask::init(const std::string& model_path, const runtime_t runtime)
{
    switch (runtime) {
        case CPU:
            m_runtime = zdl::DlSystem::Runtime_t::CPU;
            break;
        case GPU:
            m_runtime = zdl::DlSystem::Runtime_t::GPU;
            break;
        case GPU_FLOAT16:
            m_runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
            break;
        case DSP:
            m_runtime = zdl::DlSystem::Runtime_t::DSP;
            break;
        case AIP:
            m_runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
            break;
        default:
            m_runtime = zdl::DlSystem::Runtime_t::CPU;
            break;
    }

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(m_runtime)) {
        printf("ERROR: Selected runtime not present. Falling back to CPU.\n");
        m_runtime = zdl::DlSystem::Runtime_t::CPU;
    }

    zdl::DlSystem::PerformanceProfile_t profile = zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE;

    m_container = zdl::DlContainer::IDlContainer::open(model_path);

    zdl::DlSystem::RuntimeList runtimeList;
    runtimeList.add(m_runtime);
    // runtimeList.add(zdl::DlSystem::Runtime_t::DSP);
    runtimeList.add(zdl::DlSystem::Runtime_t::CPU);

    zdl::SNPE::SNPEBuilder snpeBuilder(m_container.get());
    m_snpe = snpeBuilder.setOutputLayers(m_outputLayers)
       .setRuntimeProcessorOrder(runtimeList)
       .setPerformanceProfile(profile)
       .setUseUserSuppliedBuffers(false)
       .build();

    if (nullptr == m_snpe.get()) {
        const char* errStr = zdl::DlSystem::getLastErrorString();
        printf("ERROR: SNPE build failed: %s\n", errStr);
        return false;
    }

    // get input tensor names of the network that need to be populated
    const auto& inputNamesOpt = m_snpe->getInputTensorNames();
    if (!inputNamesOpt) throw std::runtime_error("Error obtaining input tensor names");
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;

    // create SNPE user buffers for each application storage buffer
    for (const char* name : inputNames) {
        // get attributes of buffer by name
        auto bufferAttributesOpt = m_snpe->getInputOutputBufferAttributes(name);
        if (!bufferAttributesOpt) {
            printf("ERROR: Error obtaining attributes for input tensor: %s\n", name);
            return false;
        }

        const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
        std::vector<size_t> tensorShape;
        for (size_t j = 0; j < bufferShape.rank(); j++) {
            tensorShape.push_back(bufferShape[j]);
        }
        m_inputShapes.emplace(name, tensorShape);

        createUserBuffer(m_inputUserBufferMap, m_inputTensors, m_inputUserBuffers, bufferShape, name);
    }

    // get output tensor names of the network that need to be populated
    const auto& outputNamesOpt = m_snpe->getOutputTensorNames();
    if (!outputNamesOpt) throw std::runtime_error("Error obtaining output tensor names");
    const zdl::DlSystem::StringList& outputNames = *outputNamesOpt;

    // create SNPE user buffers for each application storage buffer
    for (const char* name : outputNames) {
        // get attributes of buffer by name
        auto bufferAttributesOpt = m_snpe->getInputOutputBufferAttributes(name);
        if (!bufferAttributesOpt) {
            printf("ERROR: Error obtaining attributes for input tensor: %s\n", name);
            return false;
        }
        // printf("!!! Get Output [%s] \n", name);

        const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims();
        std::vector<size_t> tensorShape;
        for (size_t j = 0; j < bufferShape.rank(); j++) {
            tensorShape.push_back(bufferShape[j]);
        }
        m_outputShapes.emplace(name, tensorShape);

        createUserBuffer(m_outputUserBufferMap, m_outputTensors, m_outputUserBuffers, bufferShape, name);
    }

    m_isInit = true;

    return true;
}

bool SNPETask::deInit()
{
    if (nullptr != m_snpe) {
        m_snpe.reset(nullptr);
    }

    for (auto [k, v] : m_inputTensors) delete [] v;
    for (auto [k, v] : m_outputTensors) delete [] v;
    return true;
}

bool SNPETask::setOutputLayers(std::vector<std::string>& outputLayers)
{
    for (size_t i = 0; i < outputLayers.size(); i ++) {
        m_outputLayers.append(outputLayers[i].c_str());
    }

    return true;
}

std::vector<size_t> SNPETask::getInputShape(const std::string& name)
{
    if (isInit()) {
        if (m_inputShapes.find(name) != m_inputShapes.end()) {
            return m_inputShapes.at(name);
        }
        printf("ERROR: Can't find any input layer named %s\n", name.c_str());
        return {};
    } else {
        printf("ERROR: The getInputShape() needs to be called after AICContext is initialized!\n");
        return {};
    }
}

std::vector<size_t> SNPETask::getOutputShape(const std::string& name)
{
    // printf("!!! getOutputShape get name [%s]\n", name.c_str());
    if (isInit()) {
        if (m_outputShapes.find(name) != m_outputShapes.end()) {
            return m_outputShapes.at(name);
        }
        printf("ERROR: Can't find any ouput layer named %s\n", name.c_str());
        return {};
    } else {
        printf("ERROR: The getOutputShape() needs to be called after AICContext is initialized!\n");
        return {};
    }
}

float* SNPETask::getInputTensor(const std::string& name)
{
    if (isInit()) {
        if (m_inputTensors.find(name) != m_inputTensors.end()) {
            return m_inputTensors.at(name);
        }
        printf("ERROR: Can't find any input tensor named %s\n", name.c_str());
        return nullptr;
    } else {
        printf("ERROR: The getInputTensor() needs to be called after AICContext is initialized!\n");
        return nullptr;
    }
}

float* SNPETask::getOutputTensor(const std::string& name)
{
    if (isInit()) {
        if (m_outputTensors.find(name) != m_outputTensors.end()) {
            return m_outputTensors.at(name);
        }
        printf("ERROR: Can't find any output tensor named %s\n", name.c_str());
        return nullptr;
    } else {
        printf("ERROR: The getOutputTensor() needs to be called after AICContext is initialized!\n");
        return nullptr;
    }
}

bool SNPETask::execute()
{
    // if (!m_snpe->execute(m_inputUserBufferMap, m_outputUserBufferMap)) {
    //     printf("ERROR: SNPETask execute failed: %s\n", zdl::DlSystem::getLastErrorString());
    //     return false;
    // }
    // return true;

    std::unique_ptr<zdl::DlSystem::ITensor> inputTensor;
    inputTensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(m_inputShapes["images"]);
    std::copy(m_inputTensors["images"], m_inputTensors["images"]+640*640*3, inputTensor->begin());

    zdl::DlSystem::TensorMap outputTensorMap;

    if (!m_snpe->execute(inputTensor.get(), outputTensorMap)) {
        printf("ERROR:SNPETask execute failed: %s\n", zdl::DlSystem::getLastErrorString());
        return false;
    }

    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
    for(auto& name : tensorNames)
    {
        int ptr = 0;
        auto tensorPtr = outputTensorMap.getTensor(name);
        for (auto i:*(tensorPtr)) {
            *(m_outputTensors[std::string(name)] + ptr) = i;
            ptr++;
        }
    }

    return true;
}

}   // namespace snpetask