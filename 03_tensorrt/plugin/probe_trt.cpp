#include "NvInfer.h"

#include <iostream>


class Logger final : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << message << std::endl;
        }
    }
};


int main()
{
    Logger logger;
    auto* runtime = nvinfer1::createInferRuntime(logger);
    if (runtime == nullptr)
    {
        return 1;
    }

    std::cout << "TensorRT C++ SDK probe passed" << std::endl;
    delete runtime;
    return 0;
}
