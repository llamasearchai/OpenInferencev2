/**
 * Distributed Manager Header
 * Handles multi-GPU coordination and communication
 */
#pragma once

#include <memory>

namespace openinferencev2 {

class DistributedManager {
public:
    DistributedManager(int num_gpus, int tensor_parallel_size, int pipeline_parallel_size);
    ~DistributedManager();
    
    bool initialize();
    void shutdown();
    bool is_initialized() const;
    
    int get_rank() const;
    int get_world_size() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
};

} // namespace openinferencev2 