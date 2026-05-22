// Cross-platform hardware detection — picks the right implementation
// at compile time. See src/hardware_{macos,linux,windows}.cpp.
#pragma once

#include "modelfit/types.h"

namespace modelfit {

// Probe the current machine using the most appropriate platform API.
HardwareProfile detect_hardware();

// Memory-bandwidth lookups (GB/s) — keep the C++ side in sync with the
// Python `GPU_BANDWIDTH` table.
double gpu_bandwidth_for(const std::string& name);

} // namespace modelfit
