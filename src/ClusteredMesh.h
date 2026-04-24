#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <webgpu/webgpu.h>

struct MeshVertex {
    float x, y, z;
    float nx, ny, nz;
    float tu, tv;
};

struct ClusterN {
    int32_t  refined;
    float    groupCenter[3];
    float    groupRadius;
    float    groupError;
    float    refinedCenter[3];
    float    refinedRadius;
    float    refinedError;
    uint32_t meshletVertexOffset;
    uint32_t meshletTriangleOffset;
    uint32_t vertexCount;
    uint32_t triangleCount;
};

struct ClusteredMeshGPU {
    WGPUBuffer vertexBuffer;
    WGPUBuffer clusterBuffer;
    WGPUBuffer meshletVertexBuffer;
    WGPUBuffer meshletTriangleBuffer;
    uint32_t vertexCount;
    uint32_t clusterCount;
    uint32_t meshletVertexCount;
    uint32_t meshletTriangleByteCount;
};

bool loadGltfMeshToGPU(const std::string& path, WGPUDevice device, WGPUQueue queue, ClusteredMeshGPU& out);

void buildNanite(
    const std::vector<MeshVertex>& vertices,
    const std::vector<uint32_t>&   indices,
    std::vector<ClusterN>&         outClusters,
    std::vector<uint32_t>&         outMeshletVertices,
    std::vector<uint8_t>&          outMeshletTriangles);
