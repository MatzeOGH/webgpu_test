#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <webgpu/webgpu.h>

struct MeshVertex {
    float    x, y, z;
    uint32_t normal; // oct-encoded: low byte = ox (i8), next byte = oy (i8); upper 16 bits unused
    uint32_t uv;     // packed half2 (tu, tv) — unpack with unpack2x16float in WGSL
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
    uint8_t  vertexCount;
    uint8_t  triangleCount;
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

bool buildClusteredMeshFromGltf(
    const std::string&       path,
    std::vector<MeshVertex>& outVertices,
    std::vector<ClusterN>&   outClusters,
    std::vector<uint32_t>&   outMeshletVertices,
    std::vector<uint8_t>&    outMeshletTriangles);

bool saveClusteredMesh(
    const std::string&             path,
    const std::vector<MeshVertex>& vertices,
    const std::vector<ClusterN>&   clusters,
    const std::vector<uint32_t>&   meshletVertices,
    const std::vector<uint8_t>&    meshletTriangles);

bool loadClusteredMeshFromFile(
    const std::string& path,
    WGPUDevice device, WGPUQueue queue,
    ClusteredMeshGPU& out);

bool loadGltfMeshToGPU(const std::string& path, WGPUDevice device, WGPUQueue queue, ClusteredMeshGPU& out);

void buildNanite(
    const std::vector<MeshVertex>& vertices,
    const std::vector<uint32_t>&   indices,
    std::vector<ClusterN>&         outClusters,
    std::vector<uint32_t>&         outMeshletVertices,
    std::vector<uint8_t>&          outMeshletTriangles);
