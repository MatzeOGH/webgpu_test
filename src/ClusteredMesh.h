#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <webgpu/webgpu.h>

// Packed runtime/on-disk/GPU layout. Must match the WGSL `struct MeshVertex`.
// pos[2]: 21 bits per axis packed as x[0..20] | y[0..10] in word0; y[11..20] | z[0..20] in word1
// uv    : packed half2 (tu, tv) — unpack with unpack2x16float in WGSL
// normal: oct-encoded, byte 0 = ox (i8), byte 1 = oy (i8)
struct MeshVertex {
    uint32_t pos[2];
    uint32_t uv;
    uint16_t normal;
};
static_assert(sizeof(MeshVertex) == 16, "MeshVertex size must match WGSL layout");

// Uncompressed in-memory vertex used while loading and processing.
// Encoded into MeshVertex only at save time.
struct MeshVertexRaw {
    float pos[3];
    float normal[3];
    float uv[2];
};

// GPU/on-disk cluster. Must match the WGSL `struct ClusterN` exactly (56 bytes).
struct ClusterN {
    int32_t  refined;
    float    groupCenterX, groupCenterY, groupCenterZ;
    float    groupRadius, groupError;
    float    refinedCenterX, refinedCenterY, refinedCenterZ;
    float    refinedRadius, refinedError;
    uint32_t meshletVertexOffset;   // index into meshletVertex buffer (global vertex index list)
    uint32_t meshletTriangleOffset; // byte index into meshletTriangle buffer (uint8_t local indices)
    uint8_t  vertexCount;
    uint8_t  triangleCount;
    // 2 bytes implicit padding — WGSL reads all four bytes as packedCounts : u32
};
static_assert(sizeof(ClusterN) == 56, "ClusterN size must match WGSL layout");

struct ClusteredMeshGPU {
    WGPUBuffer vertexBuffer;
    WGPUBuffer clusterBuffer;
    WGPUBuffer meshletVertexBuffer;   // uint32_t global vertex indices, one per cluster-local slot
    WGPUBuffer meshletTriangleBuffer; // uint8_t local triangle indices
    uint32_t vertexCount;
    uint32_t clusterCount;
    uint32_t meshletVertexCount;      // total entries in meshletVertexBuffer
    uint32_t meshletTriangleByteCount;
    float dequant_summand[3]; // world-space bounding box min
    float dequant_factor[3];  // (max - min) / (2^21 - 1) per axis
};

struct LodGroup {
    float    center[3];
    float    radius;
    float    error;
    uint32_t clusterOffset;   // first index into outClusters produced by this group
    uint32_t clusterCount;    // number of clusters produced by this group
    uint32_t depth;           // DAG level from clodGroup::depth (0 = finest)
};

static constexpr uint32_t kBvhLeafBit = 0x80000000u;

// 40-byte N-ary BVH node. sphere.xyz=center, sphere.w=radius.
// Internal: childOffset = first child node index (MSB clear), childCount = # children.
// Leaf:     childOffset = group index | kBvhLeafBit, childCount = 1.
// depth:    LOD level this node belongs to (0 = finest); UINT32_MAX for the global root.
struct BvhNode {
    float    sphere[4];
    float    minError;
    float    maxError;
    uint32_t childOffset;
    uint32_t childCount;
    uint32_t depth;
    uint32_t _pad;
};
static_assert(sizeof(BvhNode) == 40, "BvhNode must be 40 bytes");

// Loads and clusters the glTF. outVertices is the unique global vertex pool;
// outMeshletVertices holds per-cluster global vertex indices (no duplication).
bool buildClusteredMeshFromGltf(
    const std::string&           path,
    std::vector<MeshVertexRaw>&  outVertices,
    std::vector<ClusterN>&       outClusters,
    std::vector<uint32_t>&       outMeshletVertices,
    std::vector<uint8_t>&        outMeshletTriangles,
    std::vector<LodGroup>&       outGroups);

// Encodes and writes the mesh file. Quantizes vertex positions only.
bool saveClusteredMesh(
    const std::string&                path,
    const std::vector<MeshVertexRaw>& vertices,
    const std::vector<ClusterN>&      clusters,
    const std::vector<uint32_t>&      meshletVertices,
    const std::vector<uint8_t>&       meshletTriangles);

bool loadClusteredMeshFromFile(
    const std::string& path,
    WGPUDevice device, WGPUQueue queue,
    ClusteredMeshGPU& out);

bool loadGltfMeshToGPU(const std::string& path, WGPUDevice device, WGPUQueue queue, ClusteredMeshGPU& out);

// Builds the Nanite cluster hierarchy. outMeshletVertices receives the global
// vertex indices for each cluster's local slots (no vertex data duplication).
void buildNanite(
    const std::vector<MeshVertexRaw>& vertices,
    const std::vector<uint32_t>&      indices,
    std::vector<ClusterN>&            outClusters,
    std::vector<uint32_t>&            outMeshletVertices,
    std::vector<uint8_t>&             outMeshletTriangles,
    std::vector<LodGroup>&            outGroups);

// Builds a per-LOD-level BVH over the cluster groups. First node (index 0) is the root.
void buildBvh(const std::vector<LodGroup>& groups,
              std::vector<BvhNode>&        outNodes);

// Writes the BVH as a Graphviz dot graph. Pass stdout, stderr, or any opened FILE*.
// Pipe to `dot -Tsvg` or paste into https://dreampuf.github.io/GraphvizOnline.
void writeBvhDotGraph(const std::vector<BvhNode>& nodes, std::FILE* out);
