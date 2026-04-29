#include "ClusteredMesh.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "AlpUtils.h"

static inline void octDecodeDir(int8_t ox, int8_t oy, float out[3])
{
    float nx = float(ox) / 127.0f;
    float ny = float(oy) / 127.0f;
    float nz = 1.0f - fabsf(nx) - fabsf(ny);
    if (nz < 0.0f)
    {
        float tx = (1.0f - fabsf(ny)) * (nx >= 0.0f ? 1.0f : -1.0f);
        float ty = (1.0f - fabsf(nx)) * (ny >= 0.0f ? 1.0f : -1.0f);
        nx = tx;
        ny = ty;
    }
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    out[0] = nx / len;
    out[1] = ny / len;
    out[2] = nz / len;
}

static inline void octEncodeNormal(float nx, float ny, float nz, int8_t* ox, int8_t* oy)
{
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 0.0f) { nx /= len; ny /= len; nz /= len; }
    else            { nx = 0.0f; ny = 0.0f; nz = 1.0f; }

    // L1 normalization projection
    float denom = fabsf(nx) + fabsf(ny) + fabsf(nz);
    float ax = nx / denom;
    float ay = ny / denom;

    // lower hemisphere fold
    if (nz < 0.0f)
    {
        float tx = (1.0f - fabsf(ay)) * (ax >= 0.0f ? 1.0f : -1.0f);
        float ty = (1.0f - fabsf(ax)) * (ay >= 0.0f ? 1.0f : -1.0f);
        ax = tx;
        ay = ty;
    }

    // clamp to snorm
    auto clamp127 = [](int32_t v) -> int8_t {
        if (v >  127) v =  127;
        if (v < -127) v = -127;
        return (int8_t)v;
    };

    int32_t bx = (int32_t)floorf(ax * 127.0f + 0.5f);
    int32_t by = (int32_t)floorf(ay * 127.0f + 0.5f);

    // neighbor search to find lowest error candidate
    float bestErr = 1e30f;
    int8_t bestX = 0, bestY = 0;
    for (int32_t dx = 0; dx <= 1; dx++)
    {
        for (int32_t dy = 0; dy <= 1; dy++)
        {
            int8_t cx = clamp127(bx + dx);
            int8_t cy = clamp127(by + dy);
            float d[3];
            octDecodeDir(cx, cy, d);
            float dot = d[0] * nx + d[1] * ny + d[2] * nz;
            float err = 1.0f - dot;
            if (err < bestErr) { bestErr = err; bestX = cx; bestY = cy; }
        }
    }
    *ox = bestX;
    *oy = bestY;
}

static inline uint16_t packOctNormal(float nx, float ny, float nz)
{
    int8_t ox, oy;
    octEncodeNormal(nx, ny, nz, &ox, &oy);
    return (uint16_t)((uint8_t)ox | ((uint16_t)(uint8_t)oy << 8));
}

static constexpr uint32_t kPos21Max = (1u << 21) - 1u; // 2097151

static void computeBounds(const std::vector<MeshVertexRaw>& verts,
                          float summand[3], float factor[3], float inv_range[3])
{
    float mn[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (const auto& v : verts)
        for (int i = 0; i < 3; i++) {
            mn[i] = std::min(mn[i], v.pos[i]);
            mx[i] = std::max(mx[i], v.pos[i]);
        }
    for (int i = 0; i < 3; i++) {
        summand[i]   = mn[i];
        float range  = mx[i] - mn[i];
        factor[i]    = range > 0.f ? range / (float)kPos21Max : 1.f;
        inv_range[i] = range > 0.f ? (float)kPos21Max / range  : 0.f;
    }
}

static void packPosition(float px, float py, float pz,
                         const float summand[3], const float inv_range[3],
                         uint32_t out[2])
{
    auto quantize = [&](float v, int axis) -> uint32_t {
        int32_t q = (int32_t)((v - summand[axis]) * inv_range[axis] + 0.5f);
        if (q < 0) q = 0;
        if (q > (int32_t)kPos21Max) q = (int32_t)kPos21Max;
        return (uint32_t)q;
    };
    uint32_t x = quantize(px, 0), y = quantize(py, 1), z = quantize(pz, 2);
    out[0] = x | ((y & 0x7FFu) << 21);
    out[1] = (y >> 11u) | (z << 10u);
}

#define CGLTF_IMPLEMENTATION
#include "extern/cgltf/cgltf.h"

#define CLUSTERLOD_IMPLEMENTATION
#include "extern/meshoptimizer/src/meshoptimizer.h"
#include "extern/meshoptimizer/demo/clusterlod.h"

static bool loadPrimitive(const cgltf_primitive* primitive, std::vector<MeshVertexRaw>& vertices, std::vector<uint32_t>& indices);

static MeshVertex encodeMeshVertex(const MeshVertexRaw& v,
                                   const float summand[3], const float inv_range[3])
{
    MeshVertex out{};
    packPosition(v.pos[0], v.pos[1], v.pos[2], summand, inv_range, out.pos);
    out.normal = packOctNormal(v.normal[0], v.normal[1], v.normal[2]);
    uint16_t hu = meshopt_quantizeHalf(v.uv[0]);
    uint16_t hv = meshopt_quantizeHalf(v.uv[1]);
    out.uv = (uint32_t(hv) << 16) | uint32_t(hu);
    return out;
}


static ClusteredMeshGPU uploadClusteredMesh(
    WGPUDevice device, WGPUQueue queue,
    const std::vector<MeshVertex>& vertices,
    const std::vector<ClusterN>&   clusters,
    const std::vector<uint32_t>&   meshletVertices,
    const std::vector<uint8_t>&    meshletTriangles)
{
    auto makeBuffer = [&](const void* data, size_t byteSize, WGPUBufferUsage usage, const char* label) -> WGPUBuffer
    {
        const size_t aligned = (byteSize + 3) & ~size_t(3);
        WGPUBufferDescriptor desc{
            .label = WGPUStringView{ .data = label, .length = strlen(label) },
            .usage = usage | WGPUBufferUsage_CopyDst,
            .size  = aligned,
        };
        WGPUBuffer buf = wgpuDeviceCreateBuffer(device, &desc);
        if (aligned != byteSize) {
            std::vector<uint8_t> padded(aligned, 0);
            memcpy(padded.data(), data, byteSize);
            wgpuQueueWriteBuffer(queue, buf, 0, padded.data(), aligned);
        } else {
            wgpuQueueWriteBuffer(queue, buf, 0, data, byteSize);
        }
        return buf;
    };

    constexpr WGPUBufferUsage kStorage = WGPUBufferUsage_Storage;

    ClusteredMeshGPU gpu{};
    gpu.vertexBuffer = makeBuffer(vertices.data(),
        vertices.size() * sizeof(MeshVertex),
        WGPUBufferUsage_Vertex | kStorage, "clustered mesh vertices");
    gpu.clusterBuffer = makeBuffer(clusters.data(),
        clusters.size() * sizeof(ClusterN),
        kStorage, "clustered mesh clusters");
    gpu.meshletVertexBuffer = makeBuffer(meshletVertices.data(),
        meshletVertices.size() * sizeof(uint32_t),
        kStorage, "clustered mesh meshlet vertices");
    gpu.meshletTriangleBuffer = makeBuffer(meshletTriangles.data(),
        meshletTriangles.size() * sizeof(uint8_t),
        kStorage, "clustered mesh meshlet triangles");

    gpu.vertexCount              = (uint32_t)vertices.size();
    gpu.clusterCount             = (uint32_t)clusters.size();
    gpu.meshletVertexCount       = (uint32_t)meshletVertices.size();
    gpu.meshletTriangleByteCount = (uint32_t)meshletTriangles.size();
    return gpu;
}

bool buildClusteredMeshFromGltf(
    const std::string&          path,
    std::vector<MeshVertexRaw>& outVertices,
    std::vector<ClusterN>&      outClusters,
    std::vector<uint32_t>&      outMeshletVertices,
    std::vector<uint8_t>&       outMeshletTriangles)
{
    cgltf_options options{};
    cgltf_data* gltf{};

    const char* cPath = path.c_str();
    cgltf_result result = cgltf_parse_file(&options, cPath, &gltf);
    if (result != cgltf_result_success)
    {
        fprintf(stderr, "buildClusteredMeshFromGltf: failed to parse %s\n", cPath);
        return false;
    }
    defer { cgltf_free(gltf); };

    result = cgltf_load_buffers(&options, gltf, cPath);
    if (result != cgltf_result_success)
    {
        fprintf(stderr, "buildClusteredMeshFromGltf: failed to load buffers for %s\n", cPath);
        return false;
    }
    result = cgltf_validate(gltf);
    if (result != cgltf_result_success)
    {
        fprintf(stderr, "buildClusteredMeshFromGltf: validation failed for %s\n", cPath);
        return false;
    }

    for (cgltf_size meshIdx = 0; meshIdx < gltf->meshes_count; meshIdx++)
    {
        const cgltf_mesh* mesh = &gltf->meshes[meshIdx];
        for (cgltf_size primIdx = 0; primIdx < mesh->primitives_count; primIdx++)
        {
            std::vector<MeshVertexRaw> vertices;
            std::vector<uint32_t>      indices;
            if (!loadPrimitive(&mesh->primitives[primIdx], vertices, indices))
                continue;

            const uint32_t clusterVertexBase   = (uint32_t)outVertices.size();
            const uint32_t meshletVertexBase   = (uint32_t)outMeshletVertices.size();
            const uint32_t meshletTriangleBase = (uint32_t)outMeshletTriangles.size();

            std::vector<ClusterN>  clusters;
            std::vector<uint32_t>  meshletVerts;
            std::vector<uint8_t>   meshletTriangles;
            buildNanite(vertices, indices, clusters, meshletVerts, meshletTriangles);

            for (ClusterN& c : clusters)
            {
                c.meshletVertexOffset   += meshletVertexBase;
                c.meshletTriangleOffset += meshletTriangleBase;
            }
            // offset global vertex indices from primitive-local to file-global
            for (uint32_t& vi : meshletVerts)
                vi += clusterVertexBase;

            outVertices.insert(outVertices.end(), vertices.begin(), vertices.end());
            outClusters.insert(outClusters.end(), clusters.begin(), clusters.end());
            outMeshletVertices.insert(outMeshletVertices.end(), meshletVerts.begin(), meshletVerts.end());
            outMeshletTriangles.insert(outMeshletTriangles.end(), meshletTriangles.begin(), meshletTriangles.end());
        }
    }

    if (outVertices.empty() || outClusters.empty())
    {
        fprintf(stderr, "buildClusteredMeshFromGltf: no usable primitives in %s\n", cPath);
        return false;
    }
    return true;
}

static constexpr uint32_t kClusteredMeshMagic   = 0x494E414E; // 'NANI'
static constexpr uint32_t kClusteredMeshVersion = 10;

struct ClusteredMeshFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t vertexCount;                  // unique global vertices
    uint32_t clusterCount;
    uint32_t meshletVertexCount;           // uint32_t entries in meshlet vertex index buffer
    uint32_t meshletTriangleByteCount;
    uint32_t encodedVertexByteCount;       // compressed vertex blob size
    uint32_t encodedClusterByteCount;      // compressed ClusterN blob size
    uint32_t encodedMeshletTotalBytes;     // total bytes of all per-cluster encoded meshlet blobs
    float    dequant_summand[3];
    float    dequant_factor[3];
};


bool saveClusteredMesh(
    const std::string&                path,
    const std::vector<MeshVertexRaw>& vertices,
    const std::vector<ClusterN>&      clusters,
    const std::vector<uint32_t>&      meshletVertices,
    const std::vector<uint8_t>&       meshletTriangles)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "saveClusteredMesh: cannot open %s for writing\n", path.c_str());
        return false;
    }

    float summand[3], factor[3], inv_range[3];
    computeBounds(vertices, summand, factor, inv_range);

    ClusteredMeshFileHeader hdr{
        .magic                    = kClusteredMeshMagic,
        .version                  = kClusteredMeshVersion,
        .vertexCount              = (uint32_t)vertices.size(),
        .clusterCount             = (uint32_t)clusters.size(),
        .meshletVertexCount       = (uint32_t)meshletVertices.size(),
        .meshletTriangleByteCount = (uint32_t)meshletTriangles.size(),
    };
    std::copy(summand, summand + 3, hdr.dequant_summand);
    std::copy(factor,  factor  + 3, hdr.dequant_factor);

    std::vector<MeshVertex> encodedVerts(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++)
        encodedVerts[i] = encodeMeshVertex(vertices[i], summand, inv_range);

    size_t bound = meshopt_encodeVertexBufferBound(encodedVerts.size(), sizeof(MeshVertex));
    std::vector<uint8_t> compressed(bound);
    size_t compressedSize = meshopt_encodeVertexBuffer(
        compressed.data(), compressed.size(),
        encodedVerts.data(), encodedVerts.size(), sizeof(MeshVertex));
    hdr.encodedVertexByteCount = (uint32_t)compressedSize;

    size_t clusterBound = meshopt_encodeVertexBufferBound(clusters.size(), sizeof(ClusterN));
    std::vector<uint8_t> compressedClusters(clusterBound);
    size_t compressedClusterSize = meshopt_encodeVertexBuffer(
        compressedClusters.data(), compressedClusters.size(),
        clusters.data(), clusters.size(), sizeof(ClusterN));
    hdr.encodedClusterByteCount = (uint32_t)compressedClusterSize;

    std::vector<uint32_t> meshletEncodedSizes(clusters.size());
    std::vector<uint8_t>  meshletBlob;
    {
        size_t maxBound = meshopt_encodeMeshletBound(64, 126);
        std::vector<uint8_t> tmp(maxBound);
        for (size_t ci = 0; ci < clusters.size(); ci++) {
            const ClusterN& c = clusters[ci];
            size_t n = meshopt_encodeMeshlet(
                tmp.data(), tmp.size(),
                &meshletVertices[c.meshletVertexOffset], c.vertexCount,
                &meshletTriangles[c.meshletTriangleOffset], c.triangleCount);
            meshletEncodedSizes[ci] = (uint32_t)n;
            meshletBlob.insert(meshletBlob.end(), tmp.begin(), tmp.begin() + n);
        }
    }
    hdr.encodedMeshletTotalBytes = (uint32_t)meshletBlob.size();

    bool ok =
        fwrite(&hdr,                        sizeof(hdr),      1,                             f) == 1 &&
        fwrite(compressed.data(),           1,                compressedSize,                f) == compressedSize &&
        fwrite(compressedClusters.data(),   1,                compressedClusterSize,         f) == compressedClusterSize &&
        fwrite(meshletEncodedSizes.data(),  sizeof(uint32_t), meshletEncodedSizes.size(),    f) == meshletEncodedSizes.size() &&
        fwrite(meshletBlob.data(),          1,                meshletBlob.size(),            f) == meshletBlob.size();

    fclose(f);
    if (!ok)
        fprintf(stderr, "saveClusteredMesh: write error on %s\n", path.c_str());
    return ok;
}

bool loadClusteredMeshFromFile(
    const std::string& path,
    WGPUDevice device, WGPUQueue queue,
    ClusteredMeshGPU& out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "loadClusteredMeshFromFile: cannot open %s\n", path.c_str());
        return false;
    }

    ClusteredMeshFileHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 ||
        hdr.magic   != kClusteredMeshMagic ||
        hdr.version != kClusteredMeshVersion)
    {
        fprintf(stderr, "loadClusteredMeshFromFile: invalid or unsupported file %s\n", path.c_str());
        fclose(f);
        return false;
    }

    std::vector<uint8_t>    compressed(hdr.encodedVertexByteCount);
    std::vector<MeshVertex> vertices(hdr.vertexCount);
    std::vector<uint8_t>    compressedClusters(hdr.encodedClusterByteCount);
    std::vector<ClusterN>   clusters(hdr.clusterCount);
    std::vector<uint32_t>   meshletEncodedSizes(hdr.clusterCount);
    std::vector<uint8_t>    meshletBlob(hdr.encodedMeshletTotalBytes);

    bool ok =
        fread(compressed.data(),          1,                hdr.encodedVertexByteCount,    f) == hdr.encodedVertexByteCount &&
        fread(compressedClusters.data(),  1,                hdr.encodedClusterByteCount,   f) == hdr.encodedClusterByteCount &&
        fread(meshletEncodedSizes.data(), sizeof(uint32_t), hdr.clusterCount,              f) == hdr.clusterCount &&
        fread(meshletBlob.data(),         1,                hdr.encodedMeshletTotalBytes,  f) == hdr.encodedMeshletTotalBytes;

    fclose(f);
    if (!ok) {
        fprintf(stderr, "loadClusteredMeshFromFile: read error on %s\n", path.c_str());
        return false;
    }

    if (meshopt_decodeVertexBuffer(vertices.data(), hdr.vertexCount, sizeof(MeshVertex),
                                   compressed.data(), hdr.encodedVertexByteCount) != 0)
    {
        fprintf(stderr, "loadClusteredMeshFromFile: vertex decode failed on %s\n", path.c_str());
        return false;
    }

    if (meshopt_decodeVertexBuffer(clusters.data(), hdr.clusterCount, sizeof(ClusterN),
                                   compressedClusters.data(), hdr.encodedClusterByteCount) != 0)
    {
        fprintf(stderr, "loadClusteredMeshFromFile: cluster decode failed on %s\n", path.c_str());
        return false;
    }

    std::vector<uint32_t> meshletVertices(hdr.meshletVertexCount);
    std::vector<uint8_t>  meshletTriangles(hdr.meshletTriangleByteCount);
    {
        uint32_t vertOff = 0, triOff = 0, blobOff = 0;
        for (size_t ci = 0; ci < clusters.size(); ci++) {
            ClusterN& c = clusters[ci];
            int rc = meshopt_decodeMeshlet(
                &meshletVertices[vertOff],   c.vertexCount,   sizeof(uint32_t),
                &meshletTriangles[triOff],   c.triangleCount, 3,
                &meshletBlob[blobOff], meshletEncodedSizes[ci]);
            if (rc != 0) {
                fprintf(stderr, "loadClusteredMeshFromFile: meshlet decode failed at cluster %zu in %s\n", ci, path.c_str());
                return false;
            }
            c.meshletVertexOffset   = vertOff;
            c.meshletTriangleOffset = triOff;
            vertOff  += c.vertexCount;
            triOff   += (uint32_t)c.triangleCount * 3;
            blobOff  += meshletEncodedSizes[ci];
        }
    }

    out = uploadClusteredMesh(device, queue, vertices, clusters, meshletVertices, meshletTriangles);
    std::copy(hdr.dequant_summand, hdr.dequant_summand + 3, out.dequant_summand);
    std::copy(hdr.dequant_factor,  hdr.dequant_factor  + 3, out.dequant_factor);

    printf("loadClusteredMeshFromFile: %s — vertices: %u  clusters: %u  meshletTris(bytes): %u\n",
        path.c_str(), out.vertexCount, out.clusterCount, out.meshletTriangleByteCount);
    return true;
}



bool loadGltfMeshToGPU(const std::string& path, WGPUDevice device, WGPUQueue queue, ClusteredMeshGPU& out)
{
    std::vector<MeshVertexRaw> vertices;
    std::vector<ClusterN>      clusters;
    std::vector<uint32_t>      meshletVertices;
    std::vector<uint8_t>       meshletTriangles;
    if (!buildClusteredMeshFromGltf(path, vertices, clusters, meshletVertices, meshletTriangles))
        return false;

    float summand[3], factor[3], inv_range[3];
    computeBounds(vertices, summand, factor, inv_range);

    std::vector<MeshVertex> encodedVerts(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++)
        encodedVerts[i] = encodeMeshVertex(vertices[i], summand, inv_range);

    out = uploadClusteredMesh(device, queue, encodedVerts, clusters, meshletVertices, meshletTriangles);
    std::copy(summand, summand + 3, out.dequant_summand);
    std::copy(factor,  factor  + 3, out.dequant_factor);

    printf("loadGltfMeshToGPU: %s — vertices: %u  clusters: %u  meshletTris(bytes): %u\n",
        path.c_str(), out.vertexCount, out.clusterCount, out.meshletTriangleByteCount);
    return true;
}

// Takes the raw vertex and index buffer and computes the clustered lod
void buildNanite(
    const std::vector<MeshVertexRaw>& vertices,
    const std::vector<uint32_t>&      indices,
    std::vector<ClusterN>&            outClusters,
    std::vector<uint32_t>&            outMeshletVertices,
    std::vector<uint8_t>&             outMeshletTriangles)
{
    clodConfig config = clodDefaultConfig(126);
    config.max_vertices = 64;

    const float attributeWeights[3] = {0.5f, 0.5f, 0.5f};

    clodMesh mesh{};
    mesh.indices                  = indices.data();
    mesh.index_count              = indices.size();
    mesh.vertex_count             = vertices.size();
    mesh.vertex_positions         = &vertices[0].pos[0];
    mesh.vertex_positions_stride  = sizeof(MeshVertexRaw);
    mesh.vertex_attributes        = &vertices[0].normal[0];
    mesh.vertex_attributes_stride = sizeof(MeshVertexRaw);
    mesh.attribute_weights        = attributeWeights;
    mesh.attribute_count          = 3;
    mesh.attribute_protect_mask   = (1 << 3) | (1 << 4);

    size_t reservedClusters = ((indices.size() / 3) + 127) / 128 * 3;
    outClusters.reserve(outClusters.size() + reservedClusters);

    std::vector<clodBounds> groups;
    groups.reserve((reservedClusters + 15) / 16);

    clodBuild(config, mesh,
        [&](clodGroup group, const clodCluster* clusters, size_t count) -> int
    {
        int groupIndex = (int)groups.size();
        groups.push_back(group.simplified);

        for (size_t i = 0; i < count; i++)
        {
            const clodCluster& cluster = clusters[i];

            std::vector<uint32_t> localVerts(cluster.vertex_count);
            std::vector<uint8_t>  meshletTris(cluster.index_count);

            size_t actualVertCount = clodLocalIndices(localVerts.data(), meshletTris.data(), cluster.indices, cluster.index_count);
            localVerts.resize(actualVertCount);

            ClusterN c{};
            c.refined      = cluster.refined;
            c.groupCenterX = group.simplified.center[0];
            c.groupCenterY = group.simplified.center[1];
            c.groupCenterZ = group.simplified.center[2];
            c.groupRadius  = group.simplified.radius;
            c.groupError   = group.simplified.error;

            if (cluster.refined >= 0 && cluster.refined < (int)groups.size())
            {
                const clodBounds& ref = groups[cluster.refined];
                c.refinedCenterX = ref.center[0];
                c.refinedCenterY = ref.center[1];
                c.refinedCenterZ = ref.center[2];
                c.refinedRadius  = ref.radius;
                c.refinedError   = ref.error;
            }

            c.meshletVertexOffset   = (uint32_t)outMeshletVertices.size();
            c.meshletTriangleOffset = (uint32_t)outMeshletTriangles.size();
            c.vertexCount           = (uint8_t)actualVertCount;
            c.triangleCount         = (uint8_t)(cluster.index_count / 3);

            outClusters.push_back(c);
            outMeshletVertices.insert(outMeshletVertices.end(), localVerts.begin(), localVerts.end());
            outMeshletTriangles.insert(outMeshletTriangles.end(), meshletTris.begin(), meshletTris.end());
        }

        return groupIndex;
    });
}

// takes the group and
void buildBvh()
{

}

static bool loadPrimitive(
    const cgltf_primitive* primitive,
    std::vector<MeshVertexRaw>& vertices,
    std::vector<uint32_t>& indices)
{
    if (primitive->type != cgltf_primitive_type_triangles ||
        primitive->indices == nullptr)
        return false;

    const cgltf_accessor* pos =
        cgltf_find_accessor(primitive, cgltf_attribute_type_position, 0);
    if (!pos)
        return false;

    vertices.assign(pos->count, MeshVertexRaw{});

    std::vector<float> scratch(pos->count * 4);

    cgltf_accessor_unpack_floats(pos, scratch.data(), pos->count * 3);
    for (size_t i = 0; i < pos->count; i++)
    {
        vertices[i].pos[0] = scratch[i * 3 + 0];
        vertices[i].pos[1] = scratch[i * 3 + 1];
        vertices[i].pos[2] = scratch[i * 3 + 2];
    }

    if (auto n = cgltf_find_accessor(primitive, cgltf_attribute_type_normal, 0))
    {
        cgltf_accessor_unpack_floats(n, scratch.data(), n->count * 3);
        for (size_t i = 0; i < pos->count; i++)
        {
            vertices[i].normal[0] = scratch[i * 3 + 0];
            vertices[i].normal[1] = scratch[i * 3 + 1];
            vertices[i].normal[2] = scratch[i * 3 + 2];
        }
    }
    else
    {
        // Default to +Z when normals are absent.
        for (size_t i = 0; i < pos->count; i++)
        {
            vertices[i].normal[0] = 0.0f;
            vertices[i].normal[1] = 0.0f;
            vertices[i].normal[2] = 1.0f;
        }
    }

    if (auto uv = cgltf_find_accessor(primitive, cgltf_attribute_type_texcoord, 0))
    {
        cgltf_accessor_unpack_floats(uv, scratch.data(), uv->count * 2);
        for (size_t i = 0; i < pos->count; i++)
        {
            vertices[i].uv[0] = scratch[i * 2 + 0];
            vertices[i].uv[1] = scratch[i * 2 + 1];
        }
    }

    size_t indexCount = primitive->indices->count;
    indices.resize(indexCount);
    cgltf_accessor_unpack_indices(primitive->indices, indices.data(), 4, indexCount);

    std::vector<uint32_t> remap(pos->count);
    size_t uniqueVertices = meshopt_generateVertexRemap(
        remap.data(), indices.data(), indexCount,
        vertices.data(), vertices.size(), sizeof(MeshVertexRaw));

    meshopt_remapVertexBuffer(vertices.data(), vertices.data(), vertices.size(), sizeof(MeshVertexRaw), remap.data());
    meshopt_remapIndexBuffer(indices.data(), indices.data(), indexCount, remap.data());
    meshopt_optimizeVertexCache(indices.data(), indices.data(), indexCount, uniqueVertices);
    meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indexCount, vertices.data(), uniqueVertices, sizeof(MeshVertexRaw));

    vertices.resize(uniqueVertices);
    return true;
}

// meshoptimizer unity build
#include "extern/meshoptimizer/src/allocator.cpp"
#include "extern/meshoptimizer/src/clusterizer.cpp"
#include "extern/meshoptimizer/src/indexanalyzer.cpp"
#include "extern/meshoptimizer/src/indexcodec.cpp"
#include "extern/meshoptimizer/src/indexgenerator.cpp"
#include "extern/meshoptimizer/src/meshletcodec.cpp"
#include "extern/meshoptimizer/src/meshletutils.cpp"
#include "extern/meshoptimizer/src/opacitymap.cpp"
#include "extern/meshoptimizer/src/overdrawoptimizer.cpp"
#include "extern/meshoptimizer/src/partition.cpp"
#include "extern/meshoptimizer/src/quantization.cpp"
#include "extern/meshoptimizer/src/rasterizer.cpp"
#include "extern/meshoptimizer/src/simplifier.cpp"
#include "extern/meshoptimizer/src/spatialorder.cpp"
#include "extern/meshoptimizer/src/stripifier.cpp"
#include "extern/meshoptimizer/src/vcacheoptimizer.cpp"
#include "extern/meshoptimizer/src/vertexcodec.cpp"
#include "extern/meshoptimizer/src/vertexfilter.cpp"
#include "extern/meshoptimizer/src/vfetchoptimizer.cpp"
