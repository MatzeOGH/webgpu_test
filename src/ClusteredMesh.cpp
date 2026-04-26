#include "ClusteredMesh.h"
#include <algorithm>
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

    float denom = fabsf(nx) + fabsf(ny) + fabsf(nz);
    float ax = nx / denom;
    float ay = ny / denom;
    if (nz < 0.0f)
    {
        float tx = (1.0f - fabsf(ay)) * (ax >= 0.0f ? 1.0f : -1.0f);
        float ty = (1.0f - fabsf(ax)) * (ay >= 0.0f ? 1.0f : -1.0f);
        ax = tx;
        ay = ty;
    }

    auto clamp127 = [](int32_t v) -> int8_t {
        if (v >  127) v =  127;
        if (v < -127) v = -127;
        return (int8_t)v;
    };

    int32_t bx = (int32_t)floorf(ax * 127.0f + 0.5f);
    int32_t by = (int32_t)floorf(ay * 127.0f + 0.5f);

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

static inline uint32_t packOctNormal(float nx, float ny, float nz)
{
    int8_t ox, oy;
    octEncodeNormal(nx, ny, nz, &ox, &oy);
    return (uint32_t)(uint8_t)ox | ((uint32_t)(uint8_t)oy << 8);
}

#define CGLTF_IMPLEMENTATION
#include "extern/cgltf/cgltf.h"

#define CLUSTERLOD_IMPLEMENTATION
#include "extern/meshoptimizer/src/meshoptimizer.h"
#include "extern/meshoptimizer/demo/clusterlod.h"

static bool loadPrimitive(const cgltf_primitive* primitive, std::vector<MeshVertex>& vertices, std::vector<uint32_t>& indices);


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
    const std::string&       path,
    std::vector<MeshVertex>& outVertices,
    std::vector<ClusterN>&   outClusters,
    std::vector<uint32_t>&   outMeshletVertices,
    std::vector<uint8_t>&    outMeshletTriangles)
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
            std::vector<MeshVertex> vertices;
            std::vector<uint32_t>   indices;
            if (!loadPrimitive(&mesh->primitives[primIdx], vertices, indices))
                continue;

            const uint32_t vertexBase         = (uint32_t)outVertices.size();
            const uint32_t meshletVertexBase   = (uint32_t)outMeshletVertices.size();
            const uint32_t meshletTriangleBase = (uint32_t)outMeshletTriangles.size();

            std::vector<ClusterN> clusters;
            std::vector<uint32_t> meshletVertices;
            std::vector<uint8_t>  meshletTriangles;
            buildNanite(vertices, indices, clusters, meshletVertices, meshletTriangles);

            for (ClusterN& c : clusters)
            {
                c.meshletVertexOffset   += meshletVertexBase;
                c.meshletTriangleOffset += meshletTriangleBase;
            }
            for (uint32_t& vi : meshletVertices)
                vi += vertexBase;

            outVertices.insert(outVertices.end(), vertices.begin(), vertices.end());
            outClusters.insert(outClusters.end(), clusters.begin(), clusters.end());
            outMeshletVertices.insert(outMeshletVertices.end(), meshletVertices.begin(), meshletVertices.end());
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
static constexpr uint32_t kClusteredMeshVersion = 3;

struct ClusteredMeshFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t vertexCount;
    uint32_t clusterCount;
    uint32_t meshletVertexCount;
    uint32_t meshletTriangleByteCount;
};


bool saveClusteredMesh(
    const std::string&             path,
    const std::vector<MeshVertex>& vertices,
    const std::vector<ClusterN>&   clusters,
    const std::vector<uint32_t>&   meshletVertices,
    const std::vector<uint8_t>&    meshletTriangles)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "saveClusteredMesh: cannot open %s for writing\n", path.c_str());
        return false;
    }

    ClusteredMeshFileHeader hdr{
        .magic                   = kClusteredMeshMagic,
        .version                 = kClusteredMeshVersion,
        .vertexCount             = (uint32_t)vertices.size(),
        .clusterCount            = (uint32_t)clusters.size(),
        .meshletVertexCount      = (uint32_t)meshletVertices.size(),
        .meshletTriangleByteCount= (uint32_t)meshletTriangles.size(),
    };

    bool ok =
        fwrite(&hdr, sizeof(hdr), 1, f) == 1 &&
        fwrite(vertices.data(),        sizeof(MeshVertex), vertices.size(),        f) == vertices.size() &&
        fwrite(clusters.data(),        sizeof(ClusterN),   clusters.size(),        f) == clusters.size() &&
        fwrite(meshletVertices.data(), sizeof(uint32_t),   meshletVertices.size(), f) == meshletVertices.size() &&
        fwrite(meshletTriangles.data(),sizeof(uint8_t),    meshletTriangles.size(),f) == meshletTriangles.size();

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

    std::vector<MeshVertex> vertices(hdr.vertexCount);
    std::vector<ClusterN>   clusters(hdr.clusterCount);
    std::vector<uint32_t>   meshletVertices(hdr.meshletVertexCount);
    std::vector<uint8_t>    meshletTriangles(hdr.meshletTriangleByteCount);

    bool ok =
        fread(vertices.data(),        sizeof(MeshVertex), hdr.vertexCount,              f) == hdr.vertexCount &&
        fread(clusters.data(),        sizeof(ClusterN),   hdr.clusterCount,             f) == hdr.clusterCount &&
        fread(meshletVertices.data(), sizeof(uint32_t),   hdr.meshletVertexCount,       f) == hdr.meshletVertexCount &&
        fread(meshletTriangles.data(),sizeof(uint8_t),    hdr.meshletTriangleByteCount, f) == hdr.meshletTriangleByteCount;

    fclose(f);
    if (!ok) {
        fprintf(stderr, "loadClusteredMeshFromFile: read error on %s\n", path.c_str());
        return false;
    }

    out = uploadClusteredMesh(device, queue, vertices, clusters, meshletVertices, meshletTriangles);

    printf("loadClusteredMeshFromFile: %s — vertices: %u  clusters: %u  meshletVerts: %u  meshletTris(bytes): %u\n",
        path.c_str(), out.vertexCount, out.clusterCount, out.meshletVertexCount, out.meshletTriangleByteCount);
    return true;
}



bool loadGltfMeshToGPU(const std::string& path, WGPUDevice device, WGPUQueue queue, ClusteredMeshGPU& out)
{
    std::vector<MeshVertex> vertices;
    std::vector<ClusterN>   clusters;
    std::vector<uint32_t>   meshletVertices;
    std::vector<uint8_t>    meshletTriangles;
    if (!buildClusteredMeshFromGltf(path, vertices, clusters, meshletVertices, meshletTriangles))
        return false;

    out = uploadClusteredMesh(device, queue, vertices, clusters, meshletVertices, meshletTriangles);

    printf("loadGltfMeshToGPU: %s — vertices: %u  clusters: %u  meshletVerts: %u  meshletTris(bytes): %u\n",
        path.c_str(), out.vertexCount, out.clusterCount, out.meshletVertexCount, out.meshletTriangleByteCount);
    return true;
}

void buildNanite(
    const std::vector<MeshVertex>& vertices,
    const std::vector<uint32_t>&   indices,
    std::vector<ClusterN>&         outClusters,
    std::vector<uint32_t>&         outMeshletVertices,
    std::vector<uint8_t>&          outMeshletTriangles)
{
    clodConfig config = clodDefaultConfig(126);
    config.max_vertices = 64;

    const float attributeWeights[3] = {0.5f, 0.5f, 0.5f};

    // clusterlod consumes attributes as raw floats, but MeshVertex stores normals
    // oct-encoded. Decode into a parallel float buffer just for the build.
    std::vector<float> decodedNormals(vertices.size() * 3);
    for (size_t i = 0; i < vertices.size(); i++)
    {
        int8_t ox = (int8_t)(vertices[i].normal       & 0xFFu);
        int8_t oy = (int8_t)((vertices[i].normal >> 8) & 0xFFu);
        octDecodeDir(ox, oy, &decodedNormals[i * 3]);
    }

    clodMesh mesh{};
    mesh.indices                  = indices.data();
    mesh.index_count              = indices.size();
    mesh.vertex_count             = vertices.size();
    mesh.vertex_positions         = &vertices[0].x;
    mesh.vertex_positions_stride  = sizeof(MeshVertex);
    mesh.vertex_attributes        = decodedNormals.data();
    mesh.vertex_attributes_stride = sizeof(float) * 3;
    mesh.attribute_weights        = attributeWeights;
    mesh.attribute_count          = 3;
    mesh.attribute_protect_mask   = (1 << 3) | (1 << 4);

    size_t reservedClusters = ((indices.size() / 3) + 127) / 128 * 3;
    outClusters.reserve(reservedClusters);

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
            c.refined = cluster.refined;

            c.groupCenter[0] = group.simplified.center[0];
            c.groupCenter[1] = group.simplified.center[1];
            c.groupCenter[2] = group.simplified.center[2];
            c.groupRadius    = group.simplified.radius;
            c.groupError     = group.simplified.error;

            // if cluster is not refined: skip
            if (cluster.refined >= 0 && cluster.refined < (int)groups.size())
            {
                const clodBounds& ref = groups[cluster.refined];
                c.refinedCenter[0] = ref.center[0];
                c.refinedCenter[1] = ref.center[1];
                c.refinedCenter[2] = ref.center[2];
                c.refinedRadius    = ref.radius;
                c.refinedError     = ref.error;
            }

            c.meshletVertexOffset   = (uint32_t)outMeshletVertices.size();
            c.meshletTriangleOffset = (uint32_t)outMeshletTriangles.size();

            c.vertexCount = (uint8_t)actualVertCount;
            c.triangleCount = (uint8_t)(cluster.index_count / 3);

            outClusters.push_back(c);
            outMeshletVertices.insert(outMeshletVertices.end(), localVerts.begin(), localVerts.end());
            outMeshletTriangles.insert(outMeshletTriangles.end(), meshletTris.begin(), meshletTris.end());
        }

        return groupIndex;
    });
}

static bool loadPrimitive(
    const cgltf_primitive* primitive,
    std::vector<MeshVertex>& vertices,
    std::vector<uint32_t>& indices)
{
    if (primitive->type != cgltf_primitive_type_triangles ||
        primitive->indices == nullptr)
        return false;

    const cgltf_accessor* pos =
        cgltf_find_accessor(primitive, cgltf_attribute_type_position, 0);
    if (!pos)
        return false;

    vertices.resize(pos->count);

    std::vector<float> scratch(pos->count * 4);

    cgltf_accessor_unpack_floats(pos, scratch.data(), pos->count * 3);
    for (size_t i = 0; i < pos->count; i++)
    {
        vertices[i].x = scratch[i * 3 + 0];
        vertices[i].y = scratch[i * 3 + 1];
        vertices[i].z = scratch[i * 3 + 2];
    }

    if (auto n = cgltf_find_accessor(primitive, cgltf_attribute_type_normal, 0))
    {
        cgltf_accessor_unpack_floats(n, scratch.data(), n->count * 3);
        for (size_t i = 0; i < pos->count; i++)
        {
            vertices[i].normal = packOctNormal(
                scratch[i * 3 + 0],
                scratch[i * 3 + 1],
                scratch[i * 3 + 2]);
        }
    }
    else
    {
        // Default to +Z when normals are absent.
        for (size_t i = 0; i < pos->count; i++)
            vertices[i].normal = packOctNormal(0.0f, 0.0f, 1.0f);
    }

    if (auto uv = cgltf_find_accessor(primitive, cgltf_attribute_type_texcoord, 0))
    {
        cgltf_accessor_unpack_floats(uv, scratch.data(), uv->count * 2);
        for (size_t i = 0; i < pos->count; i++)
        {
            uint16_t hu = meshopt_quantizeHalf(scratch[i * 2 + 0]);
            uint16_t hv = meshopt_quantizeHalf(scratch[i * 2 + 1]);
            vertices[i].uv = (uint32_t(hv) << 16) | uint32_t(hu);
        }
    }

    size_t indexCount = primitive->indices->count;
    indices.resize(indexCount);
    cgltf_accessor_unpack_indices(primitive->indices, indices.data(), 4, indexCount);

    std::vector<uint32_t> remap(pos->count);
    size_t uniqueVertices = meshopt_generateVertexRemap(
        remap.data(), indices.data(), indexCount,
        vertices.data(), vertices.size(), sizeof(MeshVertex));

    meshopt_remapVertexBuffer(vertices.data(), vertices.data(), vertices.size(), sizeof(MeshVertex), remap.data());
    meshopt_remapIndexBuffer(indices.data(), indices.data(), indexCount, remap.data());
    meshopt_optimizeVertexCache(indices.data(), indices.data(), indexCount, uniqueVertices);
    meshopt_optimizeVertexFetch(vertices.data(), indices.data(), indexCount, vertices.data(), uniqueVertices, sizeof(MeshVertex));

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
