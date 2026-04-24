#include "ClusteredMesh.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include "AlpUtils.h"

#define CGLTF_IMPLEMENTATION
#include "extern/cgltf/cgltf.h"

#define CLUSTERLOD_IMPLEMENTATION
#include "extern/meshoptimizer/src/meshoptimizer.h"
#include "extern/meshoptimizer/demo/clusterlod.h"

static bool loadPrimitive(const cgltf_primitive* primitive, std::vector<MeshVertex>& vertices, std::vector<uint32_t>& indices);

bool loadGltfMeshToGPU(const std::string& path, WGPUDevice device, WGPUQueue queue, ClusteredMeshGPU& out)
{
    cgltf_options options{};
    cgltf_data* gltf{};

    const char* cPath = path.c_str();
    cgltf_result result = cgltf_parse_file(&options, cPath, &gltf);
    if (result != cgltf_result_success)
    {
        fprintf(stderr, "loadGltfMeshToGPU: failed to parse %s\n", cPath);
        return false;
    }
    defer { cgltf_free(gltf); };

    result = cgltf_load_buffers(&options, gltf, cPath);
    if (result != cgltf_result_success)
    {
        fprintf(stderr, "loadGltfMeshToGPU: failed to load buffers for %s\n", cPath);
        return false;
    }
    result = cgltf_validate(gltf);
    if (result != cgltf_result_success)
    {
        fprintf(stderr, "loadGltfMeshToGPU: validation failed for %s\n", cPath);
        return false;
    }

    std::vector<MeshVertex> allVertices;
    std::vector<ClusterN>   allClusters;
    std::vector<uint32_t>   allMeshletVertices;
    std::vector<uint8_t>    allMeshletTriangles;

    for (cgltf_size meshIdx = 0; meshIdx < gltf->meshes_count; meshIdx++)
    {
        const cgltf_mesh* mesh = &gltf->meshes[meshIdx];
        for (cgltf_size primIdx = 0; primIdx < mesh->primitives_count; primIdx++)
        {
            std::vector<MeshVertex> vertices;
            std::vector<uint32_t>   indices;
            if (!loadPrimitive(&mesh->primitives[primIdx], vertices, indices))
                continue;

            const uint32_t vertexBase         = (uint32_t)allVertices.size();
            const uint32_t meshletVertexBase   = (uint32_t)allMeshletVertices.size();
            const uint32_t meshletTriangleBase = (uint32_t)allMeshletTriangles.size();

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

            allVertices.insert(allVertices.end(), vertices.begin(), vertices.end());
            allClusters.insert(allClusters.end(), clusters.begin(), clusters.end());
            allMeshletVertices.insert(allMeshletVertices.end(), meshletVertices.begin(), meshletVertices.end());
            allMeshletTriangles.insert(allMeshletTriangles.end(), meshletTriangles.begin(), meshletTriangles.end());
        }
    }

    if (allVertices.empty() || allClusters.empty())
    {
        fprintf(stderr, "loadGltfMeshToGPU: no usable primitives in %s\n", cPath);
        return false;
    }

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
    gpu.vertexBuffer = makeBuffer(allVertices.data(),
        allVertices.size() * sizeof(MeshVertex),
        WGPUBufferUsage_Vertex | kStorage, "clustered mesh vertices");
    gpu.clusterBuffer = makeBuffer(allClusters.data(),
        allClusters.size() * sizeof(ClusterN),
        kStorage, "clustered mesh clusters");
    gpu.meshletVertexBuffer = makeBuffer(allMeshletVertices.data(),
        allMeshletVertices.size() * sizeof(uint32_t),
        kStorage, "clustered mesh meshlet vertices");
    gpu.meshletTriangleBuffer = makeBuffer(allMeshletTriangles.data(),
        allMeshletTriangles.size() * sizeof(uint8_t),
        kStorage, "clustered mesh meshlet triangles");

    gpu.vertexCount              = (uint32_t)allVertices.size();
    gpu.clusterCount             = (uint32_t)allClusters.size();
    gpu.meshletVertexCount       = (uint32_t)allMeshletVertices.size();
    gpu.meshletTriangleByteCount = (uint32_t)allMeshletTriangles.size();

    printf("loadGltfMeshToGPU: %s — vertices: %u  clusters: %u  meshletVerts: %u  meshletTris(bytes): %u\n",
        cPath, gpu.vertexCount, gpu.clusterCount, gpu.meshletVertexCount, gpu.meshletTriangleByteCount);

    out = gpu;
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

    clodMesh mesh{};
    mesh.indices                  = indices.data();
    mesh.index_count              = indices.size();
    mesh.vertex_count             = vertices.size();
    mesh.vertex_positions         = &vertices[0].x;
    mesh.vertex_positions_stride  = sizeof(MeshVertex);
    mesh.vertex_attributes        = &vertices[0].nx;
    mesh.vertex_attributes_stride = sizeof(MeshVertex);
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

            size_t actualVertCount = clodLocalIndices(
                localVerts.data(), meshletTris.data(),
                cluster.indices, cluster.index_count);

            localVerts.resize(actualVertCount);

            ClusterN c{};
            c.refined = cluster.refined;

            c.groupCenter[0] = group.simplified.center[0];
            c.groupCenter[1] = group.simplified.center[1];
            c.groupCenter[2] = group.simplified.center[2];
            c.groupRadius    = group.simplified.radius;
            c.groupError     = group.simplified.error;

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
            c.vertexCount           = (uint32_t)actualVertCount;
            c.triangleCount         = (uint32_t)(cluster.index_count / 3);

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
            vertices[i].nx = scratch[i * 3 + 0];
            vertices[i].ny = scratch[i * 3 + 1];
            vertices[i].nz = scratch[i * 3 + 2];
        }
    }

    if (auto uv = cgltf_find_accessor(primitive, cgltf_attribute_type_texcoord, 0))
    {
        cgltf_accessor_unpack_floats(uv, scratch.data(), uv->count * 2);
        for (size_t i = 0; i < pos->count; i++)
        {
            vertices[i].tu = scratch[i * 2 + 0];
            vertices[i].tv = scratch[i * 2 + 1];
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
