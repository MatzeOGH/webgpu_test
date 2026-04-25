#include "ClusteredMesh.h"
#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.gltf> <output.nanite>\n", argv[0]);
        return 1;
    }

    std::vector<MeshVertex> vertices;
    std::vector<ClusterN>   clusters;
    std::vector<uint32_t>   meshletVertices;
    std::vector<uint8_t>    meshletTriangles;
    if (!buildClusteredMeshFromGltf(argv[1], vertices, clusters, meshletVertices, meshletTriangles))
        return 1;

    if (!saveClusteredMesh(argv[2], vertices, clusters, meshletVertices, meshletTriangles))
        return 1;

    printf("Written: %s\n  vertices: %zu  clusters: %zu  meshletVerts: %zu  meshletTris(bytes): %zu\n",
        argv[2], vertices.size(), clusters.size(), meshletVertices.size(), meshletTriangles.size());

    return 0;
}
