#include "ClusteredMesh.h"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

static void printStats(
    const char*                        outPath,
    const std::vector<MeshVertexRaw>&  vertices,
    const std::vector<ClusterN>&       clusters,
    const std::vector<uint32_t>&       meshletVertices,
    const std::vector<uint8_t>&        meshletTriangles,
    double buildMs, double saveMs)
{
    printf("\n=== Mesh Build Statistics ===\n");
    printf("  Build: %.0f ms   Save: %.0f ms   Total: %.0f ms\n",
        buildMs, saveMs, buildMs + saveMs);

    // ---- quantization bounds ----
    float mn[3] = { FLT_MAX,  FLT_MAX,  FLT_MAX};
    float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (const auto& v : vertices)
        for (int i = 0; i < 3; i++) {
            mn[i] = std::min(mn[i], v.pos[i]);
            mx[i] = std::max(mx[i], v.pos[i]);
        }
    constexpr float kMax21 = float((1u << 21) - 1u);
    float extX = mx[0] - mn[0], extY = mx[1] - mn[1], extZ = mx[2] - mn[2];
    printf("\nInput\n");
    printf("  Cluster-local verts : %zu\n", vertices.size());
    printf("  AABB       : [%.3f %.3f %.3f] – [%.3f %.3f %.3f]\n",
        mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
    printf("  Quant step : X=%.6f  Y=%.6f  Z=%.6f  (21 bits/axis)\n",
        extX / kMax21, extY / kMax21, extZ / kMax21);

    // ---- cluster utilization ----
    int nClusters    = (int)clusters.size();
    int leafClusters = 0;
    int minV = 255, maxV = 0, totalV = 0;
    int minT = 255, maxT = 0, totalT = 0;
    int fineLodTris = 0;

    for (const auto& c : clusters) {
        int v = c.vertexCount, t = c.triangleCount;
        minV = std::min(minV, v); maxV = std::max(maxV, v); totalV += v;
        minT = std::min(minT, t); maxT = std::max(maxT, t); totalT += t;
        if (c.refined < 0) { leafClusters++; fineLodTris += t; }
    }
    int interiorClusters = nClusters - leafClusters;
    float avgV = nClusters ? (float)totalV / nClusters : 0.f;
    float avgT = nClusters ? (float)totalT / nClusters : 0.f;

    printf("\nClusters\n");
    printf("  Total      : %d  (leaf/finest: %d  coarser LODs: %d)\n",
        nClusters, leafClusters, interiorClusters);
    printf("  Verts/clust: min %d  avg %.1f  max %d  (cap 64)   util %.0f%%\n",
        minV, avgV, maxV, avgV / 64.f * 100.f);
    printf("  Tris/clust : min %d  avg %.1f  max %d  (cap 126)  util %.0f%%\n",
        minT, avgT, maxT, avgT / 126.f * 100.f);
    printf("  Finest-LOD tris : %d\n", fineLodTris);
    printf("  All-LOD tris    : %d\n", totalT);

    // ---- file size breakdown ----
    // ClusteredMeshFileHeader = 6 x uint32 + 6 x float = 48 bytes
    size_t szHeader   = 6 * sizeof(uint32_t) + 6 * sizeof(float);
    size_t szVerts    = vertices.size()        * sizeof(MeshVertex);
    size_t szClusters = clusters.size()        * sizeof(ClusterN);
    size_t szMVerts   = meshletVertices.size() * sizeof(uint32_t);
    size_t szMTris    = meshletTriangles.size();
    size_t szTotal    = szHeader + szVerts + szClusters + szMVerts + szMTris;

    auto pct = [&](size_t s) { return szTotal ? s * 100.0 / szTotal : 0.0; };
    auto kb  = [](size_t s)  { return s / 1024.0; };

    printf("\nFile: %s\n", outPath);
    printf("  Header          : %5.1f KB  (%4.1f%%)\n", kb(szHeader),   pct(szHeader));
    printf("  Vertices        : %5.1f KB  (%4.1f%%)  %zu x %zu B  %.1f B/vert\n",
        kb(szVerts), pct(szVerts), vertices.size(), sizeof(MeshVertex),
        vertices.empty() ? 0.f : (float)szVerts / vertices.size());
    printf("  Clusters        : %5.1f KB  (%4.1f%%)  %zu x %zu B\n",
        kb(szClusters), pct(szClusters), clusters.size(), sizeof(ClusterN));
    printf("  Meshlet verts   : %5.1f KB  (%4.1f%%)  %zu x 4 B\n",
        kb(szMVerts), pct(szMVerts), meshletVertices.size());
    printf("  Meshlet tris    : %5.1f KB  (%4.1f%%)  %zu B\n",
        kb(szMTris), pct(szMTris), meshletTriangles.size());
    printf("  Total           : %5.1f KB\n", kb(szTotal));
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.gltf> <output.nanite>\n", argv[0]);
        return 1;
    }

    std::vector<MeshVertexRaw> vertices;
    std::vector<ClusterN>      clusters;
    std::vector<uint32_t>      meshletVertices;
    std::vector<uint8_t>       meshletTriangles;

    auto t0 = std::chrono::steady_clock::now();
    if (!buildClusteredMeshFromGltf(argv[1], vertices, clusters, meshletVertices, meshletTriangles))
        return 1;

    auto t1 = std::chrono::steady_clock::now();
    if (!saveClusteredMesh(argv[2], vertices, clusters, meshletVertices, meshletTriangles))
        return 1;

    auto t2 = std::chrono::steady_clock::now();

    double buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double saveMs  = std::chrono::duration<double, std::milli>(t2 - t1).count();

    printStats(argv[2], vertices, clusters, meshletVertices, meshletTriangles, buildMs, saveMs);

    return 0;
}
