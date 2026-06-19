// Standalone driver / smoke test for the render graph.
// Not a standalone TU: #included at the end of RenderGraph.cpp so it sees the
// internal node structs. Builds a small graph, compiles it, and dumps the result.

#include "RenderGraph.h"

// ponytail: empty PassContext stub so the smoke test can invoke the stored callbacks;
// the real one (encoder/views) lands with the execute phase and replaces this.
namespace RG { struct PassContext {}; }

int main()
{
    using namespace RG;

    GraphAllocator* allocator = create_allocator();
    RenderGraph* rg = create_render_graph(allocator);

    // imported swapchain target: writing it makes a pass a sink (keeps it + its deps alive).
    // no real swapchain in the smoke test -> dummy view + fixed size (the size is what realize()
    // would later resolve Relative resources against).
    auto swapchain = rg->importe_image(WEBGPU_STR("swapchain"), /*view*/nullptr, {1280, 720, 1});
    auto texture = rg->importe_image(WEBGPU_STR("asset.albedo"), /*view*/nullptr, {1024, 1024, 1});

    auto colorAttachment = rg->create_image(WEBGPU_STR("image.a"), {
        .dimension = WGPUTextureDimension_2D,
        .format = WGPUTextureFormat_RGBA8UnormSrgb,
        .sizeKind = SizeKind::Relative,
        .scaleX = 0.5f, .scaleY = 0.5f,   // half the swapchain -> proves scale chains down to depth
        .relativeTo = swapchain,
    });
    auto depth = rg->create_image(WEBGPU_STR("image.depth"), {
        .dimension = WGPUTextureDimension_2D,
        .format = WGPUTextureFormat_Depth32Float,
        .sizeKind = SizeKind::Relative,
        .relativeTo = colorAttachment,
    });

    // pass B reads A's texture and writes the imported swapchain -> sink.
    // depends on A (RAW on texture); kept alive because it feeds the swapchain.
    rg->add_pass(
        WEBGPU_STR("pass - B"),
        PassKind::Graphics,
        [&](GraphBuilder& builder){
            builder.sampled(colorAttachment);   // RAW: A wrote texture via color
            builder.color(swapchain);   // writes the imported target -> B is a sink
        },
        [](PassContext&){ std::printf("    body executed\n"); }
    );

    rg->add_pass(
        WEBGPU_STR("pass - A"),
        PassKind::Graphics,
        [&](GraphBuilder& builder){
            builder.color(colorAttachment);
            builder.depth_stencil(depth);
            builder.sampled(texture);
        },
        [](PassContext&){
            std::printf("    body executed\n");
        }
    );
    // pass C writes only `depth`, which nobody reads and isn't imported -> not a sink,
    // nothing depends on it -> DEAD. compile() should cull it from the execution order.
    rg->add_pass(
        WEBGPU_STR("pass - C"),
        PassKind::Graphics,
        [&](GraphBuilder& builder){
            builder.sampled(texture);
            builder.color(depth);
        },
        [](PassContext&){ std::printf("    body executed\n"); }
    );



    rg->compile();

    std::printf("execution order:");
    for (PassNode* p = rg->m_passes; p; p = p->next)
        std::printf(" %s", p->name.data);
    std::printf("\n");

    // invoke the stored execute callbacks in order (proves add_pass's storage round-trips)
    std::printf("executing:\n");
    PassContext ctx{};
    for (PassNode* p = rg->m_passes; p; p = p->next) {
        std::printf("  run %s\n", p->name.data);
        if (p->exec_fn) p->exec_fn(p->exec_obj, ctx);
    }

    // emit dependency DAG as mermaid (edge p -> dep means "p depends on dep")
    auto passIndex = [&](PassNode* q){
        int k = 0;
        for (auto* x = rg->m_passes; x; x = x->next, ++k) if (x == q) return k;
        return -1;
    };
    std::printf("flowchart LR\n");
    int pi = 0;
    for (auto* pass = rg->m_passes; pass; pass = pass->next, ++pi)
        std::printf("  p%d[\"%s\"]\n", pi, pass->name.data);
    pi = 0;
    for (auto* pass = rg->m_passes; pass; pass = pass->next, ++pi)
        for (auto* a = pass->adjacency; a; a = a->next)
            std::printf("  p%d --> p%d\n", pi, passIndex(a->pass));



    auto kindName = [](ResourceNode::Kind k) {
        return k == ResourceNode::Kind::Texture ? "Texture" : "Buffer";
    };
    int n = 0;
    for (auto* r = rg->m_resouces; r; r = r->next) {
        if (r->kind == ResourceNode::Kind::Texture)
            // ponytail: dim/fmt as raw enum ints; add a name table if logs need to be readable
            std::printf("  resource[%d] id=%u %s %s  %s %ux%ux%u scale(%.2f,%.2f) dim=%d fmt=%d\n",
                        n++, r->handle.id, kindName(r->kind), r->name.data,
                        r->sizeKind == SizeKind::Absolute ? "abs" : "rel",
                        r->resolved.width, r->resolved.height, r->resolved.depthOrArrayLayers,
                        r->scaleX, r->scaleY,
                        (int)r->dimension, (int)r->format);
        else
            std::printf("  resource[%d] id=%u %s %s\n",
                        n++, r->handle.id, kindName(r->kind), r->name.data);
    }
    auto accessName = [](AccessType t) {
        switch (t) {
            case AccessType::ColorAttachment:        return "color";
            case AccessType::DepthStencilAttachment: return "depth_stencil";
            case AccessType::Sampled:                return "sampled";
            case AccessType::StorageRead:            return "storage_read";
            case AccessType::StorageWrite:           return "storage_write";
        }
        return "?";
    };
    int p = 0;
    for (auto* pass = rg->m_passes; pass; pass = pass->next, ++p) {
        std::printf("  pass[%d] \"%s\" kind=%d  %u accesses\n",
                    p, pass->name.data, (int)pass->kind, pass->accessCount);
        for (uint32_t i = 0; i < pass->accessCount; ++i) {
            const ResourceAccess& a = pass->accesses[i];
            std::printf("      %-13s res=%u  %s\n",
                        accessName(a.type), a.handle.id,
                        access_is_write(a.type) ? "(write)" : "(read)");
        }
    }
    std::printf("RenderGraph ran: %d resources, %d passes\n", n, p);
    return 0;
}
