// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native barrier contracts; --hip <arch> lowers serialized test IR from stdin. */
#include <cstdio>
#include <cstring>
#include <string>

#include "rocke/error.hpp"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"

#define CHECK(x)                                                       \
    do                                                                 \
    {                                                                  \
        if(!(x))                                                       \
        {                                                              \
            fprintf(stderr, "check failed at %d: %s\n", __LINE__, #x); \
            return 1;                                                  \
        }                                                              \
    } while(0)

static int lower_hip(rocke_ir_builder_t* b, const char* arch, bool print)
{
    rocke_lower_hip_opts_t opts = {};
    opts.arch = arch;
    rocke_strbuf_t out;
    CHECK(rocke_strbuf_init(&out, 128) == 0);
    auto status = rocke_lower_kernel_to_hip(b, b->kernel, &opts, &out);
    if(status == ROCKE_OK && print)
        fputs(rocke_strbuf_cstr(&out), stdout);
    rocke_strbuf_free(&out);
    if(status != ROCKE_OK)
        fprintf(stderr, "native HIP lowering failed: status=%d\n", status);
    return status == ROCKE_OK ? 0 : 1;
}

static int test_admission()
{
    const rocke_type_t* logical[] = {rocke_fp4e2m1(),
                                     rocke_fp6e2m3(),
                                     rocke_fp6e3m2(),
                                     rocke_e8m0(),
                                     rocke_e5m3(),
                                     rocke_tf32()};
    for(int index = 0; index < 9; ++index)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "barrier_rejection") == ROCKE_OK);
        const auto* type = index < 6    ? logical[index]
                           : index == 6 ? rocke_ptr_type(&b, rocke_f32(), "global")
                                        : rocke_vector_type(&b, rocke_f32(), 2);
        auto* value = index == 8 ? nullptr : rocke_b_param(&b, "value", type, nullptr);
        int before = b.kernel->body->num_ops;
        bool rejected = false;
        try
        {
            rocke_b_optimization_barrier(&b, value);
        }
        catch(const ckc::Error& error)
        {
            rejected = error.code() == ROCKE_ERR_VALUE
                       && strcmp(error.what(),
                                 "optimization_barrier requires a directly lowerable scalar or i1")
                              == 0;
        }
        CHECK(rejected);
        CHECK(b.kernel->body->num_ops == before);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

static int test_scalars()
{
    const rocke_type_t* types[] = {rocke_i1(),
                                   rocke_i8(),
                                   rocke_i16(),
                                   rocke_i32(),
                                   rocke_i64(),
                                   rocke_bf16(),
                                   rocke_f16(),
                                   rocke_f32(),
                                   rocke_fp8e4m3(),
                                   rocke_bf8e5m2()};
    const char* arches[] = {"gfx950", "gfx1250"};
    for(const char* arch : arches)
        for(const auto* type : types)
        {
            rocke_ir_builder_t b;
            CHECK(rocke_ir_builder_init(&b, "barrier_scalar") == ROCKE_OK);
            auto* ptr = rocke_b_param(&b, "p", rocke_ptr_type(&b, type, "global"), nullptr);
            auto* tid = rocke_b_thread_id_x(&b);
            auto* value = rocke_b_global_load(&b, ptr, tid, type, 1);
            auto* result = rocke_b_optimization_barrier(&b, value);
            CHECK(result && rocke_type_eq(result->type, type));
            rocke_b_global_store(&b, ptr, tid, result, 1);
            rocke_b_ret(&b);
            CHECK(lower_hip(&b, arch, false) == 0);
            rocke_ir_builder_free(&b);
        }
    return 0;
}

static int test_raw_asm()
{
    // Only the exact barrier form is newly supported. Malformed forms must not
    // produce a partial identity expression or read missing operand arrays.
    for(int mutation = 0; mutation < 13; ++mutation)
    {
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "barrier_raw") == ROCKE_OK);
        auto* value = rocke_b_param(&b, "value", rocke_f32(), nullptr);
        auto* result = rocke_b_optimization_barrier(&b, value);
        auto* op = result->op;
        rocke_status_t expected = ROCKE_ERR_VALUE;
        switch(mutation)
        {
        case 0:
            op->num_operands = 0;
            break;
        case 1:
            op->num_results = 0;
            break;
        case 2:
            op->operands = nullptr;
            break;
        case 3:
            op->results = nullptr;
            break;
        case 4:
            op->operands[0] = nullptr;
            break;
        case 5:
            op->results[0] = nullptr;
            break;
        case 6:
            result->type = rocke_i32();
            break;
        case 7:
            value->type = result->type = rocke_vector_type(&b, rocke_f32(), 2);
            break;
        case 8:
            rocke_attr_set_bool(&b, &op->attrs, "sideeffect", true);
            expected = ROCKE_ERR_NOTIMPL;
            break;
        case 9:
            rocke_attr_set_bool(&b, &op->attrs, "convergent", true);
            expected = ROCKE_ERR_NOTIMPL;
            break;
        case 10:
            rocke_attr_set_str(&b, &op->attrs, "clobber", "memory");
            expected = ROCKE_ERR_NOTIMPL;
            break;
        case 11:
            rocke_attr_set_str(&b, &op->attrs, "template", "v_mov_b32 $0, $1");
            expected = ROCKE_ERR_NOTIMPL;
            break;
        case 12:
            rocke_attr_set_str(&b, &op->attrs, "constraints", "=v,v");
            expected = ROCKE_ERR_NOTIMPL;
            break;
        }
        // Keep the parameter's valid signature separate from mutated operand types.
        b.kernel->params[0]->type = rocke_f32();
        rocke_strbuf_t out;
        CHECK(rocke_strbuf_init(&out, 128) == 0);
        CHECK(rocke_lower_kernel_to_hip(&b, b.kernel, nullptr, &out) == expected);
        rocke_strbuf_free(&out);
        rocke_ir_builder_free(&b);
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc == 3 && strcmp(argv[1], "--hip") == 0)
    {
        std::string input;
        char buffer[4096];
        size_t count;
        while((count = fread(buffer, 1, sizeof(buffer), stdin)) != 0)
            input.append(buffer, count);
        rocke_ir_builder_t b;
        CHECK(rocke_ir_builder_init(&b, "barrier_import") == ROCKE_OK);
        rocke_kernel_def_t* kernel = nullptr;
        auto status = rocke_ir_parse(input.c_str(), &b, &kernel);
        int result = 1;
        if(status == ROCKE_OK && kernel)
            result = lower_hip(&b, argv[2], true);
        else
            fprintf(stderr, "native IR parse failed: status=%d\n", status);
        rocke_ir_builder_free(&b);
        return result;
    }
    if(argc != 1)
    {
        fprintf(stderr, "usage: %s [--hip <arch>]\n", argv[0]);
        return 2;
    }
    int admission = test_admission();
    int scalars = test_scalars();
    int raw = test_raw_asm();
    return admission || scalars || raw;
}
