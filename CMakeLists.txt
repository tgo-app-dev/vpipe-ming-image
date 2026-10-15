cmake_minimum_required(VERSION 3.20)
project(vpipe_ming_image LANGUAGES CXX)

# Build against an INSTALLED vpipe, e.g.
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/vpipe-install
#   cmake --build build -j
# then load it:
#   vpipe --plugin build/vpipe-ming-image.so ...
#
# Built for plugin ABI 8. What it needs beyond the integer is named in
# src/ming-plugin.cc (kRequires). A plugin must be built against the
# libvpipe it loads into: re-install the SDK after rebuilding vpipe.
find_package(vpipe REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The model layer. One OBJECT library so the plugin and the tests share
# compiled objects, and so a file no test reaches is still COMPILED.
add_library(ming_model_obj OBJECT
    src/ming-catalog.cc
    src/ming-config.cc
    src/ming-dit.cc
    src/ming-family.cc
    src/ming-generator.cc
    src/ming-model-config-stage.cc
    src/ming-layout.cc
    src/ming-tokenize.cc
    src/ming-ops.cc
    src/ming-llm.cc
    src/ming-connector.cc
    src/ming-encoder.cc
    src/ming-profiles.cc
    src/ming-quant-family.cc)
target_include_directories(ming_model_obj
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(ming_model_obj PRIVATE vpipe::vpipe vpipe::toolkit)
target_compile_features(ming_model_obj PRIVATE cxx_std_20)
set_target_properties(ming_model_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
    target_compile_options(ming_model_obj PRIVATE -Wall -Werror)
endif()

# The DiT's own kernels: the entries it dispatches that the host's kernel
# contract does not promise (see src/metal/ming_ops.metal). Two dtype
# twins, as the in-tree tree builds every library; the DiT runs bf16.
vpipe_add_metal_library(ming_ops_bf16_obj ming_ops_bf16
    SRC src/metal/ming_ops.metal DEFINES VPIPE_ELT=bfloat)
vpipe_add_metal_library(ming_ops_f16_obj ming_ops_f16
    SRC src/metal/ming_ops.metal DEFINES VPIPE_ELT=half)

# The conditioning encoder's own kernels (src/metal/ming_llm.metal):
# the f32 residual ops, the q/k norm + table RoPE and the grouped MoE
# GEMMs. bf16 only -- the MLLM and the connector run nothing else.
vpipe_add_metal_library(ming_llm_obj ming_llm SRC src/metal/ming_llm.metal)

vpipe_add_plugin(vpipe-ming-image
    SOURCES
        src/ming-plugin.cc
        src/ming-conditioner-stage.cc
        src/ming-llm-kernels.cc
        $<TARGET_OBJECTS:ming_llm_obj>
        $<TARGET_OBJECTS:ming_model_obj>
        $<TARGET_OBJECTS:ming_ops_bf16_obj>
        $<TARGET_OBJECTS:ming_ops_f16_obj>
)
target_include_directories(vpipe-ming-image
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# The encoder's tokenizer normalizes with CoreFoundation (NFC, Unicode
# classes) and its MoE routers run on Accelerate (f32, as the reference).
set(MING_ENCODER_FRAMEWORKS
    "-framework CoreFoundation" "-framework Accelerate")
target_link_libraries(vpipe-ming-image PRIVATE ${MING_ENCODER_FRAMEWORKS})
if(NOT MSVC)
    target_compile_options(vpipe-ming-image PRIVATE -Wall -Werror)
endif()

# ---- tests ------------------------------------------------------------
enable_testing()

# Plain main()s: a plugin cannot reach the host's minitest. Tests that
# need the checkpoint or goldens read env vars and SKIP when unset
# (see README.md "What is verified").
function(ming_test name)
    add_executable(${name}-test tests/${name}-test.cc
                   $<TARGET_OBJECTS:ming_model_obj>)
    target_include_directories(${name}-test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(${name}-test PRIVATE vpipe::vpipe vpipe::toolkit
                          ${MING_ENCODER_FRAMEWORKS})
    target_compile_features(${name}-test PRIVATE cxx_std_20)
    if(NOT MSVC)
        target_compile_options(${name}-test PRIVATE -Wall -Werror)
    endif()
    add_test(NAME ${name} COMMAND ${name}-test)
endfunction()

ming_test(ming-config)

# The DiT against the reference's own goldens (VPIPE_MING_GOLDEN). A test
# binary is not a plugin, so it links the bf16 metallib and registers it
# itself, through the same public API the plugin uses.
ming_test(ming-dit)
target_sources(ming-dit-test PRIVATE $<TARGET_OBJECTS:ming_ops_bf16_obj>)

# The whole denoise through the production generator, against the
# reference's e2e trajectory (VPIPE_MING_TEST_MODEL_PATH + VPIPE_MING_GOLDEN).
ming_test(ming-generator)
target_sources(ming-generator-test PRIVATE
               $<TARGET_OBJECTS:ming_ops_bf16_obj>)

# The conditioning encoder against the reference's goldens
# (VPIPE_MING_TEST_MODEL_PATH + VPIPE_MING_GOLDEN). It links the encoder's
# metallib and registers it itself, as the DiT test does its own.
ming_test(ming-encoder)
target_sources(ming-encoder-test PRIVATE src/ming-llm-kernels.cc
               $<TARGET_OBJECTS:ming_llm_obj>)

# Runtime LoRA through the production plumbing: the family reads `lora` /
# `lora2` off the model_config at load, the generator applies strengths
# live per request (VPIPE_MING_TEST_MODEL_PATH + VPIPE_MING_GOLDEN). The
# adapter arithmetic itself is ming-dit's.
ming_test(ming-lora)
target_sources(ming-lora-test PRIVATE $<TARGET_OBJECTS:ming_ops_bf16_obj>)

# model-quantize's Ming-Image family, the detect profile and the pickers
# (model-free), its claims against the checkpoint and lookalikes
# (VPIPE_MING_TEST_MODEL_PATH), and the w8 / w4 packs it writes
# (VPIPE_MING_W8_PACK / VPIPE_MING_W4_PACK + VPIPE_MING_GOLDEN): the real
# forward on each, engagement of the quantized GEMM, streamed ==
# resident. It reads the conditioner's StageSpec, hence that source.
ming_test(ming-quant)
target_sources(ming-quant-test PRIVATE src/ming-conditioner-stage.cc
               $<TARGET_OBJECTS:ming_ops_bf16_obj>)
