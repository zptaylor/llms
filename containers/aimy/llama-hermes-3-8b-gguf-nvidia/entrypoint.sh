#!/bin/bash
set -e

CUDA_LIBDIR=/usr/local/cuda/lib64
COMPAT_DIR=/usr/local/cuda-12.8/compat

NIX_LIB_DIR=""
for d in /nix/store/*/lib; do
    if [ -e "$d/libcuda.so.1" ]; then
        NIX_LIB_DIR="$d"
        break
    fi
done

if [ -n "$NIX_LIB_DIR" ]; then
    echo "entrypoint: Found NixOS NVIDIA libs at $NIX_LIB_DIR"

    NIX_CUDA="$NIX_LIB_DIR/libcuda.so.1"
    [ ! -e "$CUDA_LIBDIR/libcuda.so.1" ] && ln -sf "$NIX_CUDA" "$CUDA_LIBDIR/libcuda.so.1"
    [ -d "$COMPAT_DIR" ] && [ -d "$COMPAT_DIR" ] && [ ! -e "$COMPAT_DIR/libcuda.so.1" ] && ln -sf "$NIX_CUDA" "$COMPAT_DIR/libcuda.so.1"
    echo "entrypoint: libcuda.so.1 -> $NIX_CUDA"

    NIX_PTJ="$NIX_LIB_DIR/libnvidia-ptxjitcompiler.so.1"
    if [ -e "$NIX_PTJ" ]; then
        [ ! -e "$CUDA_LIBDIR/libnvidia-ptxjitcompiler.so.1" ] && ln -sf "$NIX_PTJ" "$CUDA_LIBDIR/libnvidia-ptxjitcompiler.so.1"
        [ -d "$COMPAT_DIR" ] && [ ! -e "$COMPAT_DIR/libnvidia-ptxjitcompiler.so.1" ] && ln -sf "$NIX_PTJ" "$COMPAT_DIR/libnvidia-ptxjitcompiler.so.1"
        echo "entrypoint: libnvidia-ptxjitcompiler.so.1 -> $NIX_PTJ"
    fi

    for lib in libnvidia-ml libnvidia-cfg libcudadebugger libnvidia-nvvm; do
        NIX_SO="$NIX_LIB_DIR/${lib}.so.1"
        if [ -e "$NIX_SO" ]; then
            [ ! -e "$CUDA_LIBDIR/${lib}.so.1" ] && ln -sf "$NIX_SO" "$CUDA_LIBDIR/${lib}.so.1"
            echo "entrypoint: ${lib}.so.1 -> $NIX_SO"
        fi
    done

    export LD_LIBRARY_PATH="$CUDA_LIBDIR${COMPAT_DIR:+:$COMPAT_DIR}:$NIX_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "entrypoint: LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi

exec /app/llama-server "$@"
