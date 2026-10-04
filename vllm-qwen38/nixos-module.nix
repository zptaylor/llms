{ config, pkgs, lib, ... }:

let
  vllmQwen38Dir = ./.;

  vllmQwen38BuildDir = "/home/USER/vllm-qwen38-build";

  intelOneApiToolkit = pkgs.intel-oneapi-toolkit.override {
    components = [ "default" ];
  };

  vllmQwen38Packages = pkgs.runCommand "vllm-qwen38-env" {
    buildInputs = with pkgs; [
      python311
      python311Packages.uv
      git
      cmake
      ninja
      intelOneApiToolkit
      level-zero
      level-zero-headers
      intel-compute-runtime
      intel-media-driver
      vpl-gpu-rt
      patchelf
      pkg-config
      opencl-headers
      opencl-icd-loader
    ];
    INTEL_ONEAPI_ROOT = "${intelOneApiToolkit}";
    LD_LIBRARY_PATH = "${intelOneApiToolkit}/compiler/latest/linux/lib:${intelOneApiToolkit}/mkl/latest/lib/intel64:${intelOneApiToolkit}/tbb/latest/lib/intel64/gcc4.8:${pkgs.level-zero}/lib:${pkgs.intel-compute-runtime}/lib";
    CPLUS_INCLUDE_PATH = "${intelOneApiToolkit}/compiler/latest/linux/include:${intelOneApiToolkit}/mkl/latest/include:${intelOneApiToolkit}/tbb/latest/include:${pkgs.level-zero-headers}/include";
    LIBRARY_PATH = "${intelOneApiToolkit}/compiler/latest/linux/lib:${intelOneApiToolkit}/mkl/latest/lib/intel64:${intelOneApiToolkit}/tbb/latest/lib/intel64/gcc4.8:${pkgs.level-zero}/lib:${pkgs.intel-compute-runtime}/lib";
    PKG_CONFIG_PATH = "${pkgs.level-zero}/lib/pkgconfig:${pkgs.intel-compute-runtime}/lib/pkgconfig";
  } ''
    mkdir -p $out
    # This is a placeholder - the actual build happens via the scripts
  '';

in {
  options = {
    services.vllm-qwen38 = {
      enable = lib.mkEnableOption "vLLM Qwen3.8-27B on Intel Arc B70";
      port = lib.mkOption {
        type = lib.types.int;
        default = 19622;
        description = "Port for vLLM server";
      };
      tp = lib.mkOption {
        type = lib.types.int;
        default = 1;
        description = "Tensor parallel size (1 for single GPU)";
      };
      mtp = lib.mkOption {
        type = lib.types.int;
        default = 3;
        description = "MTP speculative tokens";
      };
      int8Head = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Enable INT8 lm_head optimization";
      };
      maxModelLen = lib.mkOption {
        type = lib.types.int;
        default = 8192;
        description = "Maximum model context length";
      };
      kvCacheDtype = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "KV cache dtype (e.g., 'fp8' for long context)";
      };
      gpuIndex = lib.mkOption {
        type = lib.types.int;
        default = 0;
        description = "GPU index for single-GPU mode";
      };
      modelDir = lib.mkOption {
        type = lib.types.path;
        default = "/home/USER/models/johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound";
        description = "Path to model directory (must be writable / outside /nix/store)";
      };
      hfToken = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "Hugging Face token for gated models";
      };
      extraEnv = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = { };
        description = "Extra environment variables";
      };
    };
  };

  config = lib.mkIf config.services.vllm-qwen38.enable {
    boot.initrd.kernelModules = [ "xe" ];
    boot.kernelModules = [ "xe" ];
    boot.kernelParams = [ "xe.force_probe=*" ];

    hardware.graphics = {
      enable = true;
      extraPackages = with pkgs; [
        intel-media-driver
        vpl-gpu-rt
        intel-compute-runtime
        intel-gpu-tools
        clinfo
      ];
    };

    hardware.enableRedistributableFirmware = true;

    environment.systemPackages = with pkgs; [
      git
      cmake
      ninja
      intelOneApiToolkit
      level-zero
      intel-compute-runtime
      intel-media-driver
      vpl-gpu-rt
      patchelf
      pkg-config
      opencl-headers
      ocl-icd
      intel-gpu-tools
      uv
    ];

    systemd.services.vllm-qwen38 = {
      description = "vLLM Qwen3.8-27B on Intel Arc B70 (XPU)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" "systemd-modules-load.service" "vllm-qwen38-setup.service" ];
      wants = [ "network-online.target" ];
      serviceConfig = {
        Type = "simple";
        Restart = "on-failure";
        RestartSec = 15;
        TimeoutStartSec = 900;
        User = "USER";
        Group = "USER";
        ConditionPathExists = "${vllmQwen38BuildDir}/.venv";
        Environment = [
          "VLLM_TARGET_DEVICE=xpu"
          "VLLM_XPU_ENABLE_XPU_GRAPH=1"
          "CCL_TOPO_P2P_ACCESS=0"
          "CCL_ZE_IPC_EXCHANGE=pidfd"
          "CCL_ATL_TRANSPORT=ofi"
          "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE"
          "VLLM_XPU_HOST_STAGED_COLLECTIVES=1"
          "ONEAPI_ROOT=${intelOneApiToolkit}"
          "ZE_ENABLE_PCI_ID=1"
          "PYTORCH_ALLOC_CONF=expandable_segments:True"
          "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
          "VLLM_WORKER_MULTIPROC_METHOD=spawn"
          "VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1"
        ] ++ lib.optional (config.services.vllm-qwen38.int8Head) "VLLM_XPU_LM_HEAD_INT8=1"
          ++ lib.optional (config.services.vllm-qwen38.kvCacheDtype != null) "KV_CACHE_DTYPE=${config.services.vllm-qwen38.kvCacheDtype}"
          ++ lib.optional (config.services.vllm-qwen38.tp == 1) "ZE_AFFINITY_MASK=${toString config.services.vllm-qwen38.gpuIndex}"
          ++ lib.optional (config.services.vllm-qwen38.tp == 1) "GPU_MEMORY_UTILIZATION=0.88"
          ++ lib.optional (config.services.vllm-qwen38.tp == 1) "ONEAPI_DEVICE_SELECTOR=level_zero:${toString config.services.vllm-qwen38.gpuIndex}"
          ++ lib.optional (config.services.vllm-qwen38.tp != 1) "GPU_MEMORY_UTILIZATION=0.95"
          ++ (builtins.attrValues config.services.vllm-qwen38.extraEnv);
        ExecStartPre = [
          "${pkgs.bash}/bin/bash -c 'for card in /sys/class/drm/card*/device; do for hwmon in \"$card\"/hwmon/hwmon*; do if [ -f \"$hwmon/power1_cap\" ]; then echo 250000000 > \"$hwmon/power1_cap\"; elif [ -f \"$hwmon/power1_max\" ]; then echo 250000000 > \"$hwmon/power1_max\"; fi; done; if [ -f \"$card/intel_gpu_power_max\" ]; then echo 250 > \"$card/intel_gpu_power_max\"; fi; done'"
          (pkgs.writeShellScript "vllm-qwen38-lazy-model" ''
            set -euo pipefail
            MD=${toString config.services.vllm-qwen38.modelDir}
            want=7
            got=$(find "$MD" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l)
            if (( got >= want )); then
              echo "vllm-qwen38: model present at $MD ($got/$want shards) - no fetch"
              exit 0
            fi
            echo "vllm-qwen38: model missing/incomplete at $MD ($got/$want) - fetching pinned revision"
            export HF_TOKEN=''${VLLM_HF_TOKEN:-${if config.services.vllm-qwen38.hfToken != null then config.services.vllm-qwen38.hfToken else ""}}
            ${vllmQwen38BuildDir}/.venv/bin/hf download \
              johannrplaster/Qwen3.8-27B-Uncensored-int4-AutoRound \
              --revision 74f84ec7e7779c8628f72935887cfaa47903116d \
              --local-dir "$MD" || true
            exit 0
          '')
        ];
        ExecStart = let
          baseCmd = [
            "${vllmQwen38BuildDir}/.venv/bin/vllm"
            "serve"
            "${config.services.vllm-qwen38.modelDir}"
            "--host" "0.0.0.0"
            "--port" "${toString config.services.vllm-qwen38.port}"
            "--served-model-name" "Qwen3.8-27B-UNC-G128-AR"
            "--tensor-parallel-size" "${toString config.services.vllm-qwen38.tp}"
            "--dtype" "float16"
            "--max-model-len" "${toString config.services.vllm-qwen38.maxModelLen}"
            "--max-num-seqs" "1"
            "--max-num-batched-tokens" "8192"
            "--quantization" "gptq"
            "--enable-prompt-tokens-details"
            "--speculative-config" "'{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${toString config.services.vllm-qwen38.mtp}}'"
            "--enforce-eager"
            "--trust-remote-code"
          ];
          kvArgs = if config.services.vllm-qwen38.kvCacheDtype != null then [
            "--kv-cache-dtype" "${config.services.vllm-qwen38.kvCacheDtype}"
            "--gpu-memory-utilization" "0.88"
          ] else [];
        in lib.concatStringsSep " " (baseCmd ++ kvArgs);
        ExecStop = "${pkgs.killall}/bin/killall vllm";
      };
    };

    networking.firewall.allowedTCPPorts = [ config.services.vllm-qwen38.port ];


    systemd.services.vllm-qwen38-setup = {
      description = "vLLM Qwen3.8-27B prepare writable build dir (non-fatal)";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "USER";
        Group = "USER";
        SuccessExitStatus = [ 0 1 127 ];
        Environment = [
          "PATH=${pkgs.bash}/bin:${pkgs.coreutils}/bin:${pkgs.gnugrep}/bin:${pkgs.gawk}/bin:${pkgs.gnused}/bin:${pkgs.findutils}/bin:${pkgs.nix}/bin:${pkgs.git}/bin:${pkgs.uv}/bin:${intelOneApiToolkit}/compiler/latest/linux/bin"
        ];
        ExecStart = "${pkgs.bash}/bin/bash -c 'mkdir -p ${vllmQwen38BuildDir} && if [ ! -d ${vllmQwen38BuildDir}/scripts ]; then cp -r ${vllmQwen38Dir}/* ${vllmQwen38BuildDir}/ 2>/dev/null || true; fi && chmod -R u+rwX ${vllmQwen38BuildDir} 2>/dev/null || true && chown -R USER:USER ${vllmQwen38BuildDir} 2>/dev/null || true';";
      };
    };
  };
}
