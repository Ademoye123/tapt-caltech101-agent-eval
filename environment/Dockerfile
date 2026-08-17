# TAPT task environment — pinned repro stack + bundled Caltech101 assets.
# Built from the same public image and deps as the team's provenance lane
# (see stage2/dependency_lock.md in the workspace provenance record):
#   torch 2.6.0+cu124 (pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel)
#   Python 3.11 · Dassl.pytorch @ c61a1b5 (patched) · TAPT @ 14d8d0e · MMoP @ 44e38d4 (patched)
# Everything the agent needs is baked in: no network dependency at runtime.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential unzip patch jq \
    && rm -rf /var/lib/apt/lists/*

COPY context/requirements.probe.txt /tmp/requirements.probe.txt
RUN sed -i '/^autoattack==/d' /tmp/requirements.probe.txt \
    && pip install --no-cache-dir --index-url https://pypi.org/simple -r /tmp/requirements.probe.txt \
    && pip install --no-cache-dir \
        "git+https://github.com/fra31/auto-attack@a39220048b3c9f2cca9a4d3a54604793c68eca7e" \
    && pip install --no-cache-dir --index-url https://pypi.org/simple "fvcore==0.1.5.post20221221"

ARG DASSL_COMMIT=c61a1b570ac6333bd50fb5ae06aea59002fb20bb
RUN git clone https://github.com/KaiyangZhou/Dassl.pytorch.git /opt/Dassl.pytorch \
    && git -C /opt/Dassl.pytorch checkout ${DASSL_COMMIT} \
    && pip install --no-cache-dir -e /opt/Dassl.pytorch --no-deps \
    && rm -rf /opt/Dassl.pytorch/.git

COPY context/dassl-weights_only.patch /tmp/dassl-weights_only.patch
RUN patch -p1 -d /opt/Dassl.pytorch < /tmp/dassl-weights_only.patch \
    && rm /tmp/dassl-weights_only.patch

COPY context/TAPT/ /opt/TAPT/
COPY context/MMoP/ /opt/MMoP/

COPY context/mmop-autoattack-save.patch /tmp/mmop-autoattack-save.patch
RUN patch -p1 -d /opt/MMoP < /tmp/mmop-autoattack-save.patch \
    && rm /tmp/mmop-autoattack-save.patch

# --- paper (agents read it, never download) ------------------------------------
COPY paper/ /workspace/paper/

# --- Caltech101 data (CoOp split scheme as the paper uses) ---------------------
COPY data/caltech-101.zip /tmp/caltech-101.zip
RUN unzip -q /tmp/caltech-101.zip -d /workspace/data \
    && rm /tmp/caltech-101.zip \
    && mkdir -p /workspace/data
COPY context/split_zhou_Caltech101.json /workspace/data/caltech-101/split_zhou_Caltech101.json

# --- bundled assets: training checkpoints + eval statistics + adv pkl ----------
COPY assets/ /workspace/assets/

# --- runtime user (non-privileged; submission dir is /workspace/submission) ----
RUN useradd -m -s /bin/bash agent \
    && mkdir -p /workspace/submission /workspace/tmp \
    && chown -R agent:agent /workspace
USER agent
WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["echo 'TAPT task env ready'"]