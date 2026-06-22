from concurrent import futures
from pathlib import Path
from typing import Optional


def _trace(msg):
    import os
    import sys
    import time
    print(
        f"[M2 {time.strftime('%H:%M:%S')} pid={os.getpid()}] {msg}",
        file=sys.stderr,
        flush=True,
    )


_trace("server.py: importing grpc")
import grpc

_trace("server.py: importing torch")
import torch

_trace("server.py: importing log, util")
import log
import util

_trace("server.py: importing specedge_grpc (generated protobuf/grpc)")
from specedge_grpc import specedge_pb2, specedge_pb2_grpc

_trace("server.py: cloud server module imports complete")


class CloudTargetServicer(specedge_pb2_grpc.SpecEdgeServiceServicer):
    """
    Cloud target M2 served over gRPC, exactly like SpecEdge's server side but for a
    *linear* draft chain and **stateless** (v1, see plan-chmura-m2 §3/§5).

    Per ``Validate``: tokenize ``prefix`` (the full committed context), append the
    draft chain ``input_ids``, run one causal forward of the target model, and
    return M2's greedy token at the last prefix position and after each draft token
    (``len(draft) + 1`` predictions). No KV cache is kept between calls, so there is
    nothing to desynchronize; persistent KV + incremental reorder is a v2 perf
    optimization, not a correctness requirement.

    When a result logger is configured the server writes ``server.jsonl`` rows in the
    SpecEdge schema (``target.forward_t``, ``target.server_end_to_end_t``,
    ``target.prefill``) so the existing ``metric/specedge.py`` can aggregate it.
    """

    def __init__(
        self,
        target_model: str,
        device: torch.device,
        dtype: torch.dtype,
        temperature: float = 0.0,
        log_results: bool = False,
    ) -> None:
        self._logger = log.get_logger()
        self._result_logger = log.get_result_logger() if log_results else None
        self._device = device

        self._temperature = temperature

        self._logger.info("Loading cloud target %s (%s) on %s", target_model, dtype, device)
        _trace(f"servicer: loading tokenizer for {target_model}")
        self._tokenizer = util.load_tokenizer(target_model)
        _trace(f"servicer: loading model {target_model} dtype={dtype} on {device}")
        self._model = util.load_model(name=target_model, device=device, dtype=dtype)
        _trace("servicer: model loaded, cloud target ready")
        self._logger.info("Cloud target ready")

    @torch.inference_mode()
    def Validate(self, request, context):
        with util.Timing(device=self._device, mode="sync") as e2e_t:
            draft_ids = util.decode(
                request.input_ids, device=self._device, dtype=torch.long, shape=(-1,)
            )
            gamma = draft_ids.numel()

            prefix_ids = self._tokenizer.encode(
                request.prefix, return_tensors="pt"
            ).to(self._device)[0]
            prefix_len = prefix_ids.numel()

            full_ids = torch.cat([prefix_ids, draft_ids], dim=-1).unsqueeze(0)

            with util.Timing(device=self._device, mode="event") as forward_t:
                logits = self._model(input_ids=full_ids).logits[0]

            # Predictions at the last prefix token and after each draft token:
            # positions [prefix_len - 1 .. prefix_len - 1 + gamma], i.e. gamma + 1.
            verify_logits = logits[prefix_len - 1 : prefix_len + gamma]
            selection = util.sampler_from_logits(
                verify_logits, temperature=self._temperature
            ).view(-1)

        if self._result_logger is not None:
            self._result_logger.log(
                {
                    "target": {
                        "forward_t": forward_t.elapsed,
                        "server_end_to_end_t": e2e_t.elapsed,
                        "prefill": 1 if request.prefill else 0,
                    }
                }
            )

        self._logger.debug(
            "Validate: prefix_len=%d gamma=%d -> %d predictions",
            prefix_len, gamma, selection.numel(),
        )

        return specedge_pb2.ValidateResponse(
            selection=util.encode(selection), prefill=1
        )

    def Sync(self, request, context):  # pragma: no cover - unused in v1
        return specedge_pb2.SyncResponse()


def serve(
    target_model: str,
    device: torch.device,
    dtype: torch.dtype,
    port: int = 8000,
    temperature: float = 0.0,
    max_workers: int = 4,
    result_path: Optional[str] = None,
    exp_name: str = "run",
) -> None:
    _trace(f"serve(): start target={target_model} device={device} dtype={dtype} port={port}")
    if result_path is not None:
        _trace(f"serve(): configuring logging under {Path(result_path) / exp_name}")
        log_config = log.get_default_log_config(Path(result_path) / exp_name, "server")
        log.configure_logging(log_config)

    logger = log.get_logger()

    _trace("serve(): constructing servicer (model load follows)")
    servicer = CloudTargetServicer(
        target_model=target_model,
        device=device,
        dtype=dtype,
        temperature=temperature,
        log_results=result_path is not None,
    )

    _trace("serve(): building gRPC server")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    specedge_pb2_grpc.add_SpecEdgeServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    _trace(f"serve(): starting gRPC server on port {port}")
    server.start()
    _trace(f"serve(): gRPC server listening on port {port}")
    logger.info("Cloud target M2 listening on port %d (inference loop ready)", port)
    server.wait_for_termination()
