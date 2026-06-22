def _trace(msg):
    import os
    import sys
    import time
    print(
        f"[M1->M2 {time.strftime('%H:%M:%S')} pid={os.getpid()}] {msg}",
        file=sys.stderr,
        flush=True,
    )


_trace("client.py: importing grpc")
import grpc

_trace("client.py: importing torch")
import torch

_trace("client.py: importing log, util")
import log
import util

_trace("client.py: importing specedge_grpc (generated protobuf/grpc)")
from specedge_grpc import specedge_pb2, specedge_pb2_grpc

_trace("client.py: cloud client module imports complete")


class CloudTargetClient:
    """
    Synchronous gRPC client for the cloud target M2.

    Mirrors SpecEdge's ``GrpcClientController`` wire contract (same ``.proto``,
    same ``encode``/``decode``, same ``Validate`` RPC), but the link is M1 -> M2
    and the proposal is a *linear* draft chain, not a tree (see plan-chmura-m2 §2).

    v1 is stateless on the server side: every call carries the full committed
    ``prefix`` plus the draft chain ``draft_ids``; the server prefills the prefix
    and forwards the chain from scratch. The response is M2's greedy prediction at
    the last prefix position and after each draft token: ``len(draft_ids) + 1``
    token ids. Acceptance is computed by the caller (``verify``'s consumer).
    """

    def __init__(self, host: str) -> None:
        self._logger = log.get_logger()
        _trace(f"CloudTargetClient: opening gRPC channel to {host}")
        self._host = host
        self._channel = grpc.insecure_channel(host)
        self._stub = specedge_pb2_grpc.SpecEdgeServiceStub(self._channel)
        _trace(f"CloudTargetClient: channel + stub ready for {host}")

    def verify(
        self, prefix: str, draft_ids: torch.Tensor, prefill: bool = True
    ) -> torch.Tensor:
        if draft_ids.ndim != 1:
            raise ValueError(f"draft_ids must be 1-D, got shape {tuple(draft_ids.shape)}")
        if draft_ids.numel() == 0:
            raise ValueError("draft_ids is empty; nothing to verify")

        request = specedge_pb2.ValidateRequest(
            client_idx=0,
            req_idx=0,
            input_ids=util.encode(draft_ids.to(torch.long)),
            position_ids=b"",
            cache_seq_indices=b"",
            parent_indices=b"",
            attention_mask=b"",
            prefill=prefill,
            prefix=prefix,
        )

        self._logger.debug(
            "verify: prefix_chars=%d gamma=%d prefill=%s -> Validate RPC",
            len(prefix), draft_ids.numel(), prefill,
        )
        resp = self._stub.Validate(request)

        selection = util.decode(
            resp.selection,
            device=torch.device("cpu"),
            dtype=torch.long,
            shape=(-1,),
        )

        expected = draft_ids.numel() + 1
        if selection.numel() != expected:
            raise RuntimeError(
                f"Cloud target returned {selection.numel()} predictions, "
                f"expected {expected} (len(draft)+1). Protocol mismatch."
            )
        return selection

    def close(self) -> None:
        self._channel.close()
