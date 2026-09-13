import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from projects.minimax_h3.modeling.transformer import model_sp


def _group_mean_attention(q, _k, v, *, q_lens, k_lens, **_kwargs):
    """Small deterministic stand-in for varlen FlashAttention."""
    outputs = []
    q_cursor = 0
    k_cursor = 0
    for q_len, k_len in zip(q_lens.tolist(), k_lens.tolist(), strict=True):
        q_chunk = q[q_cursor : q_cursor + q_len]
        v_chunk = v[k_cursor : k_cursor + k_len]
        outputs.append(v_chunk.mean(dim=0, keepdim=True).expand_as(q_chunk))
        q_cursor += q_len
        k_cursor += k_len
    return torch.cat(outputs, dim=0)


class DecoupledReferenceAttentionTest(unittest.TestCase):
    def test_reference_queries_are_isolated_but_active_queries_see_document(self):
        # Document 0: active [0,1,6,7], ref A [2,4), ref B [4,6).
        # Document 1: ref C [8,9), active [9].
        q = torch.zeros(10, 1, 1)
        k = torch.arange(10, dtype=torch.float32).view(10, 1, 1)
        v = k.clone()
        spans = torch.tensor([[4, 6], [8, 9], [2, 4]], dtype=torch.long)

        with mock.patch.object(
            model_sp,
            "_MINIMAX_H3_FLASH_ATTENTION",
            side_effect=_group_mean_attention,
        ):
            output = model_sp._decoupled_ref_attention_core(
                SimpleNamespace(softmax_scale=1.0),
                q,
                k,
                v,
                cu_seqlens_host=(0, 8, 10),
                asset_spans=spans,
            )

        expected = torch.tensor(
            [3.5, 3.5, 2.5, 2.5, 4.5, 4.5, 3.5, 3.5, 8.0, 8.5]
        ).view(10, 1, 1)
        torch.testing.assert_close(output, expected)

    def test_asset_span_cannot_cross_document_boundary(self):
        q = torch.zeros(10, 1, 1)
        with self.assertRaisesRegex(ValueError, "crosses packed-document boundary"):
            model_sp._decoupled_ref_attention_core(
                SimpleNamespace(softmax_scale=1.0),
                q,
                q,
                q,
                cu_seqlens_host=(0, 8, 10),
                asset_spans=torch.tensor([[7, 9]], dtype=torch.long),
            )

    def test_asset_spans_cannot_overlap(self):
        q = torch.zeros(8, 1, 1)
        with self.assertRaisesRegex(ValueError, "must be non-overlapping"):
            model_sp._decoupled_ref_attention_core(
                SimpleNamespace(softmax_scale=1.0),
                q,
                q,
                q,
                cu_seqlens_host=(0, 8),
                asset_spans=torch.tensor([[2, 5], [4, 6]], dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
