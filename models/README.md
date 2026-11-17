# Frozen CAES scorers

These are the two unchanged, nine-feature scikit-learn artifacts used by the frozen dataset configurations. Each is 1,742 bytes; neither contains reader weights or a corpus.

| File | SHA-256 |
|---|---|
| `caes-2wiki-train.pkl` | `6d9ca0092b14b380baa2ca8051e4ae5bfb3dc6afdc56b2146781b47e02cc5fb0` |
| `caes-hotpotqa-transfer.pkl` | `c06ccecad3f3b74cb7171ae2733edf4bdac4af3813f4897145218bc45ca34d1d` |

The first scorer is trained on 2WikiMultiHopQA and used for HotpotQA; the second is trained on HotpotQA validation-derived trajectories and shared by 2WikiMultiHopQA and MuSiQue. No release-time training was performed. Sidecars describe artifact/runtime provenance; the original pickle bytes are unchanged.

**Security:** only load trusted pickle files. Do not treat a matching checksum supplied by an unknown third party as evidence that its pickle is safe.
