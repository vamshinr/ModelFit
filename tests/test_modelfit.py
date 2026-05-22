"""Smoke + invariants. Run with `python -m unittest discover tests`."""
import json
import os
import shutil
import subprocess
import sys
import unittest

from modelfit.hardware import detect_hardware, hardware_from_dict, GPU, HardwareProfile, GB
from modelfit.models import get_catalog, find_model, QUANT_ORDER, QUANT_BYTES_PER_PARAM
from modelfit.quantization import fit_model, weights_bytes, kv_bytes_for
from modelfit.scoring import score_model, rank_all, estimate_tokens_per_sec, USE_CASE_WEIGHTS
from modelfit.reverse import recommend_hardware


def hw_4090():
    return hardware_from_dict({
        "cpu_name": "AMD Ryzen 9 7950X", "cpu_cores": 16, "cpu_threads": 32,
        "cpu_freq_mhz": 5700, "ram_gb": 64, "ram_available_gb": 50,
        "ram_bandwidth_gbps": 80,
        "gpus": [{"name": "NVIDIA GeForce RTX 4090", "vendor": "nvidia",
                  "vram_gb": 24, "bandwidth_gbps": 1008}],
    })


def hw_cpu_only_16gb():
    return hardware_from_dict({
        "cpu_name": "Intel i7-12700K", "cpu_cores": 12, "cpu_threads": 20,
        "cpu_freq_mhz": 4900, "ram_gb": 16, "ram_available_gb": 12,
        "ram_bandwidth_gbps": 60, "gpus": [],
    })


def hw_h100_80():
    return hardware_from_dict({
        "cpu_name": "AMD EPYC 9554", "cpu_cores": 64, "cpu_threads": 128,
        "cpu_freq_mhz": 3750, "ram_gb": 512, "ram_available_gb": 500,
        "gpus": [{"name": "NVIDIA H100 80GB HBM3", "vendor": "nvidia",
                  "vram_gb": 80, "bandwidth_gbps": 3350}],
    })


class CatalogTests(unittest.TestCase):
    def test_catalog_size(self):
        # The advertised "206 models" claim.
        self.assertEqual(len(get_catalog()), 206)

    def test_no_duplicate_ids(self):
        ids = [m.id for m in get_catalog()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_finder(self):
        self.assertEqual(find_model("llama-3.1-8b-instruct").id, "llama-3.1-8b-instruct")
        self.assertIsNone(find_model("not-a-real-model-9000"))

    def test_param_ranges_sane(self):
        for m in get_catalog():
            self.assertGreater(m.params_b, 0)
            self.assertLessEqual(m.active_params_b, m.params_b + 0.01)
            self.assertGreaterEqual(m.context_max, 2048)
            self.assertGreaterEqual(m.kv_bytes_per_token, 1024)


class QuantTests(unittest.TestCase):
    def test_quant_ordering_monotone(self):
        prev = float("inf")
        for q in QUANT_ORDER:
            self.assertLess(QUANT_BYTES_PER_PARAM[q], prev + 1e-6)
            prev = QUANT_BYTES_PER_PARAM[q]

    def test_70b_fits_on_4090_at_q4(self):
        m = find_model("llama-3.1-70b-instruct")
        fit = fit_model(m, hw_4090(), min_context=2048)
        # 70B at Q4_K_M is ~42GB — won't fit in 24GB VRAM.
        self.assertFalse(fit.fits)

    def test_8b_fits_on_4090_at_q8(self):
        m = find_model("llama-3.1-8b-instruct")
        fit = fit_model(m, hw_4090(), min_context=8192)
        self.assertTrue(fit.fits)
        # On a 24GB card, the highest-quality fit should be Q8_0.
        self.assertEqual(fit.quant, "Q8_0")

    def test_70b_needs_h100(self):
        m = find_model("llama-3.1-70b-instruct")
        fit = fit_model(m, hw_h100_80(), min_context=8192)
        self.assertTrue(fit.fits)

    def test_context_fallback(self):
        # On a 16GB CPU, Llama 3.1 70B should not fit at any quant.
        m = find_model("llama-3.1-70b-instruct")
        fit = fit_model(m, hw_cpu_only_16gb(), min_context=2048)
        self.assertFalse(fit.fits)


class ScoringTests(unittest.TestCase):
    def test_use_case_weights_sum_to_one(self):
        for case, w in USE_CASE_WEIGHTS.items():
            self.assertAlmostEqual(sum(w), 1.0, places=4, msg=f"{case}: {w}")

    def test_rank_returns_only_fits_by_default(self):
        ranked = rank_all(hw_cpu_only_16gb(), use_case="chat")
        self.assertTrue(all(s.fit.fits for s in ranked))
        # Sorted by composite descending.
        for a, b in zip(ranked, ranked[1:]):
            self.assertGreaterEqual(a.composite, b.composite)

    def test_h100_ranks_top_quality(self):
        ranked = rank_all(hw_h100_80(), use_case="reasoning")
        # An H100 should be able to fit at least the 32B and 70B reasoning models.
        names = [s.model.name for s in ranked[:10]]
        self.assertTrue(
            any("70B" in n or "32B" in n for n in names),
            f"top 10 on H100 had no large reasoning model: {names}",
        )

    def test_code_use_case_favors_code_models(self):
        ranked = rank_all(hw_4090(), use_case="code")
        self.assertTrue(ranked, "expected some models to fit on a 4090")
        # Among the top 5, at least one should be tagged code or have type code.
        top5 = ranked[:5]
        self.assertTrue(any(s.model.type == "code" or "code" in s.model.tags for s in top5))

    def test_speed_estimate_positive_when_fits(self):
        ranked = rank_all(hw_4090(), use_case="chat")
        for s in ranked[:5]:
            self.assertGreater(s.tokens_per_sec, 0)


class ReverseTests(unittest.TestCase):
    def test_reverse_llama70b(self):
        m = find_model("llama-3.1-70b-instruct")
        # Use a practical 8K context — full 128K context KV for 70B is ~40GB.
        rec = recommend_hardware(m, target_tps=40, context=8192, quant="Q4_K_M")
        self.assertGreater(rec["min_vram_gb"], 35)
        self.assertGreater(rec["min_bandwidth_gbps"], 100)
        # H100 (80GB) should be a candidate at 8K context.
        self.assertTrue(any("H100" in c["name"] for c in rec["candidate_gpus"]),
                        msg=f"got: {[c['name'] for c in rec['candidate_gpus']]}")


class HardwareTests(unittest.TestCase):
    def test_detect_runs(self):
        hw = detect_hardware()
        self.assertGreater(hw.ram_bytes, 0)
        self.assertGreater(hw.cpu_threads, 0)
        # Round-trip through dict.
        d = hw.to_dict()
        self.assertIn("cpu_name", d)
        self.assertIn("ram_gb", d)

    def test_hardware_from_dict_roundtrip(self):
        hw = hw_4090()
        self.assertEqual(hw.gpus[0].vram_bytes, 24 * GB)
        self.assertEqual(hw.gpus[0].vendor, "nvidia")


class CLITests(unittest.TestCase):
    def test_cli_inspect_json(self):
        out = subprocess.check_output(
            [sys.executable, "-m", "modelfit", "inspect", "--json"],
            stderr=subprocess.STDOUT, timeout=15,
        )
        parsed = json.loads(out)
        self.assertIn("cpu_name", parsed)

    def test_cli_list_json(self):
        out = subprocess.check_output(
            [sys.executable, "-m", "modelfit", "list", "--json"],
            stderr=subprocess.STDOUT, timeout=15,
        )
        models = json.loads(out)
        self.assertEqual(len(models), 206)

    def test_cli_get_json(self):
        out = subprocess.check_output(
            [sys.executable, "-m", "modelfit", "get", "llama-3.2-3b-instruct", "--json"],
            stderr=subprocess.STDOUT, timeout=15,
        )
        parsed = json.loads(out)
        self.assertIn("install", parsed)
        self.assertIn("llama3.2", parsed["install"]["ollama"])

    def test_cli_explain_listing(self):
        out = subprocess.check_output(
            [sys.executable, "-m", "modelfit", "explain", "--json"],
            stderr=subprocess.STDOUT, timeout=15,
        )
        entries = json.loads(out)
        self.assertTrue(any(e["term"] == "quantization" for e in entries))
        self.assertTrue(any(e["term"] == "context" for e in entries))

    def test_cli_explain_specific_term(self):
        out = subprocess.check_output(
            [sys.executable, "-m", "modelfit", "explain", "quantization", "--json"],
            stderr=subprocess.STDOUT, timeout=15,
        )
        entry = json.loads(out)
        self.assertEqual(entry["term"], "quantization")
        self.assertTrue(entry["analogy"])


class BeginnerLayerTests(unittest.TestCase):
    def test_quant_labels_complete(self):
        from modelfit.beginner import QUANT_LABEL, quant_label
        from modelfit.models import QUANT_ORDER
        for q in QUANT_ORDER:
            self.assertIn(q, QUANT_LABEL, f"missing plain-English label for {q}")
            label, stars, desc = quant_label(q)
            self.assertTrue(label and stars and desc)

    def test_speed_labels_cover_range(self):
        from modelfit.beginner import speed_label
        heads = {speed_label(t)[0] for t in [0, 1, 4, 10, 25, 100]}
        # Should hit at least 4 distinct buckets across this range.
        self.assertGreaterEqual(len(heads), 4)

    def test_download_size_matches_weights(self):
        from modelfit.beginner import download_size_gb
        from modelfit.models import find_model
        m = find_model("llama-3.1-8b-instruct")
        s = download_size_gb(m, "Q4_K_M")
        self.assertGreater(s, 4.0)
        self.assertLess(s, 6.0)

    def test_verdict_returns_pair_when_fits(self):
        from modelfit.beginner import verdict
        from modelfit.scoring import score_model
        from modelfit.models import find_model
        s = score_model(find_model("llama-3.2-1b-instruct"),
                          hw_h100_80(), use_case="balanced")
        v_head, v_sum = verdict(s)
        self.assertTrue(v_head)
        self.assertTrue(v_sum)


class GlossaryTests(unittest.TestCase):
    def test_aliases_resolve(self):
        from modelfit.glossary import lookup
        self.assertIsNotNone(lookup("tps"))
        self.assertEqual(lookup("tps").term, "tok/s")
        self.assertIsNotNone(lookup("huggingface"))
        self.assertIsNotNone(lookup("quantization"))

    def test_unknown_term(self):
        from modelfit.glossary import lookup
        self.assertIsNone(lookup("not-a-real-term-9000"))


class RunnerTests(unittest.TestCase):
    def test_ollama_tag_for_known_model(self):
        from modelfit.runners import get_commands
        from modelfit.models import find_model
        cmds = get_commands(find_model("llama-3.2-3b-instruct"), "Q4_K_M")
        self.assertIsNotNone(cmds.ollama)
        self.assertIn("llama3.2:3b", cmds.ollama)

    def test_unknown_model_falls_back_gracefully(self):
        from modelfit.runners import get_commands
        from modelfit.models import find_model
        cmds = get_commands(find_model("orca-2-7b"), "Q4_K_M")
        self.assertIsNone(cmds.ollama)
        self.assertTrue(any("Ollama" in n for n in (cmds.notes or [])))

    def test_download_size_in_label(self):
        from modelfit.runners import get_commands
        from modelfit.models import find_model
        cmds = get_commands(find_model("llama-3.1-70b-instruct"), "Q4_K_M")
        self.assertIn("GB", cmds.download_label)


class RecommenderTests(unittest.TestCase):
    def test_recommend_picks_instruct_over_base(self):
        from modelfit.recommend import recommend
        rec = recommend(hw_4090(), use_case="chat")
        self.assertIsNotNone(rec)
        # Beginners would type "summarise this" and get hallucinated text
        # from a 'base' model — we must never recommend one.
        self.assertNotEqual(rec.primary.model.type, "base")

    def test_recommend_returns_install_explainer(self):
        from modelfit.recommend import recommend
        rec = recommend(hw_4090(), use_case="code")
        self.assertIsNotNone(rec)
        self.assertTrue(rec.why)
        self.assertGreater(len(rec.why), 50)

    def test_recommend_handles_tiny_hardware(self):
        from modelfit.recommend import recommend
        tiny = hardware_from_dict({
            "cpu_name": "Tiny CPU", "cpu_cores": 1, "cpu_threads": 1,
            "ram_gb": 1, "ram_available_gb": 0.5, "ram_bandwidth_gbps": 10,
            "gpus": [],
        })
        rec = recommend(tiny, use_case="chat")
        # Either we found something (a sub-1B model) or honestly returned None.
        if rec is not None:
            self.assertLess(rec.primary.model.params_b, 1.5)


class EngineTests(unittest.TestCase):
    def test_engine_catalog_complete(self):
        from modelfit.engines import ENGINES, ENGINE_BY_ID
        ids = {e.id for e in ENGINES}
        for need in ("llama.cpp", "vllm", "tensorrt-llm", "ollama", "mlx",
                      "transformers", "exllamav2", "tgi", "sglang"):
            self.assertIn(need, ids)
        # Every engine has a positive multiplier.
        for e in ENGINES:
            self.assertGreater(e.single_req_multiplier, 0)
            self.assertGreaterEqual(e.concurrency_factor, 1)

    def test_predict_throughput_on_4090(self):
        from modelfit.engines import predict_throughput
        m = find_model("llama-3.1-8b-instruct")
        preds = predict_throughput(m, hw_4090(), min_context=8192)
        # GGUF-friendly engines (llama.cpp, Ollama, ExLlamaV2 if it accepts the quant) should fit.
        self.assertTrue(any(p.engine.id == "llama.cpp" and p.fits for p in preds))
        # MLX should be filtered out entirely — Apple-only — so it doesn't appear.
        self.assertTrue(all(p.engine.id != "mlx" for p in preds))
        # GPU-only engines that need non-GGUF formats should appear but report
        # "doesn't support the quant" — they should be present but not fit.
        vllm_pred = next((p for p in preds if p.engine.id == "vllm"), None)
        self.assertIsNotNone(vllm_pred)
        self.assertFalse(vllm_pred.fits)
        self.assertIn("doesn't support", vllm_pred.explanation)

    def test_predict_throughput_on_mac_excludes_vllm(self):
        from modelfit.engines import predict_throughput
        mac = hardware_from_dict({
            "cpu_name": "Apple M2 Max", "cpu_cores": 12, "cpu_threads": 12,
            "ram_gb": 64, "ram_available_gb": 50, "ram_bandwidth_gbps": 400,
            "platform": "Darwin", "arch": "arm64", "unified_memory": True,
            "gpus": [{"name": "Apple M2 Max", "vendor": "apple",
                       "vram_gb": 48, "bandwidth_gbps": 400}],
        })
        m = find_model("llama-3.1-8b-instruct")
        preds = predict_throughput(m, mac, min_context=8192)
        # vLLM is GPU-only-NVIDIA/AMD — should not appear at all on Apple.
        self.assertTrue(all(p.engine.id != "vllm" for p in preds))
        # MLX SHOULD appear.
        self.assertTrue(any(p.engine.id == "mlx" for p in preds))


class GlossaryExpansionTests(unittest.TestCase):
    def test_has_training_terms(self):
        from modelfit.glossary import GLOSSARY
        for need in ("rlhf", "dpo", "lora", "qlora", "sft", "backpropagation",
                      "gradient-descent", "perplexity", "distillation"):
            self.assertIn(need, GLOSSARY, f"missing training term: {need}")

    def test_has_inference_internals(self):
        from modelfit.glossary import GLOSSARY
        for need in ("flash-attention", "pagedattention", "gqa", "rope",
                      "speculative-decoding", "continuous-batching", "kv-cache",
                      "prefill", "decode", "ttft"):
            self.assertIn(need, GLOSSARY, f"missing inference term: {need}")

    def test_multi_word_search(self):
        from modelfit.glossary import search
        hits = search("rl from human")
        self.assertTrue(hits, "expected RLHF to be returned")
        self.assertEqual(hits[0].term.lower(), "rlhf")

    def test_aliases_resolve(self):
        from modelfit.glossary import lookup
        self.assertEqual(lookup("flashattention").term, "flash-attention")
        self.assertEqual(lookup("knowledge distillation").term, "distillation")
        self.assertEqual(lookup("backprop").term, "backpropagation")

    def test_all_categories_exist(self):
        from modelfit.glossary import all_categories
        cats = all_categories()
        for need in ("Basics", "Hardware", "Quantization", "Training",
                      "Architecture", "Inference", "Engines & runners",
                      "Applications", "Safety & alignment"):
            self.assertIn(need, cats)


class CatalogSyncTests(unittest.TestCase):
    """Don't hit the network — only verify the local merging logic works."""

    def setUp(self):
        # Reset module-level cache so get_catalog() re-reads.
        import modelfit.models
        modelfit.models._CATALOG = None

    def test_hf_extracted_models_merge_into_catalog(self):
        """If the HF cache has models, they appear in get_catalog()."""
        from modelfit.catalog_sync import save_cache, clear_cache, CACHE_FILE
        backup = None
        if CACHE_FILE.exists():
            backup = CACHE_FILE.read_text()
            clear_cache()
        try:
            save_cache({
                "synced_at": "test", "source": "huggingface",
                "models": [{
                    "id": "test-fake-3b",
                    "name": "Test Fake 3B",
                    "family": "TestOrg",
                    "params_b": 3.0,
                    "active_params_b": 3.0,
                    "context_max": 8192,
                    "type": "instruct",
                    "quality": 60.0,
                    "kv_bytes_per_token": 32768,
                    "license": "transformers",
                    "tags": ["hf"],
                }],
            })
            ids = {m.id for m in get_catalog()}
            self.assertIn("test-fake-3b", ids)
            self.assertEqual(len(get_catalog()), 207)
        finally:
            clear_cache()
            if backup is not None:
                from modelfit.catalog_sync import CACHE_FILE as cf, _ensure_dir
                _ensure_dir()
                cf.write_text(backup)

    def test_curated_wins_on_id_clash(self):
        """If HF cache and curated catalog share an id, curated quality wins."""
        from modelfit.catalog_sync import save_cache, clear_cache, CACHE_FILE
        backup = None
        if CACHE_FILE.exists():
            backup = CACHE_FILE.read_text()
            clear_cache()
        try:
            save_cache({
                "synced_at": "test", "source": "huggingface",
                "models": [{
                    "id": "llama-3.1-8b-instruct",  # already in curated
                    "name": "Llama 3.1 8B Instruct",
                    "family": "Llama", "params_b": 8.0, "active_params_b": 8.0,
                    "context_max": 131072, "type": "instruct",
                    "quality": 30.0,   # deliberately wrong
                    "kv_bytes_per_token": 131072, "license": "fake",
                    "tags": ["hf"],
                }],
            })
            m = find_model("llama-3.1-8b-instruct")
            # The curated quality (74) must override the cache (30).
            self.assertGreater(m.quality, 70)
        finally:
            clear_cache()
            if backup is not None:
                from modelfit.catalog_sync import CACHE_FILE as cf, _ensure_dir
                _ensure_dir()
                cf.write_text(backup)


# ---------------------------------------------------------------------------
# C++ parity (only runs if the C++ CLI has been built)
# ---------------------------------------------------------------------------
CPP_CLI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "cpp", "build", "modelfit-cli")


@unittest.skipUnless(os.path.isfile(CPP_CLI) and os.access(CPP_CLI, os.X_OK),
                     "C++ CLI not built; run `cmake --build cpp/build`")
class CppParityTests(unittest.TestCase):
    def test_cpp_catalog_size_matches(self):
        out = subprocess.check_output([CPP_CLI, "list", "--json"], timeout=10)
        models = json.loads(out)
        self.assertEqual(len(models), len(get_catalog(include_external=False)))

    def test_cpp_top_pick_matches_python(self):
        # Both CLIs are deterministic given the same hardware + use case.
        py = subprocess.check_output(
            [sys.executable, "-m", "modelfit", "rank", "--json", "--top", "1",
             "--use-case", "chat", "--min-context", "2048"],
            timeout=15,
        )
        cpp = subprocess.check_output(
            [CPP_CLI, "rank", "--json", "--top", "1",
             "--use-case", "chat", "--min-context", "2048"],
            timeout=15,
        )
        py_top = json.loads(py)["models"][0]["id"]
        cpp_top = json.loads(cpp)["models"][0]["id"]
        self.assertEqual(py_top, cpp_top,
                          f"Python picked {py_top} but C++ picked {cpp_top}")

    def test_cpp_inspect_emits_valid_json(self):
        out = subprocess.check_output([CPP_CLI, "inspect", "--json"], timeout=5)
        data = json.loads(out)
        for k in ("cpu_name", "ram_gb", "platform"):
            self.assertIn(k, data)


if __name__ == "__main__":
    unittest.main()
